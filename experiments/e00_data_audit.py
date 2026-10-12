"""E00: audit the development pool before modelling."""

import re

import pandas as pd

from common import PHRESH_DIR, load_dev, load_pool, load_tranco, save

raw_rows = len(pd.read_parquet(PHRESH_DIR / "train_full.parquet", columns=["url"]))
pool = load_pool()
dev = load_dev()
tranco = load_tranco()
top10k = set(tranco.loc[tranco["rank"] <= 10_000, "domain"])
top100k = set(tranco["domain"])

result = {"raw_train_rows": raw_rows, "pool_rows_after_dedup": len(pool)}

result["partitions"] = (
    pool.groupby("partition")["label"].value_counts().unstack().to_dict(orient="index")
)

result["homepage_like_share_by_label"] = pool.groupby("label")["homepage_like"].mean().to_dict()

# Concentration: how many rows the largest registered domains contribute
domain_counts = pool.groupby(["label", "domain"]).size()
result["top_domains"] = {
    label: domain_counts[label].sort_values(ascending=False).head(15).to_dict()
    for label in ["phish", "benign"]
}
result["unique_domains_by_label"] = pool.groupby("label")["domain"].nunique().to_dict()

# Popular domains in the pool
pool["in_top10k"] = pool["domain"].isin(top10k)
pool["in_top100k"] = pool["domain"].isin(top100k)
result["share_in_tranco_top10k_by_label"] = pool.groupby("label")["in_top10k"].mean().to_dict()
result["share_in_tranco_top100k_by_label"] = pool.groupby("label")["in_top100k"].mean().to_dict()

exact_home_of_popular = pool["homepage_like"] & (pool["host"] == pool["domain"]) & pool["in_top100k"]
result["bare_homepage_of_tranco_top100k_by_label"] = (
    pool.loc[exact_home_of_popular, "label"].value_counts().to_dict()
)
result["bare_homepage_of_tranco_top100k_phish_examples"] = (
    pool.loc[exact_home_of_popular & (pool["label"] == "phish"), "url"].head(20).tolist()
)

# Benign homepages by popularity
benign_home = pool.loc[(pool["label"] == "benign") & pool["homepage_like"]]
result["benign_homepages_total"] = len(benign_home)
result["benign_homepages_in_top10k"] = int(benign_home["in_top10k"].sum())

# Phishing targets
phish = pool.loc[pool["label"] == "phish"].copy()
phish["has_target"] = ~phish["target"].isin(["None", "", None])
result["phish_with_target_share"] = float(phish["has_target"].mean())
result["top_targets"] = phish.loc[phish["has_target"], "target"].value_counts().head(25).to_dict()


def brand_token(target):
    return re.sub(r"[^a-z0-9]", "", str(target).lower())


phish["brand_in_host"] = [
    bool(token) and len(token) >= 4 and token in host
    for token, host in zip(phish["target"].map(brand_token), phish["host"])
]
phish["brand_in_text"] = [
    bool(token) and len(token) >= 4 and token in text
    for token, text in zip(phish["target"].map(brand_token), phish["text"])
]
targeted = phish.loc[phish["has_target"]]
result["targeted_phish_brand_token_in_host_share"] = float(targeted["brand_in_host"].mean())
result["targeted_phish_brand_token_in_text_share"] = float(targeted["brand_in_text"].mean())

# Time
result["date_range_by_partition"] = {
    name: [str(group["date"].min().date()), str(group["date"].max().date())]
    for name, group in pool.groupby("partition")
}
result["pool_rows_by_quarter"] = (
    pool.groupby([pool["date"].dt.to_period("Q").astype(str), "label"]).size().unstack().to_dict(orient="index")
)

# Dev benchmark relation to the pool
fit_domains = set(pool.loc[pool["partition"] == "fit", "domain"])
result["dev_rows_with_domain_in_fit"] = float(dev["domain"].isin(fit_domains).mean())
result["dev_homepage_like_share_by_label"] = dev.groupby("label")["homepage_like"].mean().to_dict()
result["dev_share_in_tranco_top10k_by_label"] = dev.assign(
    t=dev["domain"].isin(top10k)
).groupby("label")["t"].mean().to_dict()

print(save("e00_data_audit", result).read_text())
