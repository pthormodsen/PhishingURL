"""Shared data loading, splitting and evaluation for the experiments.

Protocol
--------
- Development pool: the full PhreshPhish train split, deduplicated by
  model text. Texts that occur with both labels are dropped, and URLs that
  Python cannot parse are dropped (the backend rejects them with HTTP 400).
- Grouping key ("site"): the registrable domain from the public suffix
  list, including its private section. The private section marks domains
  such as github.io or pages.dev where every subdomain has an independent
  owner, so abc.github.io and xyz.github.io are different sites, while
  mail.example.com and www.example.com are the same site.
- Every row is assigned to a partition by a stable hash of its site:
  70% fit, 15% val, 15% holdout. URLs of one site never cross partitions.
  Tranco domains use the same hash, so external examples follow the split.
- A few sites contribute thousands of rows (for example google.com phishing
  in val and wikipedia.org in holdout). Evaluation therefore uses
  site-capped sets with at most CAP rows per site and label, and also
  reports the uncapped sets.
- val: tuning and threshold choice. holdout: finalist comparison only.
  dev benchmark: 10k rows from the later official test split (temporal
  check, already inspected earlier). The final test file is not used here.
"""

import hashlib
import html
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.features import url_to_model_text  # noqa: E402

PHRESH_DIR = ROOT / "data" / "phreshphish"
EXTERNAL_DIR = ROOT / "data" / "external"
RESULTS_DIR = ROOT / "results" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {"phish": 0, "benign": 1}
CAP = 20
TARGET_FAR = 0.01

# Bundled public suffix list snapshot: no network access, stable results
_icann = tldextract.TLDExtract(suffix_list_urls=())
_private = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def split_model_text(text):
    for position, character in enumerate(text):
        if character in "/?#":
            return text[:position], text[position:]
    return text, ""


@lru_cache(maxsize=None)
def registered_domain(hostname):
    return _icann(hostname).top_domain_under_public_suffix or hostname


@lru_cache(maxsize=None)
def site_of(hostname):
    return _private(hostname).top_domain_under_public_suffix or hostname


def is_homepage_like(url):
    url = url.strip()
    parsed = urlsplit(url if "://" in url else "https://" + url)
    return len(parsed.path) <= 1 and len(parsed.query) == 0


def partition_of(key):
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "fit"
    if bucket < 85:
        return "val"
    return "holdout"


def canonical_text(text):
    """Preprocessing v2: an empty path is equivalent to "/" (RFC 3986 6.2.3).

    "netflix.com" and "netflix.com/" request the same resource. In the
    training data almost every benign homepage ends with "/", so without
    this step a typed bare domain looks unusual to the model.
    """
    host, rest = split_model_text(text)
    if not rest.startswith("/"):
        rest = "/" + rest
    return host + rest


def safe_model_text(url):
    try:
        return url_to_model_text(url)
    except ValueError:
        return None


def normalise_target(target):
    target = html.unescape(str(target)).strip().lower()
    return "none" if target in ("", "none", "nan") else target


PREPROCESSING = {"v1": lambda text: text, "v2": canonical_text}


def add_columns(frame, preprocessing="v1"):
    frame = frame.copy()
    frame["text"] = frame["url"].map(safe_model_text)
    unparseable = frame["text"].isna()
    if unparseable.any():
        print(f"Dropping {int(unparseable.sum())} unparseable URLs")
        frame = frame.loc[~unparseable]
    frame["text"] = frame["text"].map(PREPROCESSING[preprocessing])
    parts = frame["text"].map(split_model_text)
    frame["host"] = parts.str[0]
    frame["rest"] = parts.str[1]
    frame["domain"] = frame["host"].map(registered_domain)
    frame["site"] = frame["host"].map(site_of)
    frame["homepage_like"] = frame["url"].map(is_homepage_like)
    if "label" in frame:
        frame["y"] = frame["label"].map(LABELS).astype(int)
    if "target" in frame:
        frame["target"] = frame["target"].map(normalise_target)
    return frame.reset_index(drop=True)


def load_pool(preprocessing="v1"):
    cache = PHRESH_DIR / f"pool_cache_sites_{preprocessing}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    pool = add_columns(pd.read_parquet(PHRESH_DIR / "train_full.parquet"), preprocessing)
    labels_per_text = pool.groupby("text")["y"].nunique()
    conflicting = labels_per_text.index[labels_per_text > 1]
    print(f"Dropping {len(conflicting)} texts with conflicting labels")
    pool = pool.loc[~pool["text"].isin(conflicting)]
    pool = pool.drop_duplicates("text").reset_index(drop=True)
    pool["partition"] = pool["site"].map(partition_of)
    pool.to_parquet(cache, index=False)
    return pool


def load_dev(preprocessing="v1"):
    return add_columns(pd.read_parquet(PHRESH_DIR / "dev_benchmark_meta.parquet"), preprocessing)


def load_tranco(top=100_000):
    tranco = pd.read_parquet(EXTERNAL_DIR / "tranco_Y83YG_top100k.parquet")
    tranco = tranco.loc[tranco["rank"] <= top].copy()
    tranco["site"] = tranco["domain"].map(site_of)
    tranco = tranco.drop_duplicates("site")
    tranco["partition"] = tranco["site"].map(partition_of)
    return tranco.reset_index(drop=True)


def homepage_frame(urls, preprocessing="v1"):
    return add_columns(pd.DataFrame({"url": list(urls)}), preprocessing)


def cap_per_site(frame, cap=CAP, seed=0):
    shuffled = frame.sample(frac=1.0, random_state=seed)
    return shuffled.groupby(["site", "y"], sort=False).head(cap).sort_index()


# ---------------------------------------------------------------- metrics

def threshold_at_far(scores, y, far=TARGET_FAR):
    """Lowest threshold whose benign false alarm rate is at most `far`."""
    benign = np.sort(scores[y == 1])[::-1]
    allowed = int(np.floor(far * len(benign) + 1e-9))
    return float(np.nextafter(benign[allowed], np.inf))


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return [round(float(c - m), 4), round(float(c + m), 4)]


def rate(hit, mask):
    k, n = int((hit & mask).sum()), int(mask.sum())
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def precision_at_base_rate(recall, far, base_rate):
    tp = recall * base_rate
    fp = far * (1 - base_rate)
    return tp / (tp + fp) if tp + fp else float("nan")


def evaluate(frame, scores, threshold):
    y = frame["y"].to_numpy()
    home = frame["homepage_like"].to_numpy()
    flagged = scores >= threshold
    phish, benign = y == 0, y == 1

    tp = int((flagged & phish).sum())
    fp = int((flagged & benign).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / phish.sum()
    far = fp / benign.sum()

    return {
        "n": len(frame),
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if tp else 0.0,
        "benign_far": round(far, 4),
        "pr_auc": round(float(average_precision_score(phish, scores)), 4),
        "roc_auc": round(float(roc_auc_score(phish, scores)), 4),
        "benign_homepage_far": rate(flagged, benign & home),
        "phish_homepage_recall": rate(flagged, phish & home),
        "phish_other_recall": rate(flagged, phish & ~home),
        "precision_at_1pct_prevalence": round(precision_at_base_rate(recall, far, 0.01), 4),
    }


def brand_slices(frame, scores, threshold, top=8):
    phish = frame.loc[frame["y"] == 0]
    flagged = pd.Series(scores >= threshold, index=frame.index).loc[phish.index]
    out = {}
    named = phish.loc[phish["target"] != "other", "target"]
    for target in named.value_counts().head(top).index:
        mask = (phish["target"] == target).to_numpy()
        out[target] = rate(flagged.to_numpy(), mask)
    out["other (no named brand)"] = rate(flagged.to_numpy(), (phish["target"] == "other").to_numpy())

    tokens = phish["target"].map(lambda t: re.sub(r"[^a-z0-9]", "", t))
    in_host = np.array([len(t) >= 4 and t in h for t, h in zip(tokens, phish["host"])])
    out["brand name in hostname"] = rate(flagged.to_numpy(), in_host)
    return out


class EvalSets:
    """Fixed evaluation sets shared by every experiment."""

    def __init__(self, include_holdout=False, preprocessing="v1"):
        self.preprocessing = preprocessing
        pool = load_pool(preprocessing)
        self.pool = pool
        self.fit = pool.loc[pool["partition"] == "fit"].reset_index(drop=True)
        self.sets = {}
        parts = ["val"] + (["holdout"] if include_holdout else [])
        for name in parts:
            rows = pool.loc[pool["partition"] == name].reset_index(drop=True)
            self.sets[f"{name}_capped"] = cap_per_site(rows).reset_index(drop=True)
            self.sets[name] = rows

        dev = load_dev(preprocessing)
        fit_sites = set(self.fit["site"])
        self.sets["dev"] = dev
        self.sets["dev_unseen_site"] = dev.loc[~dev["site"].isin(fit_sites)].reset_index(drop=True)

        tranco = load_tranco()
        self.tranco = tranco
        self.homepages = {}
        for name in parts:
            subset = tranco.loc[tranco["partition"] == name]
            frame = homepage_frame(subset["domain"], preprocessing)
            frame["rank"] = subset["rank"].to_numpy()
            self.homepages[name] = frame

    def report(self, name, score_fn, extra=None):
        scores = {key: score_fn(frame) for key, frame in self.sets.items()}
        val = self.sets["val_capped"]
        threshold = threshold_at_far(scores["val_capped"], val["y"].to_numpy())

        result = {"name": name, "threshold_rule": f"benign FAR {TARGET_FAR} on val_capped", **(extra or {})}
        for rule, t in [("far1", threshold), ("t05", 0.5)]:
            block = {key: evaluate(frame, scores[key], t) for key, frame in self.sets.items()}
            for part, frame in self.homepages.items():
                home_scores = score_fn(frame)
                flagged = home_scores >= t
                ranks = frame["rank"].to_numpy()
                block[f"tranco_homepages_{part}"] = {
                    "top_1k": rate(flagged, ranks <= 1_000),
                    "top_10k": rate(flagged, ranks <= 10_000),
                    "10k_100k": rate(flagged, ranks > 10_000),
                }
            block["brands_val"] = brand_slices(self.sets["val"], scores["val"], t)
            block["brands_dev"] = brand_slices(self.sets["dev"], scores["dev"], t)
            result[rule] = block
        save(name, result)
        return result


def summary_row(result, rule="far1"):
    block = result[rule]
    vc, dev = block["val_capped"], block["dev"]
    row = {
        "model": result["name"],
        "val_capped PR-AUC": vc["pr_auc"],
        "val_capped recall": vc["recall"],
        "val_capped benign homepage FAR": vc["benign_homepage_far"]["rate"],
        "val_capped phish homepage recall": vc["phish_homepage_recall"]["rate"],
        "val (uncapped) PR-AUC": block["val"]["pr_auc"],
        "Tranco top10k homepage FAR (val)": block["tranco_homepages_val"]["top_10k"]["rate"],
        "Tranco 10k-100k homepage FAR (val)": block["tranco_homepages_val"]["10k_100k"]["rate"],
        "dev PR-AUC": dev["pr_auc"],
        "dev recall": dev["recall"],
        "dev benign FAR": dev["benign_far"],
        "dev benign homepage FAR": dev["benign_homepage_far"]["rate"],
        "dev phish homepage recall": dev["phish_homepage_recall"]["rate"],
        "dev unseen-site PR-AUC": block["dev_unseen_site"]["pr_auc"],
    }
    return row


def save(name, payload):
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path
