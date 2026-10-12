"""Prepare data for the model improvement experiments.

Run from the repository root:

    .venv/bin/python experiments/prepare_data.py

Writes:

- data/phreshphish/train_full.parquet: the complete official train split
  (url, label, target, date) at the pinned revision.
- data/phreshphish/dev_benchmark_meta.parquet: target and date for the
  10k dev benchmark rows. Only test rows whose URL is in the dev benchmark
  are kept, so final test rows are never stored here.
- data/external/tranco_Y83YG_top100k.parquet: the top 100,000 registered
  domains of Tranco list Y83YG (generated 2026-10-01, 30-day average of
  CrUX, Farsight, Majestic, Radar and Umbrella). The raw download stays in
  data/external/tranco_raw/, which is not tracked.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
PHRESH_DIR = ROOT / "data" / "phreshphish"
EXTERNAL_DIR = ROOT / "data" / "external"

REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"
COLUMNS = ["url", "label", "target", "date"]

TRANCO_ID = "Y83YG"
TRANCO_URL = f"https://tranco-list.eu/download/{TRANCO_ID}/1000000"
TRANCO_RAW = EXTERNAL_DIR / "tranco_raw" / f"tranco_{TRANCO_ID}.csv"
TRANCO_SHA256 = "b3f7246defc90158d80324b59c75fb78cf7bf0ed1d2f11a1279caf88e5a6a7c0"
TRANCO_TOP = 100_000


def stream(split):
    return load_dataset(
        "phreshphish/phreshphish",
        split=split,
        streaming=True,
        columns=COLUMNS,
        revision=REVISION,
    )


def main():
    print("Streaming full train split...")
    train = pd.DataFrame(list(stream("train")))
    train["date"] = pd.to_datetime(train["date"])
    train.to_parquet(PHRESH_DIR / "train_full.parquet", index=False)

    print("Collecting dev benchmark metadata...")
    dev = pd.read_parquet(PHRESH_DIR / "dev_benchmark.parquet")
    dev_urls = set(dev["url"])
    dev_rows = [row for row in stream("test") if row["url"] in dev_urls]
    dev_meta = (
        pd.DataFrame(dev_rows)
        .drop_duplicates("url")
        .set_index("url")
        .loc[dev["url"]]
        .reset_index()
    )
    assert dev_meta["label"].tolist() == dev["label"].tolist()
    dev_meta["date"] = pd.to_datetime(dev_meta["date"])
    dev_meta.to_parquet(PHRESH_DIR / "dev_benchmark_meta.parquet", index=False)

    print("Preparing Tranco subset...")
    if not TRANCO_RAW.exists():
        raise SystemExit(f"Download {TRANCO_URL} to {TRANCO_RAW} first")
    digest = hashlib.sha256(TRANCO_RAW.read_bytes()).hexdigest()
    assert digest == TRANCO_SHA256, "Unexpected Tranco file"

    tranco = pd.read_csv(TRANCO_RAW, names=["rank", "domain"], nrows=TRANCO_TOP)
    tranco["domain"] = tranco["domain"].str.lower()
    tranco.to_parquet(EXTERNAL_DIR / f"tranco_{TRANCO_ID}_top100k.parquet", index=False)

    summary = {
        "train_full_rows": len(train),
        "train_label_counts": train["label"].value_counts().to_dict(),
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "dev_benchmark_rows": len(dev_meta),
        "dev_date_range": [str(dev_meta["date"].min().date()), str(dev_meta["date"].max().date())],
        "tranco_list": TRANCO_ID,
        "tranco_rows": len(tranco),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
