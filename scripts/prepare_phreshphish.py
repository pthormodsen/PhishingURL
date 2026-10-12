"""Create fixed PhreshPhish data files for reproducible experiments.

Run from the repository root:

    .venv/bin/python scripts/prepare_phreshphish.py

Writes to data/phreshphish/:

- train_sample.parquet: 30,000 rows from the official train split, drawn
  with the same streaming code as notebook 03.
- dev_benchmark.parquet: 10,000 rows from the official test split, drawn
  with the same streaming code as notebook 03. These rows were already
  inspected, so they are a development benchmark, not a final test.
- final_test.parquet: 10,000 rows sampled uniformly from the rest of the
  test split. Rows whose model text also occurs in train_sample or
  dev_benchmark are removed first. Do not use this file for model
  development.
- manifest.json: revision, seeds, counts and file hashes.

Only URL text is compared when building final_test. The script never
reads or prints final test labels.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.features import url_to_model_text  # noqa: E402

DATASET = "phreshphish/phreshphish"
# Latest commit on 2026-02-09. Notebook 03 ran after this without a pinned
# revision, so it used the same data.
REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"

STREAM_SEED = 42
STREAM_BUFFER = 10_000
TRAIN_SAMPLE_SIZE = 30_000
DEV_BENCHMARK_SIZE = 10_000
FINAL_TEST_SIZE = 10_000
FINAL_TEST_SEED = 2026

OUTPUT_DIR = ROOT / "data" / "phreshphish"


def stream_split(split):
    return load_dataset(
        DATASET,
        split=split,
        streaming=True,
        columns=["url", "label"],
        revision=REVISION,
    )


def notebook_03_sample(split, size):
    # Same shuffle and take calls as notebook 03
    sample = stream_split(split).shuffle(
        seed=STREAM_SEED,
        buffer_size=STREAM_BUFFER,
    ).take(size)
    return pd.DataFrame(list(sample))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Streaming train sample...")
    train_sample = notebook_03_sample("train", TRAIN_SAMPLE_SIZE)

    print("Streaming dev benchmark...")
    dev_benchmark = notebook_03_sample("test", DEV_BENCHMARK_SIZE)

    print("Streaming full test split...")
    test_pool = pd.DataFrame(list(stream_split("test")))
    full_test_rows = len(test_pool)

    used_texts = set(
        pd.concat([train_sample["url"], dev_benchmark["url"]])
        .map(url_to_model_text)
    )
    pool_texts = test_pool["url"].map(url_to_model_text)

    overlapping = pool_texts.isin(used_texts)
    test_pool = test_pool.loc[~overlapping]
    pool_texts = pool_texts.loc[~overlapping]

    # Keep one row per model text so the final test has no repeated inputs
    duplicated = pool_texts.duplicated()
    test_pool = test_pool.loc[~duplicated]

    final_test = test_pool.sample(
        n=FINAL_TEST_SIZE,
        random_state=FINAL_TEST_SEED,
    ).reset_index(drop=True)

    files = {
        "train_sample.parquet": train_sample,
        "dev_benchmark.parquet": dev_benchmark,
        "final_test.parquet": final_test,
    }

    for name, frame in files.items():
        frame.to_parquet(OUTPUT_DIR / name, index=False)

    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "stream_seed": STREAM_SEED,
        "stream_buffer_size": STREAM_BUFFER,
        "final_test_seed": FINAL_TEST_SEED,
        "full_test_split_rows": full_test_rows,
        "test_rows_removed_for_text_overlap": int(overlapping.sum()),
        "test_rows_removed_as_duplicates": int(duplicated.sum()),
        "final_test_pool_rows": len(test_pool),
        "files": {
            name: {"rows": len(frame), "sha256": sha256(OUTPUT_DIR / name)}
            for name, frame in files.items()
        },
    }

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
