"""E08: one evaluation of the final model on the reserved final test set.

Final model: the deployed baseline, models/phresh_text_model.joblib, with
the deployed threshold (phishing score >= 0.5, backend/main.py). Nothing
is tuned on these results.

Usage:
    .venv/bin/python experiments/e08_final_test.py --check   # dev benchmark, code check only
    .venv/bin/python experiments/e08_final_test.py           # final test, run once
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix

from common import PHRESH_DIR, RESULTS_DIR, ROOT, add_columns, brand_slices, evaluate, save

THRESHOLD = 0.5
REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"

model = joblib.load(ROOT / "models" / "phresh_text_model.joblib")
phish_column = list(model.classes_).index(0)

# Sites in the sample the deployed model was trained from (30k rows; the
# model used 24k of them). Used to split results into seen and unseen sites.
training_sites = set(add_columns(pd.read_parquet(PHRESH_DIR / "train_sample.parquet"))["site"])


def final_test_with_metadata():
    final = pd.read_parquet(PHRESH_DIR / "final_test.parquet")
    wanted = set(final["url"])
    stream = load_dataset("phreshphish/phreshphish", split="test", streaming=True,
                          columns=["url", "label", "target", "date"], revision=REVISION)
    rows = pd.DataFrame([row for row in stream if row["url"] in wanted]).drop_duplicates("url")
    merged = final.merge(rows[["url", "target", "date"]], on="url", how="left")
    # Some test rows store "no target" as null instead of "None";
    # common.normalise_target maps both to "none"
    assert len(merged) == len(final)
    return merged


def run(frame, name):
    frame = add_columns(frame)
    scores = model.predict_proba(frame["text"])[:, phish_column]
    predictions = np.where(scores >= THRESHOLD, 0, 1)
    y = frame["y"].to_numpy()

    seen = frame["site"].isin(training_sites).to_numpy()
    result = {
        "model": "models/phresh_text_model.joblib",
        "threshold": THRESHOLD,
        "rows": len(frame),
        "label_counts": frame["label"].value_counts().to_dict(),
        "accuracy": round(accuracy_score(y, predictions), 4),
        "confusion_matrix [[phish->phish, phish->benign], [benign->phish, benign->benign]]":
            confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "all": evaluate(frame, scores, THRESHOLD),
        "seen_site": evaluate(frame.loc[seen], scores[seen], THRESHOLD),
        "unseen_site": evaluate(frame.loc[~seen], scores[~seen], THRESHOLD),
        "brands": brand_slices(frame, scores, THRESHOLD),
    }
    save(name, result)
    return result


if "--check" in sys.argv:
    dev = pd.read_parquet(PHRESH_DIR / "dev_benchmark_meta.parquet")
    result = run(dev, "e08_check_on_dev")
    # Must reproduce the recorded notebook 03 test numbers exactly
    assert result["accuracy"] == 0.9004, result["accuracy"]
    assert result["all"]["recall"] == round(0.8000870132695236, 4)
    print("Check passed:", json.dumps({k: result["all"][k] for k in ["precision", "recall", "f1", "benign_far"]}))
else:
    marker = RESULTS_DIR / "e08_final_test.json"
    if marker.exists():
        raise SystemExit("The final test has already been evaluated. Read results/experiments/e08_final_test.json.")
    result = run(final_test_with_metadata(), "e08_final_test")
    print(json.dumps(result, indent=2))
