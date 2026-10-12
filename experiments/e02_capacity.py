"""E02: classifier, regularisation and feature capacity on the full fit partition.

Every configuration uses the same fit rows, val threshold rule and
evaluation sets. Preprocessing is unchanged (backend model text).
"""

import time

import pandas as pd

from common import RESULTS_DIR, EvalSets, summary_row
from features import combine, host_block, text_block
from models import Prefit, fit_matrix, logistic, sgd

sets = EvalSets()
y = sets.fit["y"].to_numpy()

FEATURES = {
    "F0 baseline char3-5 100k": lambda: combine(text=text_block()),
    "F1 char2-6 1M sublinear": lambda: combine(
        text=text_block(ngram_range=(2, 6), max_features=1_000_000, sublinear_tf=True)),
    "F2 F1 + anchored host + structural": lambda: combine(
        text=text_block(ngram_range=(2, 6), max_features=1_000_000, sublinear_tf=True),
        host=host_block(), structural=True),
}

CLASSIFIERS = {
    "F0 baseline char3-5 100k": {
        "sgd a1e-4": lambda: sgd(1e-4),
        "sgd a1e-5": lambda: sgd(1e-5),
        "sgd a1e-6": lambda: sgd(1e-6),
        "logreg C1": lambda: logistic(1.0),
        "logreg C10": lambda: logistic(10.0),
    },
    "F1 char2-6 1M sublinear": {
        "sgd a1e-6": lambda: sgd(1e-6),
        "logreg C10": lambda: logistic(10.0),
    },
    "F2 F1 + anchored host + structural": {
        "sgd a1e-6": lambda: sgd(1e-6),
        "logreg C10": lambda: logistic(10.0),
    },
}

rows = []
for feature_name, make_features in FEATURES.items():
    start = time.time()
    features = make_features()
    X = features.fit_transform(sets.fit)
    vectorise_seconds = time.time() - start
    print(f"{feature_name}: {X.shape}, {vectorise_seconds:.0f}s", flush=True)

    for classifier_name, make_classifier in CLASSIFIERS[feature_name].items():
        start = time.time()
        classifier = fit_matrix(make_classifier(), X, y)
        seconds = time.time() - start
        name = f"e02 {feature_name} | {classifier_name}"
        model = Prefit(features, classifier)
        result = sets.report(name.replace(" ", "_").replace("|", "").replace("+", "plus"),
                             model.score, {"features": X.shape[1], "fit_seconds": seconds})
        row = summary_row(result)
        row["model"] = name
        rows.append(row)
        print(f"  {classifier_name}: {seconds:.0f}s  val_capped PR-AUC {row['val_capped PR-AUC']}"
              f"  Tranco top10k FAR {row['Tranco top10k homepage FAR (val)']}", flush=True)

table = pd.DataFrame(rows).set_index("model")
table.to_csv(RESULTS_DIR / "e02_summary.csv")
print(table.T.to_string())
