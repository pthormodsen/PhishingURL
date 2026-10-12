"""E03: feature engineering (trailing slash) and a regularisation validation curve.

Course topics: feature engineering (2.9), regularisation (2.2),
linear models trained with gradient descent (2.5).

Preprocessing v2 treats an empty path as "/" (RFC 3986 6.2.3), so
"netflix.com" and "netflix.com/" become the same input. Motivation: 99.9%
of benign homepage-like training URLs end with "/", versus 86.7% of
phishing ones.

For each preprocessing version, the baseline features (character 3-5
grams, 100k) are fitted once, then SGDClassifier and LogisticRegression
are trained across a range of regularisation strengths.
"""

import time

import pandas as pd

from common import RESULTS_DIR, EvalSets, summary_row
from features import combine, text_block
from models import Prefit, fit_matrix, logistic, sgd

CLASSIFIERS = {
    "sgd alpha=1e-4": lambda: sgd(1e-4),
    "sgd alpha=1e-5": lambda: sgd(1e-5),
    "sgd alpha=1e-6": lambda: sgd(1e-6),
    "logreg C=0.1": lambda: logistic(0.1),
    "logreg C=1": lambda: logistic(1.0),
    "logreg C=10": lambda: logistic(10.0),
}

rows = []
for preprocessing in ["v1", "v2"]:
    sets = EvalSets(preprocessing=preprocessing)
    y = sets.fit["y"].to_numpy()
    features = combine(text=text_block())
    X = features.fit_transform(sets.fit)
    print(f"{preprocessing}: fit rows {len(sets.fit)}", flush=True)

    for classifier_name, make_classifier in CLASSIFIERS.items():
        start = time.time()
        classifier = fit_matrix(make_classifier(), X, y)
        name = f"e03 {preprocessing} | {classifier_name}"
        result = sets.report(
            name.replace(" ", "_").replace("|", "").replace("=", ""),
            Prefit(features, classifier).score,
            {"preprocessing": preprocessing, "fit_seconds": time.time() - start},
        )
        row = summary_row(result)
        row["model"] = name
        rows.append(row)
        print(f"  {classifier_name}: val_capped PR-AUC {row['val_capped PR-AUC']}  "
              f"Tranco top10k FAR {row['Tranco top10k homepage FAR (val)']}  "
              f"dev recall {row['dev recall']}", flush=True)

pd.DataFrame(rows).set_index("model").to_csv(RESULTS_DIR / "e03_summary.csv")
