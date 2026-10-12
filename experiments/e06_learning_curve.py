"""E06: learning curves (course 2.1) on the site-grouped split, preprocessing v2.

Each size is a random subset of the fit partition (seed 42); evaluation
sets and the threshold rule are unchanged.
"""

import pandas as pd

from common import RESULTS_DIR, EvalSets, summary_row
from features import combine, text_block
from models import Prefit, fit_matrix, sgd

sets = EvalSets(preprocessing="v2")
MODELS = {
    "baseline features | sgd alpha=1e-4": (lambda: combine(text=text_block()), lambda: sgd(1e-4)),
    "tfidf + structural | sgd alpha=1e-6": (lambda: combine(text=text_block(), structural=True), lambda: sgd(1e-6)),
}

rows = []
for size in [6_000, 24_000, 96_000, len(sets.fit)]:
    subset = sets.fit.sample(n=size, random_state=42)
    for model_name, (make_features, make_classifier) in MODELS.items():
        features = make_features()
        classifier = fit_matrix(make_classifier(), features.fit_transform(subset), subset["y"].to_numpy())
        name = f"e06 n={size} | {model_name}"
        result = sets.report(name.replace(" ", "_").replace("|", "").replace("=", "").replace("+", "plus"),
                             Prefit(features, classifier).score, {"train_rows": size})
        row = summary_row(result)
        row.update(model=model_name, train_rows=size)
        rows.append(row)
        print(f"{name}: val_capped PR-AUC {row['val_capped PR-AUC']}  "
              f"Tranco top10k FAR {row['Tranco top10k homepage FAR (val)']}  dev recall {row['dev recall']}", flush=True)

pd.DataFrame(rows).to_csv(RESULTS_DIR / "e06_learning_curve.csv", index=False)
