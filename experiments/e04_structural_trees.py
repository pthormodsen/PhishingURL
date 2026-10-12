"""E04: structural features, tree ensembles, and TF-IDF plus structural features.

Course topics: feature engineering (2.9), decision trees (2.3), random
forests (2.4), gradient boosting (handson-ml3 chapter 7), scaling (2.8)
and pipelines (2.10). Preprocessing v2 throughout.
"""

import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from common import RESULTS_DIR, EvalSets, summary_row
from features import STRUCTURAL_NAMES, Structural, combine, text_block
from models import Prefit, fit_matrix, logistic, sgd

sets = EvalSets(preprocessing="v2")
y = sets.fit["y"].to_numpy()

structural = Structural()
X_structural = structural.transform(sets.fit)

TREE_MODELS = {
    "structural | decision tree depth 10": lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42),
    "structural | random forest": lambda: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "structural | gradient boosting": lambda: HistGradientBoostingClassifier(random_state=42),
}

rows = []


def record(name, score, extra=None):
    result = sets.report(name.replace(" ", "_").replace("|", "").replace("=", "").replace("+", "plus"), score, extra)
    row = summary_row(result)
    row["model"] = name
    rows.append(row)
    print(f"{name}: val_capped PR-AUC {row['val_capped PR-AUC']}  "
          f"Tranco top10k FAR {row['Tranco top10k homepage FAR (val)']}  dev recall {row['dev recall']}", flush=True)
    return result


for name, make_model in TREE_MODELS.items():
    start = time.time()
    model = make_model().fit(X_structural, y)
    record(f"e04 {name}", Prefit(structural, model).score, {"fit_seconds": time.time() - start})
    if name.endswith("random forest"):
        importances = pd.Series(model.feature_importances_, index=STRUCTURAL_NAMES).sort_values(ascending=False)
        importances.round(4).to_csv(RESULTS_DIR / "e04_random_forest_importances.csv")
        print(importances.round(3).to_string())

features = combine(text=text_block(), structural=True)
X = features.fit_transform(sets.fit)
for name, make_classifier in {
    "tfidf + structural | sgd alpha=1e-4": lambda: sgd(1e-4),
    "tfidf + structural | sgd alpha=1e-6": lambda: sgd(1e-6),
    "tfidf + structural | logreg C=1": lambda: logistic(1.0),
}.items():
    start = time.time()
    classifier = fit_matrix(make_classifier(), X, y)
    record(f"e04 {name}", Prefit(features, classifier).score, {"fit_seconds": time.time() - start})

pd.DataFrame(rows).set_index("model").to_csv(RESULTS_DIR / "e04_summary.csv")
