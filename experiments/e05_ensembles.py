"""E05: ensembles of the text model and the structural tree model.

Course topics: ensemble learning, soft voting and stacking (handson-ml3
chapter 7), cross-validation. Preprocessing v2 throughout.

- Text model: TF-IDF (char 3-5, 100k) + structural features, SGD alpha=1e-6
  (the best text configuration from E04).
- Structural model: HistGradientBoosting on structural features.
- Soft voting: mean of the two phishing probabilities.
- Stacking: HistGradientBoosting on structural features plus the text
  model score. On the fit partition the text score comes from 5-fold
  GroupKFold by site (out-of-fold), so the stacker never sees a score the
  text model produced for its own training rows.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from common import RESULTS_DIR, EvalSets, summary_row
from features import Structural, combine, text_block
from models import Prefit, fit_matrix, sgd

sets = EvalSets(preprocessing="v2")
fit_rows = sets.fit
y = fit_rows["y"].to_numpy()
structural = Structural()
X_structural = structural.transform(fit_rows)

rows = []


def record(name, score, extra=None):
    result = sets.report(name.replace(" ", "_").replace("+", "plus"), score, extra)
    row = summary_row(result)
    row["model"] = name
    rows.append(row)
    print(f"{name}: val_capped PR-AUC {row['val_capped PR-AUC']}  "
          f"Tranco top10k FAR {row['Tranco top10k homepage FAR (val)']}  dev recall {row['dev recall']}", flush=True)


def make_text_model():
    return combine(text=text_block(), structural=True), sgd(1e-6)


start = time.time()
text_features, text_classifier = make_text_model()
text_model = Prefit(text_features, fit_matrix(text_classifier, text_features.fit_transform(fit_rows), y))
tree_model = Prefit(structural, HistGradientBoostingClassifier(random_state=42).fit(X_structural, y))

record("e05 soft voting text + structural GB",
       lambda frame: (text_model.score(frame) + tree_model.score(frame)) / 2)

# Out-of-fold text scores for the stacker
oof = np.zeros(len(fit_rows))
for fold, (train_index, test_index) in enumerate(GroupKFold(n_splits=5).split(fit_rows, y, groups=fit_rows["site"])):
    features, classifier = make_text_model()
    part = fit_rows.iloc[train_index]
    model = Prefit(features, fit_matrix(classifier, features.fit_transform(part), part["y"].to_numpy()))
    oof[test_index] = model.score(fit_rows.iloc[test_index])
    print(f"  fold {fold} done", flush=True)

stacker = HistGradientBoostingClassifier(random_state=42).fit(np.column_stack([X_structural, oof]), y)
stack_column = list(stacker.classes_).index(0)


def stacked_score(frame):
    features = np.column_stack([structural.transform(frame), text_model.score(frame)])
    return stacker.predict_proba(features)[:, stack_column]


record("e05 stacking GB on structural + text score", stacked_score,
       {"fit_seconds_total": time.time() - start})

pd.DataFrame(rows).set_index("model").to_csv(RESULTS_DIR / "e05_summary.csv")
