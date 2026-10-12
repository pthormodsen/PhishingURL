"""E07: final comparison, current baseline against the improved model.

Both models are logistic regression and are trained on the same fit
partition of the site-grouped split. The current baseline uses SGDClassifier
with log loss, which is logistic regression fitted by stochastic gradient
descent.

- Baseline: current production configuration. Preprocessing v1, character
  3-5 gram TF-IDF (100k), SGDClassifier(loss="log_loss", alpha=1e-4).
- Improved: preprocessing v2 (an empty path becomes "/"), the same TF-IDF
  plus scaled structural URL features, LogisticRegression(C=1). The
  configuration was chosen on val in E04, before this comparison.

Same false positive rate: each threshold is set on val_capped so that 1% of
benign URLs are flagged. Comparison sets: holdout (never used before), the
dev benchmark (later in time) and Tranco popular homepages in holdout.
The final test file is not used.

Decision rule, fixed before running. Choose the improved model only if:
1. phishing recall improves by at least 0.02 on holdout_capped and on dev,
2. PR-AUC is not lower on holdout_capped and on dev,
3. false alarms on popular homepages (Tranco top 10k, holdout) and on
   PhreshPhish benign homepages (holdout_capped) do not rise by more than
   0.02.
Otherwise keep the baseline.
"""

import json

import pandas as pd

from common import RESULTS_DIR, EvalSets
from features import combine, text_block
from models import Prefit, fit_matrix, logistic, sgd

MODELS = {
    "baseline": ("v1", lambda: combine(text=text_block()), lambda: sgd(1e-4)),
    "improved": ("v2", lambda: combine(text=text_block(), structural=True), lambda: logistic(1.0)),
}

results = {}
for name, (version, make_features, make_classifier) in MODELS.items():
    sets = EvalSets(include_holdout=True, preprocessing=version)
    features = make_features()
    X = features.fit_transform(sets.fit)
    model = Prefit(features, fit_matrix(make_classifier(), X, sets.fit["y"].to_numpy()))
    results[name] = sets.report(f"e07_{name}", model.score, {"preprocessing": version})["far1"]


def row(block):
    return {
        "threshold (from val)": block["val_capped"]["threshold"],
        "holdout PR-AUC": block["holdout_capped"]["pr_auc"],
        "holdout phishing recall": block["holdout_capped"]["recall"],
        "holdout precision": block["holdout_capped"]["precision"],
        "holdout F1": block["holdout_capped"]["f1"],
        "holdout false positive rate": block["holdout_capped"]["benign_far"],
        "holdout benign homepage false alarms": block["holdout_capped"]["benign_homepage_far"]["rate"],
        "holdout phishing homepage recall": block["holdout_capped"]["phish_homepage_recall"]["rate"],
        "dev PR-AUC": block["dev"]["pr_auc"],
        "dev phishing recall": block["dev"]["recall"],
        "dev precision": block["dev"]["precision"],
        "dev F1": block["dev"]["f1"],
        "dev false positive rate": block["dev"]["benign_far"],
        "dev benign homepage false alarms": block["dev"]["benign_homepage_far"]["rate"],
        "dev phishing homepage recall": block["dev"]["phish_homepage_recall"]["rate"],
        "popular homepages flagged (Tranco top 1k)": block["tranco_homepages_holdout"]["top_1k"]["rate"],
        "popular homepages flagged (Tranco top 10k)": block["tranco_homepages_holdout"]["top_10k"]["rate"],
        "popular homepages flagged (Tranco 10k-100k)": block["tranco_homepages_holdout"]["10k_100k"]["rate"],
    }


table = pd.DataFrame({name: row(block) for name, block in results.items()})
table["difference"] = table["improved"] - table["baseline"]
table.to_csv(RESULTS_DIR / "e07_final_comparison.csv")

d = table["difference"]
criteria = {
    "1 recall +0.02 on holdout and dev": bool(d["holdout phishing recall"] >= 0.02 and d["dev phishing recall"] >= 0.02),
    "2 PR-AUC not lower on holdout and dev": bool(d["holdout PR-AUC"] >= 0 and d["dev PR-AUC"] >= 0),
    "3 homepage false alarms rise at most 0.02": bool(
        d["popular homepages flagged (Tranco top 10k)"] <= 0.02
        and d["holdout benign homepage false alarms"] <= 0.02
    ),
}
decision = "improved" if all(criteria.values()) else "baseline"
(RESULTS_DIR / "e07_decision.json").write_text(json.dumps({"criteria": criteria, "decision": decision}, indent=2) + "\n")

pd.set_option("display.width", 200)
print(table.round(4).to_string())
print(json.dumps(criteria, indent=2))
print("Decision:", decision)
