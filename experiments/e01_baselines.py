"""E01: baselines under the site-grouped protocol.

- production: models/phresh_text_model.joblib as deployed. Its 24k training
  rows were a random sample of the whole train split, so they include sites
  from val. Its val numbers are therefore optimistic; dev and Tranco are fair.
- baseline_24k: same configuration, trained on 24k random fit rows.
- baseline_full: same configuration, trained on the whole fit partition.
"""

import time

import joblib
import pandas as pd

from common import ROOT, EvalSets, summary_row
from models import build, fit, scorer, text_scorer

sets = EvalSets()
rows = []

production = joblib.load(ROOT / "models" / "phresh_text_model.joblib")
rows.append(summary_row(sets.report("e01_production", text_scorer(production),
                                    {"note": "val optimistic: training sample overlaps val sites"})))

for name, frame in [
    ("e01_baseline_24k", sets.fit.sample(n=24_000, random_state=42)),
    ("e01_baseline_full", sets.fit),
]:
    start = time.time()
    model = fit(build(), frame)
    elapsed = time.time() - start
    rows.append(summary_row(sets.report(name, scorer(model), {"train_rows": len(frame), "fit_seconds": elapsed})))

table = pd.DataFrame(rows).set_index("model").T
print(table.to_string())
