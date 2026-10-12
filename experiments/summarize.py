"""Collect experiment JSON results into one table.

Usage: .venv/bin/python experiments/summarize.py [prefix ...]
"""

import json
import sys

import pandas as pd

from common import RESULTS_DIR, summary_row

prefixes = sys.argv[1:] or [""]
rows = []
for path in sorted(RESULTS_DIR.glob("*.json")):
    if not any(path.stem.startswith(p) for p in prefixes):
        continue
    result = json.loads(path.read_text())
    if "far1" not in result:
        continue
    rows.append(summary_row(result))

table = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(table.to_string())
