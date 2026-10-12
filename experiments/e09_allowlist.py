"""E09: a Tranco allowlist in front of the URL model.

E07 rejected the improved model only because it raised false alarms on
homepages (rule 3). E09 tests whether an allowlist of popular domains, applied
before the model, removes that cost without letting phishing through.
URL-only: no HTML features.

Models: the two E07 configurations, retrained exactly as in E07 on the fit
partition. The script stops if their thresholds, holdout_capped recall or dev
recall differ from the recorded E07 results.

- reference: baseline (v1, char 3-5 TF-IDF, SGD alpha=1e-4), no allowlist.
  This is the current production configuration.
- candidate: improved (v2, TF-IDF plus structural, LogisticRegression C=1).

Thresholds: the E07 rule, unchanged. Each model's threshold is set on
val_capped, without an allowlist, so that 1% of benign URLs are flagged.
The allowlist then overrides the model: an allowlisted URL is legitimate
(score 0). Thresholds are not re-tuned after the allowlist is added.

Allowlist: the Tranco list already used in E00-E07 (Y83YG, top 100k
registered domains). Primary size: top 10,000. Sizes 1,000 and 100,000 are
reported as a sensitivity check only and play no part in the decision.

Arms (from narrowest to broadest):

- none: model only.
- exact_homepage: the URL is the bare homepage of a listed domain. After
  lowercasing and removing one leading "www.", the hostname must equal a
  listed domain exactly, the path must be empty or "/", and there must be no
  query, fragment, port or user info. Subdomains and paths go to the model.
- site: the URL's site (registrable domain from the public suffix list,
  including its private section) is listed. abc.github.io is its own site, so
  hosting platforms in the private section are not covered by github.io
  being listed. docs.google.com is covered by google.com.
- registered_domain: the URL's ICANN registered domain is listed. Every
  subdomain and path of a listed domain is allowed, including user content on
  hosting platforms such as abc.github.io.

Evaluation sets: holdout_capped, holdout (uncapped) and dev. val is used only
for the thresholds. Tranco homepages in the holdout partition are scored too.
The final test file is not used.

Success criteria, fixed before the first run. Primary size top 10k.
"Reference" is the baseline without an allowlist. An arm passes only if all
of these hold for candidate + arm:

C1 safety: at most 0.5% of phishing URLs are allowlisted, on holdout_capped,
   holdout and dev.
C2 recall: phishing recall is at least reference + 0.02 on holdout_capped and
   on dev.
C3 ranking: PR-AUC (allowlisted rows scored 0) is not lower than reference
   on holdout_capped and on dev.
C4 homepage false alarms: the false alarm rate on PhreshPhish benign
   homepages rises by at most 0.02 over reference on holdout_capped and on
   dev, and the false alarm rate on Tranco 10k-100k homepages (holdout,
   outside the primary allowlist) rises by at most 0.02. Tranco top 10k
   homepages are reported, but they are allowlisted by construction.
C5 non-allowlisted rows: on the rows the arm does not allowlist, the candidate
   model's PR-AUC is not lower than the reference model's on the same rows,
   and its benign false alarm rate is at most reference + 0.005, on
   holdout_capped and on dev. This checks that the gain does not come from
   the allowlist alone.

Decision: the narrowest passing arm, in the order exact_homepage, site,
registered_domain. If no arm passes, keep the baseline. Training on the full
train split (E10) is only investigated if an arm passes.

Nothing is saved to models/ and production is not changed.
"""

import json
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from common import RESULTS_DIR, EvalSets, evaluate, rate, save, threshold_at_far
from features import combine, text_block
from models import Prefit, fit_matrix, logistic, sgd

PRIMARY_TOP = 10_000
SIZES = [1_000, 10_000, 100_000]
ARMS = ["none", "exact_homepage", "site", "registered_domain"]
CANDIDATE_ARMS = ARMS[1:]
EVAL_SETS = ["holdout_capped", "holdout", "dev"]

MODELS = {
    "baseline": ("v1", lambda: combine(text=text_block()), lambda: sgd(1e-4)),
    "improved": ("v2", lambda: combine(text=text_block(), structural=True), lambda: logistic(1.0)),
}

# Recorded in results/experiments/e07_*.json
EXPECTED_E07 = {
    "baseline": {"threshold": 0.5483, "holdout_capped": 0.7752, "dev": 0.7255},
    "improved": {"threshold": 0.6865, "holdout_capped": 0.8749, "dev": 0.8295},
}


def exact_homepage_host(url):
    """The hostname (without "www.") if the URL is a bare homepage, else None."""
    url = url.strip()
    parsed = urlsplit(url if "://" in url else "https://" + url)
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is not None or parsed.username is not None or parsed.password is not None:
        return None
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        return None
    if "?" in url or "#" in url:
        return None
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def allowlisted(frame, arm, domains):
    if arm == "none":
        return np.zeros(len(frame), dtype=bool)
    if arm == "exact_homepage":
        return frame["url"].map(exact_homepage_host).isin(domains).to_numpy()
    if arm == "site":
        return frame["site"].isin(domains).to_numpy()
    if arm == "registered_domain":
        return frame["domain"].isin(domains).to_numpy()
    raise ValueError(arm)


def evaluate_arm(frame, scores, allow, threshold):
    y = frame["y"].to_numpy()
    phish, benign = y == 0, y == 1
    combined = np.where(allow, 0.0, scores)
    result = evaluate(frame, combined, threshold)
    result["allowlisted_rows"] = int(allow.sum())
    result["phish_allowlisted"] = rate(allow, phish)
    result["phish_lost_to_allowlist"] = rate(allow & (scores >= threshold), phish)
    result["benign_allowlisted"] = rate(allow, benign)
    result["benign_homepage_allowlisted"] = rate(allow, benign & frame["homepage_like"].to_numpy())
    result["top_allowlisted_phish_domains"] = (
        frame.loc[allow & phish, "domain"].value_counts().head(10).to_dict()
    )
    rest = ~allow
    result["not_allowlisted"] = evaluate(frame.loc[rest], scores[rest], threshold)
    return result


def tranco_homepages(frame, scores, allow, threshold):
    flagged = np.where(allow, 0.0, scores) >= threshold
    ranks = frame["rank"].to_numpy()
    return {
        "top_1k": rate(flagged, ranks <= 1_000),
        "top_10k": rate(flagged, ranks <= 10_000),
        "10k_100k": rate(flagged, ranks > 10_000),
    }


results = {}
for name, (version, make_features, make_classifier) in MODELS.items():
    sets = EvalSets(include_holdout=True, preprocessing=version)
    features = make_features()
    model = Prefit(features, fit_matrix(make_classifier(), features.fit_transform(sets.fit),
                                        sets.fit["y"].to_numpy()))

    val = sets.sets["val_capped"]
    threshold = threshold_at_far(model.score(val), val["y"].to_numpy())
    scores = {key: model.score(sets.sets[key]) for key in EVAL_SETS}
    homepages = sets.homepages["holdout"]
    homepage_scores = model.score(homepages)

    expected = EXPECTED_E07[name]
    reproduced = {
        "threshold": round(threshold, 4),
        **{key: evaluate(sets.sets[key], scores[key], threshold)["recall"]
           for key in ["holdout_capped", "dev"]},
    }
    # E07 ran under Python 3.14, this .venv uses 3.13 (3.14 is not installed).
    # The data and evaluation sets are identical, but the retrained models
    # differ slightly: the baseline detects one more holdout_capped phishing
    # URL (15,580 vs 15,579), and the improved model's threshold is 0.6881
    # instead of 0.6865 (dev recall 0.8273 vs 0.8295). The fit is
    # deterministic (unchanged across NumPy seeds), so this is a numerical
    # difference between environments. Both models are retrained in this run,
    # so the comparison below is internally consistent; the tolerance only
    # guards against a wrong setup.
    if any(abs(reproduced[key] - expected[key]) > 0.005 for key in expected):
        raise SystemExit(f"{name} does not reproduce E07: {reproduced} != {expected}")
    print(f"{name}: reproduces E07 {reproduced}", flush=True)

    model_results = {
        "preprocessing": version, "threshold": round(threshold, 4),
        "e07_recorded": expected, "e07_reproduced": {k: float(v) for k, v in reproduced.items()},
        "sizes": {},
    }
    for size in SIZES:
        domains = set(sets.tranco.loc[sets.tranco["rank"] <= size, "domain"])
        by_arm = {}
        for arm in ARMS:
            block = {
                key: evaluate_arm(sets.sets[key], scores[key],
                                  allowlisted(sets.sets[key], arm, domains), threshold)
                for key in EVAL_SETS
            }
            block["tranco_homepages_holdout"] = tranco_homepages(
                homepages, homepage_scores, allowlisted(homepages, arm, domains), threshold)
            by_arm[arm] = block
        model_results["sizes"][f"top_{size}"] = by_arm
    results[name] = model_results

save("e09_allowlist", results)


# ---------------------------------------------------------------- decision

def primary(model, arm):
    return results[model]["sizes"][f"top_{PRIMARY_TOP}"][arm]


reference = primary("baseline", "none")
criteria = {}
for arm in CANDIDATE_ARMS:
    cand = primary("improved", arm)
    ref_same_arm = primary("baseline", arm)
    checks = {
        "C1 phishing allowlisted <= 0.005 (holdout_capped, holdout, dev)": all(
            cand[key]["phish_allowlisted"]["rate"] <= 0.005 for key in EVAL_SETS),
        "C2 recall >= reference + 0.02 (holdout_capped, dev)": all(
            cand[key]["recall"] - reference[key]["recall"] >= 0.02 for key in ["holdout_capped", "dev"]),
        "C3 PR-AUC >= reference (holdout_capped, dev)": all(
            cand[key]["pr_auc"] >= reference[key]["pr_auc"] for key in ["holdout_capped", "dev"]),
        "C4 homepage false alarms rise <= 0.02": all([
            *(cand[key]["benign_homepage_far"]["rate"] - reference[key]["benign_homepage_far"]["rate"] <= 0.02
              for key in ["holdout_capped", "dev"]),
            cand["tranco_homepages_holdout"]["10k_100k"]["rate"]
            - reference["tranco_homepages_holdout"]["10k_100k"]["rate"] <= 0.02,
        ]),
        "C5 non-allowlisted rows: PR-AUC >= reference, FAR <= reference + 0.005": all(
            cand[key]["not_allowlisted"]["pr_auc"] >= ref_same_arm[key]["not_allowlisted"]["pr_auc"]
            and cand[key]["not_allowlisted"]["benign_far"]
            <= ref_same_arm[key]["not_allowlisted"]["benign_far"] + 0.005
            for key in ["holdout_capped", "dev"]),
    }
    criteria[arm] = {"checks": checks, "passed": all(checks.values())}

passing = [arm for arm in CANDIDATE_ARMS if criteria[arm]["passed"]]
decision = {
    "primary_allowlist": f"Tranco top {PRIMARY_TOP}",
    "criteria": criteria,
    "decision": f"improved + {passing[0]}" if passing else "baseline (no arm passed)",
    "run_e10": bool(passing),
}
(RESULTS_DIR / "e09_decision.json").write_text(json.dumps(decision, indent=2) + "\n")


# ---------------------------------------------------------------- tables

def row(block, key):
    b = block[key]
    return {
        "PR-AUC": b["pr_auc"],
        "recall": b["recall"],
        "precision": b["precision"],
        "benign FAR": b["benign_far"],
        "benign homepage FAR": b["benign_homepage_far"]["rate"],
        "phish allowlisted": b["phish_allowlisted"]["rate"],
        "phish allowlisted (n)": b["phish_allowlisted"]["k"],
        "recall lost to allowlist": b["phish_lost_to_allowlist"]["rate"],
        "benign allowlisted": b["benign_allowlisted"]["rate"],
        "not allowlisted: PR-AUC": b["not_allowlisted"]["pr_auc"],
        "not allowlisted: recall": b["not_allowlisted"]["recall"],
        "not allowlisted: benign FAR": b["not_allowlisted"]["benign_far"],
        "not allowlisted: benign homepage FAR": b["not_allowlisted"]["benign_homepage_far"]["rate"],
    }


summary = []
for model in MODELS:
    for size in SIZES:
        for arm in ARMS:
            block = results[model]["sizes"][f"top_{size}"][arm]
            homes = block["tranco_homepages_holdout"]
            for key in EVAL_SETS:
                summary.append({
                    "model": model, "allowlist size": size, "arm": arm, "set": key, **row(block, key),
                    "Tranco top10k homepages flagged (holdout)": homes["top_10k"]["rate"],
                    "Tranco 10k-100k homepages flagged (holdout)": homes["10k_100k"]["rate"],
                })
summary = pd.DataFrame(summary)
summary.to_csv(RESULTS_DIR / "e09_summary.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
view = summary.loc[(summary["allowlist size"] == PRIMARY_TOP) & (summary["set"] != "holdout")]
print(view.drop(columns=["allowlist size"]).round(4).to_string(index=False))
print(json.dumps(decision, indent=2))
