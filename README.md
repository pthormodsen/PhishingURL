# Phishing URL Detector

DAT158 machine learning project by William Waly, Fahad Ahmed and Patrik
Thormodsen.

- **Live website:** https://phishing.patreek.no
- **Report:** [Report.pdf](Report.pdf) (source: `Report.odt`)

Enter a URL and the website tells you whether a machine learning model flags
it as possible phishing. The model reads only the URL text. It never visits
the website. The result is a warning signal, not a guarantee.

## Results

The deployed model was evaluated once on 10,000 reserved URLs from the
PhreshPhish test split (threshold 0.5). Phishing is the positive class.

| Metric | Model | Always "legitimate" baseline |
|---|---|---|
| Accuracy | 0.906 | 0.552 |
| Precision (phishing) | 0.977 | - |
| Recall (phishing) | 0.808 | 0.000 |
| F1 | 0.885 | 0.000 |
| False positive rate | 1.5% | 0% |

Known weaknesses: about 17-19% of popular homepages such as `google.com` are
flagged, phishing pages on legitimate hosting platforms are often missed, and
look-alike domains such as `netf1ix.com` are not reliably caught. See the
report and `notebooks/05_summary_and_final_evaluation.ipynb`.

## How it works

1. **Preprocessing** (`backend/features.py`): lowercase the hostname, remove
   the scheme and `www.`, keep path, query and fragment.
2. **Model:** character 3-5 gram TF-IDF + logistic regression
   (`SGDClassifier(loss="log_loss")`).
3. **Backend** (`backend/main.py`): FastAPI, `POST /predict {"url": ...}`.
   A URL is flagged as phishing if its phishing score is at least 0.5.
4. **Frontend** (`frontend/`): React + TypeScript + Vite.

**The deployed model** is `models/phresh_text_model.joblib`. It was trained
in `notebooks/03_phreshphish_datasett.ipynb` on 24,000 URLs from the
PhreshPhish train split. The later experiments (`experiments/e01` to `e09`)
tried to improve it, but no candidate met all our criteria, so this model
was kept unchanged. `models/random_forest.joblib` is an early PhiUSIIL model
from notebook 01 and is not used by the website.

Labels everywhere: **0 = phishing, 1 = legitimate**.

## Run it locally

Requirements: Python 3.14 and Node.js.

```sh
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Terminal 1: backend on http://127.0.0.1:8000 (run from the repository root)
.venv/bin/uvicorn backend.main:app --reload

# Terminal 2: website on http://localhost:5173
cd frontend
npm install
npm run dev
```

The Vite dev server forwards `/api/*` to the backend.

You can also call the backend directly:

```sh
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"url": "example.com"}'
```

## Reproduce the results

All data samples, results and the deployed model are committed, so every
notebook can be read without running anything. To rerun:

1. **Data.** The PhiUSIIL CSV and the fixed PhreshPhish samples are already
   in `data/`. To recreate the samples from the pinned dataset revision:
   `.venv/bin/python scripts/prepare_phreshphish.py`
2. **Notebooks** (run from `notebooks/`, for example in VS Code or with
   `cd notebooks && ../.venv/bin/jupyter execute <notebook>`):

   | Notebook | Content |
   |---|---|
   | `01_exploration` | PhiUSIIL exploration, first random forest |
   | `02_url_text_model` | First TF-IDF text model on PhiUSIIL |
   | `03_phreshphish_datasett` | **Trains and saves the deployed model** (downloads data from Hugging Face; overwrites `models/phresh_text_model.joblib`) |
   | `04_homepage_ablation` | Homepage augmentation experiment (no improvement) |
   | `05_summary_and_final_evaluation` | Summary of everything, trains nothing |

3. **Experiments.** First download the full train split:
   `.venv/bin/python experiments/prepare_data.py`. Then run each script from
   `experiments/`, for example `cd experiments && ../.venv/bin/python e01_baselines.py`.
   Results are written to `results/experiments/`.

   | Script | Content |
   |---|---|
   | `e00_data_audit` | Data audit |
   | `e01_baselines` | Baselines with site-grouped splits |
   | `e02_capacity` | Classifier, regularisation and n-gram range |
   | `e03_canonical_path` | Preprocessing variant and regularisation curve |
   | `e04_structural_trees` | Structural features with decision tree, random forest, gradient boosting |
   | `e05_ensembles` | Soft voting and stacking |
   | `e06_learning_curve` | Learning curves |
   | `e07_final_comparison` | Deployed model against the best improved model (kept the deployed model) |
   | `e08_final_test` | One-time final test. **Do not rerun:** the script refuses to run again |
   | `e09_allowlist` | Allowlist of popular domains in front of the model (no improvement) |

   Shared code is in `common.py` (splits and evaluation), `features.py` and
   `models.py`.

## Repository layout

| Path | Content |
|---|---|
| `Report.pdf`, `Report.odt` | The project report |
| `notebooks/` | Exploration, model training and the summary |
| `experiments/` | Experiment scripts `e00` to `e09` |
| `scripts/prepare_phreshphish.py` | Creates the fixed PhreshPhish samples, including the reserved test set |
| `data/` | Datasets and samples |
| `models/` | Saved models (`phresh_text_model.joblib` is deployed) |
| `results/` | Saved experiment results |
| `backend/` | FastAPI service |
| `frontend/` | React website and Nginx configuration |
| `compose.yaml`, `Makefile` | Production deployment |

## Our production setup

The live website runs on our own server (docker01) with Docker Compose:

```
Internet -> Cloudflare Tunnel -> frontend (Nginx :80) -> /api/predict -> backend (FastAPI :8000)
```

- `frontend` serves the built React app and is the only container on the
  external `cloudflare` Docker network (alias `phishing`, tunnel target
  `http://phishing:80`). No host ports are published.
- `backend` is only on an internal network, runs as a non-root user with a
  read-only filesystem, and uses `backend/requirements.txt`, pinned to the
  library versions the model was saved with. Do not change scikit-learn,
  numpy, scipy or joblib without retraining the model.

```sh
make deploy        # build, start, wait for health, smoke test
make update        # git pull, rebuild, restart, smoke test
make ps            # status;  make logs: follow logs;  make help: all targets
```

`make deploy` needs the external network `cloudflare` (for a local test:
`docker network create cloudflare`). To try the site on your own machine,
use [Run it locally](#run-it-locally) instead.

## Data and references

- **PhreshPhish:** T. Dalton et al., "PhreshPhish: A Real-World, High-Quality,
  Large-Scale Phishing Website Dataset and Benchmark", arXiv:2507.10854, 2025.
  Dataset `phreshphish/phreshphish` on Hugging Face (revision `eabec4b`),
  CC BY 4.0.
- **PhiUSIIL:** A. Prasad and S. Chandra, "PhiUSIIL: A diverse security profile
  empowered phishing URL detection framework based on similarity index and
  incremental learning", Computers & Security 136, 2024. UCI Machine Learning
  Repository, CC BY 4.0. Used in notebooks 01 and 02 only.
- **Tranco:** V. Le Pochat et al., "Tranco: A Research-Oriented Top Sites
  Ranking Hardened Against Manipulation", NDSS 2019. List `Y83YG`, used only
  to evaluate false alarms on popular homepages.
- **Tools:** scikit-learn (Pedregosa et al., JMLR 12, 2011), FastAPI, React,
  Vite, Docker, Nginx, Cloudflare Tunnel.

## AI assistance

Claude Code and ChatGPT were used to assist with parts of the code,
experiments, evaluation, web application, deployment configuration and
report writing. The project group reviewed the generated content, verified
the results and made the final decisions.
