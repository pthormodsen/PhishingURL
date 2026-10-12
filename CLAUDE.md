# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Phishing URL detector (DAT158 course project). Models are trained in Jupyter notebooks and saved to `models/*.joblib`. A FastAPI backend loads one model and serves predictions. A React + Vite frontend calls the backend. The written report lives in `Report.odt`.

## Commands

Python uses the repo-local virtualenv `.venv` (Python 3.14, with fastapi, uvicorn, scikit-learn, pandas, datasets and jupyter). There is no requirements file.

```sh
# Backend: run from the repo root, because main.py imports `backend.features`
.venv/bin/uvicorn backend.main:app --reload        # serves on 127.0.0.1:8000

# Frontend (from frontend/)
npm install
npm run dev       # Vite dev server; proxies /api/* -> http://127.0.0.1:8000 (and strips the /api prefix)
npm run build     # tsc -b && vite build
npm run lint      # eslint
```

There are no tests.

```sh
# Production (docker01): see README "Deployment on docker01"
make deploy          # build, start, wait for health, smoke test through Nginx
make smoke-tunnel    # test http://phishing:80 on the external cloudflare network
make help            # all targets
```

`compose.yaml`: `frontend` (Nginx, built React app, proxies `/api/predict` to `backend:8000/predict`) is the only service on the external `cloudflare` network (alias `phishing`); `backend` is only on an internal network. No host ports. The backend image uses `backend/requirements.txt`, pinned to the versions the model was pickled with; do not bump scikit-learn/numpy/scipy/joblib without retraining.

## Architecture

- **Label convention: `0` = phishing, `1` = legitimate.** This applies everywhere: in the datasets, the notebooks and the API response's `prediction` field. Metrics use `pos_label=0`. To get the phishing probability, use `list(model.classes_).index(0)`. Never assume it is column 1.
- **Notebooks are the training pipeline** (run them from `notebooks/`, since paths are relative, e.g. `../data`, `../models`):
  - `01_exploration.ipynb` trains on the local PhiUSIIL CSV (`data/PhiUSIIL_Phishing_URL_Dataset.csv`) using 3 numeric URL features, and saves `models/random_forest.joblib`. The backend does not use this model.
  - `02_url_text_model.ipynb` runs a char n-gram TF-IDF + `SGDClassifier(log_loss)` experiment on PhiUSIIL.
  - `03_phreshphish_datasett.ipynb` streams a 30k sample of the Hugging Face `phreshphish/phreshphish` dataset (labels `phish`/`benign` mapped to 0/1), trains the same TF-IDF + SGD pipeline, and saves **`models/phresh_text_model.joblib`**. This is the model the backend serves.
  - `scripts/prepare_phreshphish.py` writes fixed PhreshPhish files to `data/phreshphish/` (pinned revision, deterministic). `train_sample` and `dev_benchmark` reproduce the notebook 03 samples. `final_test.parquet` is reserved for one final evaluation: do not read its labels during development.
  - `04_homepage_ablation.ipynb` holds the 2x2 homepage ablation; no arm met its criteria.
  - `experiments/` holds the later experiments (`e00` audit to `e08` final test), sharing `common.py` (site-grouped fit/val/holdout split, evaluation) and `features.py`. Run scripts from `experiments/`; results go to `results/experiments/`. `prepare_data.py` downloads the full train split and the Tranco list. Notebook `05_summary_and_final_evaluation.ipynb` reads those results and trains nothing.
  - Final decision (`e07`): keep the baseline (`models/phresh_text_model.joblib`, threshold 0.5). The final test has been evaluated once (`e08`); `e08_final_test.py` refuses to run again. Do not tune on it.
- **Train/serve preprocessing must match.** `backend/features.py::url_to_model_text` is a copy of the function defined in notebook 03. It lowercases the hostname, strips `www.`, drops the scheme, and keeps path, query and fragment. If you change either copy, change the other too and retrain or re-save the model.
- **Backend (`backend/main.py`)** loads the model at import time. `POST /predict {url}` prepends `https://` if no scheme is present, validates the URL as http(s) with a hostname and no whitespace (returns 400 otherwise), and returns `{prediction, label, phishing_score, model_input}` (`model_input` is the preprocessed text the model saw; the frontend shows it but never shows the score). `THRESHOLD = 0.5` is applied to the phishing probability.
- **Frontend (`frontend/src/App.tsx`)** is a single component that POSTs to `/api/predict` and shows `label` or the backend's `detail` error message. It depends on the Vite dev proxy; there is no production API base URL configured.
