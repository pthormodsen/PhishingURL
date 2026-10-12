# Phishing URL Detector

DAT158 machine learning project. A website where a user enters a URL and
gets a machine learning based assessment of whether the URL looks like
phishing. The model reads only the URL text. It never visits the website.

- **Model:** character 3-5 gram TF-IDF + logistic regression
  (`SGDClassifier(loss="log_loss")`), trained on the PhreshPhish dataset.
- **Final test (10,000 URLs, run once):** accuracy 0.906, phishing precision
  0.977, phishing recall 0.808, F1 0.885, false positive rate 1.5%.
- **Known weakness:** popular homepages such as `google.com` are flagged
  about 17-19% of the time, and look-alike domains are not reliably caught.
  See `notebooks/05_summary_and_final_evaluation.ipynb`.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/01` to `04` | Exploration, first models, PhreshPhish model, homepage experiment |
| `notebooks/05_summary_and_final_evaluation.ipynb` | Summary of all experiments and the final test |
| `experiments/` | Reproducible experiment scripts (`e00` to `e08`) and shared code |
| `scripts/prepare_phreshphish.py` | Creates the fixed PhreshPhish samples, including the reserved test set |
| `models/phresh_text_model.joblib` | The deployed model |
| `backend/` | FastAPI service (`POST /predict`) |
| `frontend/` | React + TypeScript + Vite website, Nginx configuration |
| `compose.yaml`, `Makefile` | Production deployment |

Labels: **0 = phishing, 1 = legitimate**.

## Local development

```sh
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Backend on http://127.0.0.1:8000 (run from the repository root)
.venv/bin/uvicorn backend.main:app --reload

# Frontend on http://localhost:5173, proxies /api to the backend
cd frontend && npm install && npm run dev
```

To reproduce the data and experiments:

```sh
.venv/bin/python scripts/prepare_phreshphish.py   # fixed samples, pinned dataset revision
.venv/bin/python experiments/prepare_data.py      # full train split and Tranco list
cd experiments && ../.venv/bin/python e01_baselines.py   # and so on, see experiments/
```

Do not rerun `experiments/e08_final_test.py`. The final test has been used
once, and the script refuses to run again.

## Deployment on docker01

### Architecture

```
Internet -> Cloudflare Tunnel -> [cloudflare network] -> frontend (Nginx :80)
                                                             |  /api/predict
                                         [internal network]  v
                                                          backend (FastAPI :8000)
```

- **frontend:** Nginx serves the built React app and proxies
  `/api/predict` to the backend `/predict`. It is the only service on the
  external `cloudflare` network, where it has the alias `phishing`.
- **backend:** FastAPI with Uvicorn (no `--reload`). It is only on the
  internal Compose network, which has no internet access, and it runs as a
  non-root user with a read-only filesystem.
- **No host ports are published.** The site is reachable only through the
  tunnel.
- **What the backend image contains:** `models/phresh_text_model.joblib`
  and `backend/features.py`, so the model, the threshold (0.5) and the
  preprocessing are identical to development. `backend/requirements.txt`
  pins the exact library versions the model was saved with.

### Cloudflare Tunnel target

In the Cloudflare Tunnel configuration, point the public hostname to:

```
http://phishing:80
```

The tunnel container (`cloudflared`) must be attached to the same external
Docker network, `cloudflare`.

### First startup

Requirements on docker01: Docker with the Compose plugin, `make`, `git`,
and the existing external network `cloudflare`.

```sh
git clone <repository-url> phishing-url-detector
cd phishing-url-detector
make deploy        # checks the network, builds, starts, waits for health, runs a smoke test
make smoke-tunnel  # tests http://phishing:80 exactly as the tunnel reaches it
```

A successful smoke test prints a JSON prediction, for example:

```json
{"prediction":1,"label":"Not flagged by the model","phishing_score":0.47...,"model_input":"example.com"}
```

### Everyday commands

| Command | Purpose |
|---|---|
| `make ps` | Container status and health |
| `make logs` | Follow logs from both services |
| `make restart` | Restart both containers |
| `make down` | Stop and remove the containers |
| `make check` | Validate `compose.yaml` |
| `make help` | List all targets |

### Updating

```sh
make update        # git pull --ff-only, rebuild, restart, smoke test
make clean         # optional: remove old dangling images
```

Nginx looks up the backend address at request time, so a recreated backend
container is found again automatically. Requests can fail for up to about
10 seconds while the DNS cache expires.

To replace the model, save the new model as `models/phresh_text_model.joblib`
(with matching preprocessing in `backend/features.py` and matching versions in
`backend/requirements.txt`), then run `make deploy`.

### Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Network 'cloudflare' not found` | The external network is missing. It belongs to the Cloudflare Tunnel setup; check with `docker network ls`. For a local test only: `docker network create cloudflare`. |
| Website shows "The service is not responding" | Backend unhealthy or still starting. Check `make ps` and `docker compose logs backend`. |
| Cloudflare shows error 502 or 1033 | The tunnel cannot reach `phishing:80`. Check that `cloudflared` is on the `cloudflare` network (`docker network inspect cloudflare`) and that the target is `http://phishing:80`. Run `make smoke-tunnel`. |
| `make deploy` hangs on "Waiting" | A health check fails. `docker compose logs backend` usually shows the model loading error. |
| Backend fails with an unpickling or version error | Library versions differ from the ones the model was saved with. Keep `backend/requirements.txt` unchanged, or retrain and save the model with the new versions. |
| Nginx returns 502 right after an update | The backend is restarting. Wait a few seconds; if it persists, check `docker compose logs backend`. |
| `permission denied ... docker.sock` | Your user is not in the `docker` group, or the session started before it was added. Log out and in again. |
| Port conflict | Not possible: no host ports are published. |

## Data and references

- **PhreshPhish:** T. Dalton et al., "PhreshPhish: A Real-World, High-Quality,
  Large-Scale Phishing Website Dataset and Benchmark", arXiv:2507.10854, 2025.
  Dataset `phreshphish/phreshphish` on Hugging Face (revision `eabec4b`),
  CC BY 4.0.
- **PhiUSIIL:** A. Prasad and S. Chandra, "PhiUSIIL: A diverse security profile
  empowered phishing URL detection framework based on similarity index and
  incremental learning", Computers & Security 136, 2024. UCI Machine Learning
  Repository. Used in notebooks 01 and 02 only.
- **Tranco:** V. Le Pochat et al., "Tranco: A Research-Oriented Top Sites
  Ranking Hardened Against Manipulation", NDSS 2019. List `Y83YG`, used only
  to evaluate false alarms on popular homepages.
- **Tools:** scikit-learn, FastAPI, React, Vite, Nginx.

## AI assistance

Parts of the experiment code, evaluation scripts, web application changes
and deployment configuration were developed with help from Claude
(Anthropic) through Claude Code. All results were checked by running the
code, and the modelling decisions are documented in notebook 05.
