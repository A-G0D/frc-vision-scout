# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for every pipeline step. Each target is something CLAUDE
# can run via Bash with NO browser interaction (after one-time `make auth`).
#
# Pipeline order:   data -> autolabel -> review -> split -> train -> export -> eval -> gate
# Deploy:           make run / make bench  (on the Intel laptop)
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help auth deps data tba frames autolabel review split train export pull eval gate run bench clean

PY ?= python
CONF ?= configs/deploy.yaml

help:                       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

# ── One-time setup (the ONLY steps that may touch a browser) ──────────────────
auth:                       ## One-time: create cloud tokens. Run via `! make auth` so output lands in session.
	@echo "1) Modal (cloud GPU, no browser after this): modal token new"
	@echo "2) (optional) HF:  huggingface-cli login"
	@echo "3) (optional) TBA: export TBA_AUTH_KEY=...   # or rely on HTML-scrape path"

deps:                       ## Install local (deploy + ops) deps
	$(PY) -m pip install -e ".[ops,dev]"

# ── Data acquisition (fully automated — yt-dlp + TBA, no browser) ─────────────
data: frames tba           ## Fetch footage, extract frames, fetch TBA ground truth

footage:                    ## Download match footage (edit configs/sources.yaml)
	$(PY) scripts/fetch_footage.py --config configs/sources.yaml

frames: footage            ## Sample frames for labeling
	$(PY) scripts/extract_frames.py --config configs/sources.yaml

tba:                        ## Fetch TBA ground truth (HTML-scrape fallback, no key needed)
	$(PY) scripts/fetch_tba.py --event 2026ctwat --out data/tba/2026ctwat.json

# ── Labeling (teacher draft -> Claude/human review -> match-level split) ──────
autolabel:                  ## Cloud GPU: teacher ensemble draft pseudo-labels
	modal run train/modal_app.py::autolabel

review:                     ## Claude-vision / human correction pass on a sampled subset
	$(PY) scripts/label_review.py --config configs/review.yaml

split:                      ## Match-level train/val/test split (NO frame leakage)
	$(PY) scripts/split_by_match.py --config configs/sources.yaml

# ── Training: KAGGLE (chosen path — free T4, no card) ─────────────────────────
# With a Kaggle API token (~/.kaggle/kaggle.json) Claude drives all of these via Bash.
# Without it, do the browser version: upload data/footage as a Dataset, open
# kaggle/frc_train_kaggle.ipynb, set GPU T4x2 + Internet On, Run All, download the zip.
nb:                         ## (Re)generate the Kaggle notebook from the builder
	$(PY) kaggle/build_notebook.py

kaggle-up:                  ## Upload footage as a Kaggle Dataset (API)
	$(PY) scripts/kaggle_drive.py up

kaggle-push:                ## Push + run the notebook headless (GPU + internet, API)
	$(PY) scripts/kaggle_drive.py push

kaggle-status:              ## Poll the Kaggle run state
	$(PY) scripts/kaggle_drive.py status

pull:                       ## Download the trained OpenVINO zip from the Kaggle run -> weights/
	$(PY) scripts/kaggle_drive.py pull

# ── Training: MODAL (alternative — hands-off but uses credits) ────────────────
train-modal:                ## Cloud GPU via Modal: train ball + robot students
	modal run train/modal_app.py::train_all
export-modal:               ## Cloud GPU via Modal: export OpenVINO INT8 per device
	modal run train/modal_app.py::export_all

# ── Evaluation + release gate (CPU, local, no GPU needed) ─────────────────────
eval:                       ## Run all metrics vs HUMAN labels + overlay + TBA
	$(PY) -m frcscout.eval.gates --config configs/eval.yaml --report

gate:                       ## CI release gate — nonzero exit if any threshold fails
	$(PY) -m frcscout.eval.gates --config configs/eval.yaml --enforce

# ── Deploy (the Intel laptop) ─────────────────────────────────────────────────
run:                        ## Run the live 60fps engine
	$(PY) -m frcscout.pipeline.engine --config $(CONF) --source $(SOURCE)

bench:                      ## Benchmark device map + p99 frame time vs 16.6ms budget
	$(PY) -m frcscout.pipeline.engine --config $(CONF) --source $(SOURCE) --benchmark

clean:                      ## Remove regenerable local artifacts
	rm -rf runs weights/*_openvino_model data/frames renders
