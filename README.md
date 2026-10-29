# DASE7506 Mini-project 2: Selecting Reliable LLM Reasoning

Student ID: **3036706296**

Official test result: **391/467 = 83.72591006423983%**

Frozen submission tag: `mp2-final-3036706296`

This repository contains the submitted selector, its frozen inference checkpoint,
training code, official evaluation records, ablations, and the report. The method
selects one of four supplied Qwen3-0.6B GSM8K solutions using centered fixed
embeddings, answer agreement, a numeric-answer residual, a 32-prototype semantic
calibrator, arithmetic checks, and a bounded expression-coverage check.

## Submitted artifacts

- Entry point: `code/submission.py`
- Selector and checks: `code/selector.py`, `code/checks.py`
- Training: `code/train.py`
- Frozen checkpoint: `code/checkpoint/model_bundle/`
- Report: `report/REPORT.md`
- Official test output: `report/test-score.json`
- Official validation output: `report/validation-score.json`
- Experiment records: `report/validation_experiments.json`,
  `report/prototype-experiment.json`
- Later cross-validated two-stage experiment (not the leaderboard entry;
  official test score 385/467 = 82.44%): `experiments/two_stage/`, with its
  own README, checkpoint, training script and records. The report explains
  why it is reported but not submitted.

Checkpoint size: **147,098 bytes**

| File | SHA-256 |
|---|---|
| `code/checkpoint/model_bundle/model.npz` | `c6819b866f7324f5050fb5a299b26fc226c07dd37a37f1f6ab7165fc0c2c8913` |
| `code/checkpoint/model_bundle/config.json` | `bed432f8ada4f78e14e336455e8991d8c4fe5428d44b028245056fdc6e00c679` |

## Checkpoint-only reproduction

Python 3.12 is recommended. Run from `code/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

python predict.py --method submission.py \
  --checkpoint checkpoint/model_bundle \
  --split test --output runs/test-predictions.json

python score_predictions.py --split test \
  --predictions runs/test-predictions.json \
  --output runs/test-score.json
```

The resulting `accuracy` should be `83.72591006423983`.

## Training reproduction

Training reads only the supplied training and validation splits. It never opens
the test split.

```bash
python -m pip install -r training-requirements.txt
python train.py
```

Seed 0 reproduces the checkpoint hashes shown above. Scikit-learn is used only
for training the 32 mini-batch k-means centers; frozen inference requires only
NumPy.

## Resource use

Measured on an Apple M5 with four library threads:

- Complete test prediction process: 2.19 seconds wall time
- Independent scoring: 0.013 seconds
- Training: about 19 seconds
- Inference assets: 147,098 bytes

These are below the course limits of 30 seconds and 16 MiB.

## Protocol and integrity

- Protocol: `dase7506-mp2-open-hard-verifier-v5`
- Training references are used only for fitting.
- Validation references are used for development and method selection.
- Test references are used only by the independent final scorer.
- No external dataset, additional pretrained weights, LLM calls, ID lookup
  table, or cached test predictions are used by the selector.
- Supplied data, parser, scorer, and runner are unchanged.

## AI assistance

Cursor's coding assistant provided substantive help with implementation,
experiments, and report drafting. The full disclosure is in
`report/REPORT.md`. The student is responsible for understanding and submitting
the work.
