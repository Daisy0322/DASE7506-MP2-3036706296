# Experiment: cross-validated two-stage ranker

This folder is a complete, separately frozen method. It is **not** the
leaderboard submission; the submission is `code/` with
`code/checkpoint/model_bundle`. It is kept because it was developed under a
stricter protocol (all choices by cross-validation on the training split) and
because its result is part of the report's analysis.

Official test score of this method: **385/467 = 82.44111349036403%**
(`records/test-score.json`). Submission method: 391/467 = 83.73%.

## Method

1. **Stage 1.** Two pairwise logistic rankers on the 1024-d fixed embedding
   plus agreement and parseable flags: one on the embedding centered within
   the question (ridge 0.1), one on the raw embedding (ridge 0.03). Pairs are
   (correct, incorrect) candidates of the same question; unparsed candidates
   are excluded from pairs because the selector never returns them.
2. **Stage 2.** A pairwise logistic ranker on 166 standardized features:
   the two stage-1 scores, agreement, 73 features of the candidate text (raw
   and centered within the question), and the 17 numeric-answer features from
   the submission. Stage 2 is trained on **out-of-fold** stage-1 scores so it
   sees the stage-1 scores at their test-time reliability.
3. Ridge strengths are chosen by five-fold cross-validation on the 2,114
   mixed-answer training questions, summed over three fold assignments.
   Validation is scored afterwards as a held-out check only.

The trace features are listed in `checkpoint/model_bundle/config.json`
(`trace_feature_names`). The strongest learned stage-2 weights are in
`records/validation_experiments.json` under `stage2_weights`.

## Reproduce

From `code/` with its virtual environment active (NumPy only):

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
python predict.py --method ../experiments/two_stage/submission.py \
  --checkpoint ../experiments/two_stage/checkpoint/model_bundle \
  --split test --output runs/two_stage-test-predictions.json
python score_predictions.py --split test \
  --predictions runs/two_stage-test-predictions.json \
  --output runs/two_stage-test-score.json
```

Training (about 80 s on an Apple M5, seed 0, reproduces the hashes below):

```bash
cd ../experiments/two_stage && python train.py
```

Checkpoint: 31,214 bytes.

| File | SHA-256 |
|---|---|
| `checkpoint/model_bundle/model.npz` | `0bcdc98b832ae5a6aef0e408213fe79b01991a60a9f4f0e9718cb794a3f47555` |
| `checkpoint/model_bundle/config.json` | `0069dbfb71ecde9ae42d229d3e77ae7bc8177478ff5c8130589cd66e619e93ef` |

Prediction on the full test bank took 0.72 s wall time with four threads.

## Records

- `records/validation_experiments.json`: stage-1 and stage-2 ridge grids with
  per-seed cross-validation, ablations, validation breakdowns, stage-2 weights.
- `records/comparison.json`: side-by-side cross-validation / validation / test
  numbers for the submission pipeline and this one.
- `records/test-score.json`, `records/validation-score.json`: official scorer
  output for `checkpoint/model_bundle`.
- `records/test-stage1_centered-score.json`: official test score of the
  stage-1-only ablation (380/467).
- `checkpoint/ablations/`: stage-1-only, single-view, and block-removed
  stage-2 variants used in the ablation table.
