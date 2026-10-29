# Selecting Reliable LLM Reasoning

DASE7506 Mini-project 2. Student ID: 3036706296.

The submitted score is the official scorer’s `accuracy` field on the complete test bank: **83.72591006423983**. Enter that number, with student ID **3036706296**, on the MP2 website. The checkpoint that produces it is `code/checkpoint/model_bundle`. Its SHA-256 hashes are recorded below. Test questions were not used to train, tune, or choose this checkpoint.

## Question

A small language model has already written four solutions for each GSM8K question. The scorer marks a choice correct when the selected solution’s final number matches the reference, including equivalent fractions and decimals. It does not check whether every intermediate step is valid. Every question has at least one correct final answer, and more than one of the four can be correct.

The test bank is constructed differently from the training and validation banks. In validation, 443 of 798 questions have only one distinct parsed answer. On those questions, that answer is necessarily the correct one, so any method that can read the final number scores 100% there. The test bank has no such questions: every test question has at least two distinct parsed answers. A validation score computed on all 798 questions therefore mixes an easy set with the set that actually resembles the test.

The experiment asks which evidence lets us choose among disagreeing solutions, and whether that evidence is the supplied representation of the trace, agreement among the final numbers, or surface properties such as length and how many question numbers are repeated.

## Method

The submitted selector gives each candidate a linear score and returns the highest-scoring candidate that has a parsed final number. Ties go to the smaller index.

For a question, the four supplied embeddings are each 1024-dimensional. Before scoring, each embedding has the mean of the four subtracted. The score then depends on how a candidate differs from the other three, which is the comparison the project asks about. Two scalar features are appended:

- **Agreement.** The fraction of the four candidates whose parsed final number equals this candidate’s number. A candidate with no parsed number scores 0. The official scorer cannot mark an unparsed candidate correct, so unparsed candidates are also removed at decision time.
- **Parseable.** 1 when a final number can be parsed, and 0 otherwise.

Features are standardized with the mean and standard deviation computed on all training candidates. The weight vector has 1,026 entries. It is fit by least squares so that, on a training pair consisting of one correct and one incorrect candidate, the correct candidate receives a higher score. Questions on which every candidate is correct, or every candidate is incorrect, supply no such pair and are not used in the fit. There are 8,271 pairs. With pair-difference vectors stacked as \(D\), the weights solve

\[
(D^\top D + \lambda I) w = D^\top \mathbf{1}, \quad \lambda = 1 \times 8271.
\]

The linear score is supplemented by a 17-dimensional numeric-answer residual. Its features describe the parsed answer’s sign, integrality, denominator and logarithmic magnitude; its ratios to the minimum, maximum, sum and product of the question numbers; its rank among the four proposed answers; and whether it copies or falls outside the question’s numeric range. A second pairwise ridge fit predicts the residual left by the embedding score. The selected residual uses relative ridge multiplier 0.01 and contributes at scale 0.5. These are generic numeric relationships rather than answer lookups.

A small semantic-prototype residual adds local evidence that a single global direction cannot represent. Mini-batch k-means partitions the centered training embeddings into 32 prototypes. A 32-entry ridge table learns the residual reliability of each prototype after the linear, numeric, and arithmetic scores. At inference, a candidate receives the table entry of its nearest prototype at scale 0.175. The centers and table are fitted only on training data. This scale was accepted because it improved both the fixed five-fold training check and hard validation before the test run.

Two checks are then applied. Both were chosen on hard validation, from a grid fixed before the test run. A false explicit equation, including a false fraction equality such as \(0.6/9 = 2/15\), subtracts 0.2 from that candidate. Separately, if exactly one parsed answer can be obtained from at least four of the question’s own numbers by addition, subtraction, multiplication and division, and that candidate’s score is no more than 0.2 behind the current leader, the selector switches to it. The expression search uses at most the first six numbers in the question and keeps at most twelve values per subset, so it stays inside the time limit.

No external data, extra pretrained weights, or language-model calls are used. The inference asset contains the small weight vectors, standardization statistics, 32 prototype centers, and `config.json`.

## Setup

Training uses only `gsm8k_train`. The feature set and the ridge multiplier are chosen on validation. The pre-specified feature sets are: agreement alone; centered embedding alone; centered embedding plus agreement; the same embedding without centering; and the centered model with either solution length or coverage of the numbers appearing in the question. For each set, the multiplier is chosen from \(\{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}\).

The selection score is accuracy on the 355 validation questions with at least two distinct parsed answers. If another pre-specified set is within one question of the best score, the simpler set is kept. This rule matters: at the chosen multiplier, adding length or number coverage ties the centered agreement model at 297/355, so the simpler model is kept. The numeric residual raises this to 302/355 before the later corrections. As an additional training-only check, five-fold out-of-fold accuracy on the 2,114 mixed-answer training questions rises from 1,690 to 1,698 when the residual is added. Later experiments used a stricter gate: a new component had to improve both five-fold training accuracy and hard validation. The semantic-prototype residual passed at 1,705 rather than 1,704 training questions and 307 rather than 306 hard-validation questions in the final pipeline.

The checkpoint was frozen before its reported test scoring. Seed is 0. The implementation uses Python 3.12.15, NumPy 2.5.3, and scikit-learn 1.7.2 for training k-means only. Inference imports only NumPy. Training is a single CPU process and took about 19 seconds on the machine below, including the validation grids.

## Results

Official test scores, from `score_predictions.py`:

| Method | What it uses | Correct | Accuracy |
|---|---|---:|---:|
| Majority vote | most common final number | 323/467 | 69.16% |
| Agreement model | learned weights on agreement only | 323/467 | 69.16% |
| Centered embedding | representation only | 374/467 | 80.09% |
| Absolute embedding + agreement | each trace scored on its own | 374/467 | 80.09% |
| Centered embedding + agreement | linear score only | 381/467 | 81.58% |
| Linear score + length | adds log text length | 381/467 | 81.58% |
| Linear score + number overlap | adds overlap with question numbers | 381/467 | 81.58% |
| Linear score + equation check + expression coverage | previous system | 387/467 | 82.87% |
| **Numeric + semantic-prototype residuals + two checks** | **submitted method** | **391/467** | **83.73%** |

Length changes none of the 467 test choices. Number coverage changes one choice and does not change the number correct. Majority and the learned agreement-only model make the same choices.

Validation, using the same frozen predictors. “Hard” means at least two distinct parsed answers. “All” is the official validation score.

| Method | Hard validation | All validation |
|---|---:|---:|
| Majority / agreement only | 245/355 = 69.01% | 688/798 = 86.22% |
| Longest parsed solution | 212/355 = 59.72% | 82.08% |
| Highest coverage of question numbers | 215/355 = 60.56% | 82.46% |
| Centered embedding | 282/355 = 79.44% | 90.85% |
| Absolute embedding + agreement | 288/355 = 81.13% | 91.60% |
| Centered embedding + agreement | 297/355 = 83.66% | 740/798 = 92.73% |
| Linear score + two checks | 303/355 = 85.35% | 746/798 = 93.48% |
| Numeric residual + two checks | 306/355 = 86.20% | 749/798 = 93.86% |
| **Submitted method with semantic prototypes** | **307/355 = 86.48%** | **750/798 = 93.98%** |

Both majority and the submitted method answer all 443 single-answer validation questions correctly (688 − 245 = 443, and 750 − 307 = 443). The jump from 69% to 93% on the full validation bank is mostly that easy subset. The relevant comparison is the hard subset. The linear score reaches 83.66% there and 81.58% on test. The numeric residual and two checks raise hard validation to 86.20% and the semantic prototypes add one more validation answer. The final official test score is 83.73%; the prototype correction changes some choices but has zero net effect on test accuracy.

On the 89 hard validation questions where the linear score and majority select different numbers, the linear score is the only correct one 66 times, majority is the only correct one 14 times, and both are wrong 9 times.

Within a hard validation question, the embedding part of the score varies by 0.66 on average between the highest and lowest candidate. The agreement part varies by 0.16. The representation does most of the separation; agreement is a smaller correction. That matches the test table: the embedding alone reaches 80.09%, and agreement adds the remaining 1.5 points.

## What this says

**The supplied representation carries most of the improvement.** A linear score on the centered embedding, with no access to the final-number vote, already selects a correct test answer on 80.09% of questions. Agreement alone stays at the majority rate, 69.16%.

**Comparing the four traces is useful.** Subtracting the question’s mean embedding outperforms scoring each absolute embedding together with agreement: 81.58% against 80.09% on test for the linear scores, and 297 against 288 on hard validation. The two predictors differ on 108 test questions.

**Length and repetition of the question’s numbers do not explain the result.** Choosing the longest solution, or the solution that repeats the most numbers from the question, is worse than majority on hard validation (about 60%). Adding either feature to the submitted linear score does not improve test accuracy. The selector is not using a longer or more complete-looking write-up as a proxy for correctness.

**Numeric plausibility adds complementary evidence.** The residual improves hard validation from 297 to 302 before the checks and improves five-fold training accuracy in aggregate, rather than depending on one validation example. In the final pipeline it adds four test answers beyond the previous equation-and-coverage system.

**Local semantic calibration is weak but reproducible on development data.** The 32-prototype residual improves the fixed five-fold check by one question and hard validation by one question. It does not increase the frozen test score. This is evidence that the effect is small; the component is retained because it passed the pre-test gate rather than because of its test result.

**The extra computation is small.** On this laptop the official test prediction process took 2.19 seconds of wall time and scoring took 0.013 seconds, with four threads. The submitted asset is 147,098 bytes. Both are far inside the 30-second and 16 MiB limits. Relative to majority, the submitted method answers 68 more test questions correctly (391 − 323). Adding length or number-overlap features to the linear score does not buy more accuracy.

## A cross-validated two-stage model, and what it showed

The submitted pipeline's later components (numeric residual scale, arithmetic penalty, coverage override, prototype scale) were chosen on the 355 hard validation questions. That is a small set, and a grid of settings chosen on it is optimistic. After the submission was frozen, a second method was built under a stricter protocol: every choice by five-fold cross-validation on the 2,114 mixed-answer training questions (three fold assignments, 6,342 question-evaluations), with validation scored once afterwards as a held-out check and test scored once for the frozen checkpoint. The method, code, checkpoint and records are in `experiments/two_stage/`; it is not the leaderboard entry.

**Method.** Stage 1 replaces least squares with pairwise logistic regression and drops pairs involving unparsed candidates, since the selector never returns them. Two stage-1 rankers are fit: on the centered embedding and on the raw embedding. Stage 2 is a second pairwise logistic ranker over 166 standardized features: the two stage-1 scores, agreement, 17 numeric-answer features, and 73 features of the candidate text, each also centered within the question. The text features include agreement of a trace's intermediate numbers with the other three traces (Jaccard overlap of the numbers that are not in the question), how many of its intermediate numbers no other trace produces, counts of false explicit equations, whether the final number appears in the body of the solution, length and line counts, and capped counts of 46 words and symbols (`wait`, `however`, `assume`, `✅`, `\boxed`, …). Stage 2 is trained on out-of-fold stage-1 scores, so it learns how far to trust the embedding score at its test-time reliability rather than at its in-sample fit. Inference needs only NumPy; the checkpoint is 31,214 bytes; test prediction took 0.72 s.

**Cross-validation evidence.** Totals over three fold seeds, hard training questions correct out of 6,342 (`experiments/two_stage/records/comparison.json`):

| Method | Train CV (3 seeds) | Hard validation | Test |
|---|---:|---:|---:|
| Submission: linear least squares (centered + agreement) | 5,125 (80.8%) | 297 | 381 |
| Submission: + numeric residual | 5,140 (81.0%) | 302 | — |
| **Submission: + penalty + coverage override (frozen entry)** | **5,113 (80.6%)** | **307** | **391** |
| Two-stage: stage 1 only, centered logistic | 5,140 (81.0%) | 295 | 380 |
| Two-stage: stage 2 without text features | 5,136 (81.0%) | 298 | — |
| Two-stage: stage 2 without numeric features | 5,123 (80.8%) | 293 | — |
| Two-stage: stage 2 with the centered score only | 5,196 (81.9%) | 300 | — |
| **Two-stage: full** | **5,204 (82.1%)** | **299** | **385** |

Three things follow. First, the two linear embedding rankers are equivalent: least squares and logistic agree within noise in cross-validation (5,125 against 5,140) and on test (381 against 380). The embedding information is captured by one linear direction; a hidden layer, listwise softmax, cosine features, and a k-nearest-neighbour residual were also tried and none exceeded this in cross-validation. Second, the stage-2 combination is a real improvement by the stricter protocol: +64 cross-validated questions over the stage-1 ranker, roughly one point, and both the text block and the numeric block are needed (removing either returns the score to the stage-1 level). On test the same component adds five questions, 380 to 385, which is the gain cross-validation predicted for a 467-question bank. Third, the submission's validation-chosen refinements do not survive cross-validation. The numeric residual, penalty and coverage override together add 10 test questions (381 to 391) and 10 hard-validation questions, where they were chosen, but cost 12 of 6,342 cross-validated questions (5,125 to 5,113); the penalty and override alone cost 27 relative to the numeric residual. Cross-validation says that the 391 contains a favourable draw.

**Why the lower-scoring method is reported.** The two frozen selectors choose a different final number on 48 of the 467 test questions. With 48 discordant questions, the standard error of the difference in correct counts is about six, and the observed difference is six. The test bank cannot tell these two methods apart. The cross-validation evidence prefers the two-stage model by about one point; the test draw prefers the submission by about one point. The leaderboard entry remains the earlier frozen checkpoint, which was frozen and scored first, and the two-stage result is reported rather than used to pick between them. The practical lesson for this task is that a 355-question development set is too small to select among refinements worth one or two points, and that the single most reliable gain available beyond the linear embedding score comes from combining it with cheap text and numeric evidence in a second stage trained on out-of-fold scores.

**Calibration improves even though accuracy does not.** The logistic two-stage score orders its own mistakes. Splitting hard validation into quartiles by the gap between the best and second-best score gives accuracies of 79.8%, 84.1%, 84.3% and 88.8%, increasing with the gap. For the submitted least-squares score the same bins are 84.3%, 76.1%, 87.6% and 86.5% (next section). On the agreement patterns, the two-stage model's hard-validation accuracy is 134/151 when three traces agree against one, 70/88 for a 2-1-1 split, 30/34 for a 2-2 tie, and 14/20 when all four numbers differ; majority vote scores 129, 54, 14 and 5 on the same groups. The largest learned stage-2 weights, on standardized features, are the centered embedding score (0.68), agreement (0.49), whether the answer is an integer (0.42), the raw embedding score (0.33), and negative weights for a zero or negative answer and for a non-integer denominator; among text features, having more distinct intermediate numbers than the other traces is positive and having intermediate numbers that no other trace shares is negative.

## Where it succeeds and fails

When the four final numbers are all different, majority has no vote and returns the lowest index. One validation question illustrates the difference. Jay watches a concert of 1 hour 20 minutes with a 10-minute intermission. Every song is 5 minutes except one song of 10 minutes. The singing time is \(80 - 10 = 70\) minutes, and \(5(n-1) + 10 = 70\) gives \(n = 13\). The four candidates answer 16, 14, 13, and 12. Majority returns 16, from the candidate that treats 1 hour 20 minutes as 90 minutes and divides by 5. The linear score returns the candidate that sets up the equation above.

A correct final number is not the same as valid reasoning. On a validation question, a plant produces 2 baby plants 2 times a year for 4 years, so the reference answer is 16. Two candidates answer 4, and majority follows them. The selected candidate also answers 16, which the scorer accepts, but its visible calculation doubles the count each year rather than multiplying \(2 \times 2 \times 4\). The method is trained and scored on the final number, so it is credited for a trace whose intermediate story is not the right one.

The agreement vote should not always be overridden. Megan has 2 dozen eggs, breaks 3, and cracks twice that many. The correct difference between intact and cracked eggs is \(24 - 3 - 6 = 15\), then \(15 - 6 = 9\). Majority selects that answer. The linear score selects a candidate that ends at 6. This direction of error happens on 14 of the 89 hard validation disagreements between the linear score and majority.

## Uncertainty

The gap between the best and second-best score was split into quartiles on hard validation. Accuracy in those bins is 84.3%, 76.1%, 87.6%, and 86.5%. The largest gaps are not reliably the most accurate, and the second quartile is the least accurate. The least-squares score margin is not a usable confidence value. The method does rank candidates well enough to raise accuracy; it does not know when that ranking is fragile. The logistic two-stage score of the previous section does better here: its quartile accuracies rise monotonically from 79.8% to 88.8%, so a pairwise logistic loss gives a margin that can be read as confidence even though it does not raise accuracy on test.

## Resources and reproduction

Measured locally on an Apple M5, 10 cores, 16 GB RAM, macOS 26.6. The course limit is 30 seconds for prediction plus scoring on a Xeon Platinum 8457C with four threads in total. This machine was not that Xeon. The local measurement, with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` set to 4, is:

| Stage | Wall time |
|---|---:|
| `predict.py` on the full test bank | 2.19 s |
| `score_predictions.py` on those predictions | 0.013 s |
| Training, including both validation grids | about 19 s |

Submitted asset, `code/checkpoint/model_bundle`, 147,098 bytes uncompressed:

| File | SHA-256 |
|---|---|
| `model.npz` | `c6819b866f7324f5050fb5a299b26fc226c07dd37a37f1f6ab7165fc0c2c8913` |
| `config.json` | `bed432f8ada4f78e14e336455e8991d8c4fe5428d44b028245056fdc6e00c679` |

Ablation checkpoints under `code/checkpoint/ablations/` are for the table above. The evaluation command loads only `model_bundle`. The two-stage experiment is self-contained in `experiments/two_stage/` (code, 31,214-byte checkpoint, training script, official score files and cross-validation records); its README gives the commands. Its training took about 80 s on the same machine.

From `code/`, with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r training-requirements.txt
python check_data.py
python train.py

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
python predict.py --method submission.py --checkpoint checkpoint/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output ../report/test-score.json
```

`train.py` reads the training and validation splits only. Re-running it reproduces the checkpoint; the test command above does not retrain. Official outputs are `report/test-score.json` and `report/validation-score.json`. The validation grid, disagreement counts, and margin bins are in `report/validation_experiments.json`; the pre-test prototype scale gate is in `report/prototype-experiment.json`.

## AI assistance and reuse

The course data, numeric parser, prediction runner, and scorer are used unchanged. No other external dataset or model weights are used.

Substantive AI assistance: the implementation in `code/selector.py`, `code/checks.py`, `code/train.py`, and `code/submission.py`, the two-stage experiment in `experiments/two_stage/`, the experiment grids, the cross-validation comparison, and this report were drafted with Cursor’s coding assistant. The submitted behaviour is exactly what those files and the official scorer implement.
