"""Fit the two-stage candidate ranker on the training split.

Hyper-parameters (ridge strengths) are chosen by five-fold cross-validation on
the training split alone. The validation split is scored afterwards as a
held-out check and for the report. This script never opens the test split;
test scoring is a separate, later command that uses the official runner.

Inference requires only NumPy. Training also requires only NumPy.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent.parent / 'code'
sys.path.append(str(CODE))  # official common.py (parser, loader) lives in code/

from common import load_inputs, parse_answer, read_jsonl  # noqa: E402
from selector import (  # noqa: E402
    STAGE2_BLOCKS,
    apply_scaler,
    choose,
    choose_indices,
    load_checkpoint,
    majority_choices,
    numeric_answer_features,
    stage1_features,
    stage2_features,
    text_signals,
    trace_features,
)

DATA = CODE / 'data'
CHECKPOINT = ROOT / 'checkpoint'
RECORDS = ROOT / 'records'
SEED = 0
FOLDS = 5
CV_SEEDS = (0, 1, 2)  # fold assignments used for model selection
STAGE1_MODES = ('centered', 'absolute')
STAGE1_LAMBDAS = (0.03, 0.1, 0.3, 1.0)
STAGE2_LAMBDAS = (0.03, 0.1, 0.3, 1.0)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_split(split):
    if split not in ('train', 'validation'):
        raise ValueError('train.py only loads training and validation data')
    bank, arrays = load_inputs(split, DATA)
    gold = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    if [row['id'] for row in gold] != [row['id'] for row in bank]:
        raise ValueError('Reference alignment mismatch')
    return bank, arrays['embeddings'].astype(np.float64), gold


def correctness(bank, gold):
    labels = np.zeros((len(bank), 4), dtype=bool)
    for i, (row, reference) in enumerate(zip(bank, gold)):
        target = parse_answer(reference['answer'])
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            labels[i, j] = answer is not None and answer == target
    return labels


def distinct_answer_counts(bank):
    counts = np.zeros(len(bank), dtype=np.int64)
    for i, row in enumerate(bank):
        answers = {parse_answer(candidate['text']) for candidate in row['candidates']}
        answers.discard(None)
        counts[i] = len(answers)
    return counts


def selection_hits(bank, gold, choices):
    hits = np.zeros(len(bank), dtype=bool)
    for i, (row, reference, choice) in enumerate(zip(bank, gold, choices)):
        target = parse_answer(reference['answer'])
        answer = parse_answer(row['candidates'][choice]['text'])
        hits[i] = answer is not None and answer == target
    return hits


def summarize(hits, mask):
    chosen = hits if mask is None else hits[mask]
    total = int(chosen.size)
    correct = int(chosen.sum())
    accuracy = 100.0 * correct / total if total else 0.0
    return {'correct': correct, 'n': total, 'accuracy': accuracy}


def folds(n, k=FOLDS, seed=SEED):
    permutation = np.random.RandomState(seed).permutation(n)
    return [np.sort(permutation[i::k]) for i in range(k)]


def complement(n, fold):
    mask = np.ones(n, dtype=bool)
    mask[fold] = False
    return np.flatnonzero(mask)


# ----------------------------------------------------------------------------
# Pairwise logistic ranking
# ----------------------------------------------------------------------------

def pair_differences(features, labels, parseable):
    """Stack (correct - incorrect) feature differences within each question.

    Only candidates with a parsed final number take part: the selector never
    returns an unparsed candidate, so such pairs carry no decision-relevant
    information. Questions whose parsed candidates are all correct or all
    incorrect contribute nothing.
    """
    differences = []
    for question in range(features.shape[0]):
        positive = labels[question] & (parseable[question] > 0)
        negative = ~labels[question] & (parseable[question] > 0)
        if not positive.any() or not negative.any():
            continue
        delta = features[question, positive][:, None, :] - features[question, negative][None, :, :]
        differences.append(delta.reshape(-1, features.shape[-1]))
    return np.concatenate(differences, axis=0)


def fit_pairwise_logistic(differences, lambda_rel, iterations=25):
    """Maximise sum log sigmoid(d.w) - (lambda_rel * n_pairs / 2) |w|^2 by Newton's method."""
    n_pairs, dimension = differences.shape
    penalty = lambda_rel * n_pairs
    weight = np.zeros(dimension)
    for _ in range(iterations):
        margin = differences @ weight
        probability = 1.0 / (1.0 + np.exp(-margin))
        gradient = differences.T @ (1.0 - probability) - penalty * weight
        curvature = probability * (1.0 - probability)
        hessian = (differences * curvature[:, None]).T @ differences
        hessian.flat[::dimension + 1] += penalty
        step = np.linalg.solve(hessian, gradient)
        weight += step
        if np.abs(step).max() < 1e-7:
            break
    return weight


def fit_scaler(features):
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), 1e-6)
    return mean, scale


# ----------------------------------------------------------------------------
# Two-stage model
# ----------------------------------------------------------------------------

class Split:
    """Training or validation data with cached per-candidate features."""

    def __init__(self, name):
        self.name = name
        self.bank, self.embeddings, self.gold = load_split(name)
        self.labels = correctness(self.bank, self.gold)
        self.hard = distinct_answer_counts(self.bank) >= 2
        self.signals = text_signals(self.bank)
        self.parseable = self.signals['parseable']
        self.trace_names, self.trace = trace_features(self.bank)
        self.numeric = numeric_answer_features(self.bank)
        self.stage1 = {
            mode: stage1_features(self.bank, self.embeddings, mode, self.signals)
            for mode in STAGE1_MODES
        }

    def stage2(self, scores, blocks, index=None):
        if index is None:
            index = np.arange(len(self.bank))
        bank = [self.bank[i] for i in index]
        signals = {name: value[index] for name, value in self.signals.items()}
        return stage2_features(
            bank, scores, blocks, signals, self.trace[index], self.numeric[index]
        )

    def size(self):
        return len(self.bank)


_STAGE1_CACHE = {}


def fit_stage1(split, index, mode, lambda_rel):
    """Fit one stage-1 ranker. Identical fits are cached: the nested
    cross-validation and the ablations reuse the same stage-1 models."""
    key = (split.name, mode, lambda_rel, hashlib.sha1(np.ascontiguousarray(index).tobytes()).hexdigest())
    if key in _STAGE1_CACHE:
        return _STAGE1_CACHE[key]
    features = split.stage1[mode][index]
    mean, scale = fit_scaler(features)
    scaled = apply_scaler(features, mean, scale)
    differences = pair_differences(scaled, split.labels[index], split.parseable[index])
    weight = fit_pairwise_logistic(differences, lambda_rel)
    model = {'mean': mean, 'scale': scale, 'weight': weight, 'n_pairs': int(len(differences))}
    _STAGE1_CACHE[key] = model
    return model


def stage1_apply(model, features):
    return apply_scaler(features, model['mean'], model['scale']) @ model['weight']


def out_of_fold_stage1(split, index, mode, lambda_rel, seed=SEED):
    """Stage-1 scores for the questions in `index`, each produced by a model
    that did not see that question."""
    scores = np.zeros((len(index), 4))
    for fold in folds(len(index), seed=seed):
        held = index[fold]
        model = fit_stage1(split, index[complement(len(index), fold)], mode, lambda_rel)
        scores[fold] = stage1_apply(model, split.stage1[mode][held])
    return scores


def fit_two_stage(split, index, config):
    """Fit stage 1 (each embedding mode), out-of-fold stage-1 scores, then stage 2."""
    model = {'config': config}
    oof_scores = []
    for mode in config['stage1_modes']:
        lambda_rel = config['stage1_lambda'][mode]
        fitted = fit_stage1(split, index, mode, lambda_rel)
        model[f'{mode}_mean'] = fitted['mean']
        model[f'{mode}_scale'] = fitted['scale']
        model[f'{mode}_weight'] = fitted['weight']
        oof_scores.append(out_of_fold_stage1(split, index, mode, lambda_rel))
    if config.get('stage2', True):
        features = split.stage2(oof_scores, tuple(config['stage2_blocks']), index)
        mean, scale = fit_scaler(features)
        scaled = apply_scaler(features, mean, scale)
        differences = pair_differences(scaled, split.labels[index], split.parseable[index])
        model['stage2_mean'] = mean
        model['stage2_scale'] = scale
        model['stage2_weight'] = fit_pairwise_logistic(differences, config['stage2_lambda'])
        model['stage2_n_pairs'] = int(len(differences))
    return model


def two_stage_scores(model, split, index):
    config = model['config']
    scores = [
        stage1_apply(
            {'mean': model[f'{mode}_mean'], 'scale': model[f'{mode}_scale'], 'weight': model[f'{mode}_weight']},
            split.stage1[mode][index],
        )
        for mode in config['stage1_modes']
    ]
    if not config.get('stage2', True):
        return scores[0]
    features = split.stage2(scores, tuple(config['stage2_blocks']), index)
    return apply_scaler(features, model['stage2_mean'], model['stage2_scale']) @ model['stage2_weight']


def cross_validate(split, config, seed=SEED):
    """Nested five-fold cross-validation of the complete pipeline on one split."""
    scores = np.zeros((split.size(), 4))
    for fold in folds(split.size(), seed=seed):
        model = fit_two_stage(split, complement(split.size(), fold), config)
        scores[fold] = two_stage_scores(model, split, fold)
    return scores


def hits_from_scores(split, scores, index=None):
    if index is None:
        index = np.arange(split.size())
    choices = choose_indices(scores, split.parseable[index])
    return split.labels[index, choices]


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------

def rule_choices(split, signal_name):
    signals = split.signals
    if signal_name == 'majority':
        return majority_choices(split.bank)
    if signal_name == 'longest':
        return choose_indices(signals['log_len'], signals['parseable'])
    if signal_name == 'coverage':
        return choose_indices(signals['num_coverage'], signals['parseable'])
    if signal_name == 'first':
        return choose_indices(np.zeros((split.size(), 4)), signals['parseable'])
    raise ValueError(signal_name)


def save_bundle(directory, model):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in model.items()
        if isinstance(value, np.ndarray)
    }
    np.savez(directory / 'model.npz', **arrays)
    (directory / 'config.json').write_text(json.dumps(model['config'], indent=2) + '\n')


def asset_bytes(directory):
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parsed_answers(row):
    return [parse_answer(candidate['text']) for candidate in row['candidates']]


def disagreement_stats(split, model_choices, baseline_choices):
    model_hits = selection_hits(split.bank, split.gold, model_choices)
    baseline_hits = selection_hits(split.bank, split.gold, baseline_choices)
    differ = np.zeros(split.size(), dtype=bool)
    for i, row in enumerate(split.bank):
        answers = parsed_answers(row)
        differ[i] = answers[model_choices[i]] != answers[baseline_choices[i]]
    focus = differ & split.hard
    return {
        'hard_questions_with_different_answers': int(focus.sum()),
        'model_only_correct': int((model_hits & ~baseline_hits & focus).sum()),
        'baseline_only_correct': int((~model_hits & baseline_hits & focus).sum()),
        'both_correct': int((model_hits & baseline_hits & focus).sum()),
        'neither_correct': int((~model_hits & ~baseline_hits & focus).sum()),
    }


def margin_bins(scores, hits, hard):
    order = np.sort(scores, axis=1)
    margin = (order[:, -1] - order[:, -2])[hard]
    selected_hits = hits[hard]
    edges = np.quantile(margin, [0, 0.25, 0.5, 0.75, 1])
    bins = []
    for low, high, last in zip(edges[:-1], edges[1:], (False, False, False, True)):
        mask = (margin >= low) & ((margin <= high) if last else (margin < high))
        total = int(mask.sum())
        correct = int(selected_hits[mask].sum()) if total else 0
        bins.append({
            'margin_low': float(low),
            'margin_high': float(high),
            'correct': correct,
            'n': total,
            'accuracy': 100.0 * correct / total if total else 0.0,
        })
    return bins


def pattern_breakdown(split, choices):
    """Accuracy by the multiset of parsed-answer counts, hard questions only."""
    table = {}
    majority = majority_choices(split.bank)
    for i in np.flatnonzero(split.hard):
        answers = parsed_answers(split.bank[i])
        counts = {}
        for answer in answers:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        pattern = '-'.join(str(value) for value in sorted(counts.values(), reverse=True))
        entry = table.setdefault(pattern, {'n': 0, 'model_correct': 0, 'majority_correct': 0})
        entry['n'] += 1
        entry['model_correct'] += int(split.labels[i, choices[i]])
        entry['majority_correct'] += int(split.labels[i, majority[i]])
    return dict(sorted(table.items(), key=lambda item: -item[1]['n']))


def case_record(split, index, model_choice, majority_choice):
    row = split.bank[index]
    answers = parsed_answers(row)
    return {
        'id': row['id'],
        'question': row['question'],
        'reference_answer': parse_answer(split.gold[index]['answer']),
        'model_choice': model_choice,
        'majority_choice': majority_choice,
        'candidates': [
            {
                'index': candidate['index'],
                'parsed_answer': answers[candidate['index']],
                'ending': candidate['text'][-240:],
            }
            for candidate in row['candidates']
        ],
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    started = time.time()
    np.random.seed(SEED)
    train = Split('train')
    validation = Split('validation')
    n_train = train.size()
    all_train = np.arange(n_train)
    records = {
        'seed': SEED,
        'folds': FOLDS,
        'hard_train_questions': int(train.hard.sum()),
        'train_questions': n_train,
        'hard_validation_questions': int(validation.hard.sum()),
        'validation_questions': validation.size(),
        'selection_data': 'five-fold cross-validation on the training split; validation is a held-out check',
        'test_used_for_selection': False,
    }

    # Training-free rules on validation.
    records['rules_validation'] = []
    for rule in ('majority', 'longest', 'coverage', 'first'):
        hits = selection_hits(validation.bank, validation.gold, rule_choices(validation, rule))
        records['rules_validation'].append({
            'method': rule, 'hard': summarize(hits, validation.hard), 'all': summarize(hits, None),
        })

    # Stage 1: ridge strength per embedding mode, chosen by training cross-validation.
    records['stage1_grid'] = []
    stage1_lambda = {}
    for mode in STAGE1_MODES:
        best = None
        for lambda_rel in STAGE1_LAMBDAS:
            cv = [
                summarize(hits_from_scores(train, out_of_fold_stage1(train, all_train, mode, lambda_rel, seed)), train.hard)
                for seed in CV_SEEDS
            ]
            model = fit_stage1(train, all_train, mode, lambda_rel)
            val_hits = hits_from_scores(validation, stage1_apply(model, validation.stage1[mode]))
            row = {
                'mode': mode,
                'lambda_rel': lambda_rel,
                'n_pairs': model['n_pairs'],
                'train_cv_hard_by_seed': cv,
                'train_cv_hard_total': sum(entry['correct'] for entry in cv),
                'validation_hard': summarize(val_hits, validation.hard),
                'validation_all': summarize(val_hits, None),
            }
            records['stage1_grid'].append(row)
            print(json.dumps(row), flush=True)
            # Ties go to the stronger ridge.
            if best is None or row['train_cv_hard_total'] >= best['train_cv_hard_total']:
                best = row
        stage1_lambda[mode] = best['lambda_rel']
    records['stage1_lambda'] = stage1_lambda

    # Stage 2: ridge strength chosen by nested cross-validation on training data.
    records['stage2_grid'] = []
    best = None
    for lambda_rel in STAGE2_LAMBDAS:
        config = {
            'stage1_modes': list(STAGE1_MODES),
            'stage1_lambda': stage1_lambda,
            'stage2': True,
            'stage2_blocks': list(STAGE2_BLOCKS),
            'stage2_lambda': lambda_rel,
        }
        cv_scores = [cross_validate(train, config, seed) for seed in CV_SEEDS]
        cv = [summarize(hits_from_scores(train, scores), train.hard) for scores in cv_scores]
        model = fit_two_stage(train, all_train, config)
        val_hits = hits_from_scores(validation, two_stage_scores(model, validation, np.arange(validation.size())))
        row = {
            'stage2_lambda': lambda_rel,
            'train_cv_hard_by_seed': cv,
            'train_cv_hard_total': sum(entry['correct'] for entry in cv),
            'validation_hard': summarize(val_hits, validation.hard),
            'validation_all': summarize(val_hits, None),
        }
        records['stage2_grid'].append(row)
        print(json.dumps(row), flush=True)
        # Ties go to the stronger ridge.
        if best is None or row['train_cv_hard_total'] >= best[0]['train_cv_hard_total']:
            best = (row, config, cv_scores[0])
    winner_row, config, cv_scores = best

    # Final model and ablations.
    final = fit_two_stage(train, all_train, config)
    final['config'] = {
        'seed': SEED,
        'method': 'two-stage pairwise logistic ranker',
        **config,
        'stage2_dimensions': int(final['stage2_weight'].shape[0]),
        'stage2_n_pairs': final['stage2_n_pairs'],
        'trace_feature_names': train.trace_names,
        'selection_rule': 'ridge strengths chosen by five-fold cross-validation on hard training questions; validation used only as a held-out check',
        'train_cv_hard_by_seed': winner_row['train_cv_hard_by_seed'],
        'validation_hard': winner_row['validation_hard'],
        'validation_all': winner_row['validation_all'],
    }
    final.pop('stage2_n_pairs')
    bundle = CHECKPOINT / 'model_bundle'
    save_bundle(bundle, final)

    ablations = {
        'stage1_centered': {**config, 'stage1_modes': ['centered'], 'stage2': False},
        'stage1_absolute': {**config, 'stage1_modes': ['absolute'], 'stage2': False},
        'stage2_centered_only': {**config, 'stage1_modes': ['centered']},
        'stage2_no_trace': {**config, 'stage2_blocks': ['numeric']},
        'stage2_no_numeric': {**config, 'stage2_blocks': ['trace']},
        'stage2_scores_only': {**config, 'stage2_blocks': []},
    }
    records['ablations'] = {}
    for name, ablation_config in ablations.items():
        cv = [
            summarize(hits_from_scores(train, cross_validate(train, ablation_config, seed)), train.hard)
            for seed in CV_SEEDS
        ]
        model = fit_two_stage(train, all_train, ablation_config)
        model['config'] = {'seed': SEED, 'ablation': name, **ablation_config}
        model.pop('stage2_n_pairs', None)
        save_bundle(CHECKPOINT / 'ablations' / name, model)
        val_hits = hits_from_scores(validation, two_stage_scores(model, validation, np.arange(validation.size())))
        records['ablations'][name] = {
            'train_cv_hard_by_seed': cv,
            'train_cv_hard_total': sum(entry['correct'] for entry in cv),
            'validation_hard': summarize(val_hits, validation.hard),
            'validation_all': summarize(val_hits, None),
        }
        print(name, json.dumps(records['ablations'][name]), flush=True)

    # Held-out validation analysis with the saved checkpoint, through the official code path.
    saved = load_checkpoint(bundle)
    model_choices = choose(validation.bank, validation.embeddings, saved)
    model_hits = selection_hits(validation.bank, validation.gold, model_choices)
    majority = majority_choices(validation.bank)
    val_scores = two_stage_scores(final, validation, np.arange(validation.size()))
    records['final'] = {
        'config': final['config'],
        'validation_hard': summarize(model_hits, validation.hard),
        'validation_all': summarize(model_hits, None),
        'train_cv_hard_by_seed': winner_row['train_cv_hard_by_seed'],
        'train_cv_hard_total': winner_row['train_cv_hard_total'],
        'train_cv_pattern_breakdown': pattern_breakdown(train, choose_indices(cv_scores, train.parseable)),
        'validation_pattern_breakdown': pattern_breakdown(validation, model_choices),
        'disagreement_with_majority_on_hard_validation': disagreement_stats(validation, model_choices, majority),
        'hard_validation_margin_bins': margin_bins(val_scores, model_hits, validation.hard),
    }
    fixes, breaks = [], []
    for index in np.flatnonzero(validation.hard):
        majority_ok = bool(validation.labels[index, majority[index]])
        model_ok = bool(model_hits[index])
        if model_ok and not majority_ok and len(fixes) < 3:
            fixes.append(case_record(validation, int(index), model_choices[index], majority[index]))
        elif majority_ok and not model_ok and len(breaks) < 2:
            breaks.append(case_record(validation, int(index), model_choices[index], majority[index]))
    records['validation_cases'] = {'model_fixes_majority': fixes, 'model_breaks_majority': breaks}

    # Stage-2 weights on standardized features, for interpretation.
    names = [f'stage1_{mode}' for mode in config['stage1_modes']] + ['agree']
    names += train.trace_names + [f'{name}_centered' for name in train.trace_names]
    names += [f'numeric_{k}' for k in range(train.numeric.shape[-1])]
    weight = final['stage2_weight']
    records['stage2_weights'] = {
        name: float(value) for name, value in sorted(zip(names, weight), key=lambda item: -abs(item[1]))
    }
    records['asset_bytes'] = asset_bytes(bundle)
    records['asset_sha256'] = {
        'model.npz': file_sha256(bundle / 'model.npz'),
        'config.json': file_sha256(bundle / 'config.json'),
    }
    records['training_seconds'] = time.time() - started

    RECORDS.mkdir(exist_ok=True)
    (RECORDS / 'validation_experiments.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({
        'stage1_lambda': stage1_lambda,
        'stage2_lambda': config['stage2_lambda'],
        'train_cv_hard_by_seed': winner_row['train_cv_hard_by_seed'],
        'validation_hard': records['final']['validation_hard'],
        'validation_all': records['final']['validation_all'],
        'asset_bytes': records['asset_bytes'],
        'training_seconds': round(records['training_seconds'], 1),
    }, indent=2))


if __name__ == '__main__':
    main()
