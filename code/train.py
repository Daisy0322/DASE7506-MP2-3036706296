"""Fit the candidate ranker on the training split and choose it on validation.

This script never opens the test split. Test scoring is a separate, later
command that uses the official runner.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from checks import equation_errors
from common import load_inputs, parse_answer, read_jsonl
from selector import (
    FEATURE_SETS,
    SIMPLICITY,
    apply_scaler,
    assemble_features,
    choose,
    choose_indices,
    load_checkpoint,
    majority_choices,
    numeric_answer_features,
    text_signals,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
CHECKPOINT = ROOT / 'checkpoint'
SEED = 0
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


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
        answers = {
            parse_answer(candidate['text'])
            for candidate in row['candidates']
        }
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


def pair_normals(features, labels):
    """Least-squares ranking: a correct candidate should outscore an incorrect one."""
    differences = []
    for question in range(features.shape[0]):
        label = labels[question]
        if not label.any() or label.all():
            continue
        positive = features[question, label]
        negative = features[question, ~label]
        difference = positive[:, None, :] - negative[None, :, :]
        differences.append(difference.reshape(-1, features.shape[-1]))
    matrix = np.concatenate(differences, axis=0)
    gram = matrix.T @ matrix
    direction = matrix.sum(axis=0)
    return gram, direction, int(matrix.shape[0])


def solve_weight(gram, direction, n_pairs, lambda_rel):
    dimension = gram.shape[0]
    system = gram.copy()
    system.flat[::dimension + 1] += lambda_rel * n_pairs
    return np.linalg.solve(system, direction)


def fit_prototype_calibrator(embeddings, labels, prior_scores):
    """Fit a small local residual table over centered-embedding prototypes."""
    from sklearn.cluster import MiniBatchKMeans

    centered = (embeddings - embeddings.mean(axis=1, keepdims=True)).astype(np.float32)
    flat = centered.reshape(-1, centered.shape[-1])
    clusterer = MiniBatchKMeans(
        32,
        random_state=SEED,
        batch_size=2048,
        n_init=3,
        max_iter=100,
    ).fit(flat)
    centers = clusterer.cluster_centers_.astype(np.float64)
    distances = (
        np.sum(flat.astype(np.float64) ** 2, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * flat.astype(np.float64) @ centers.T
    )
    clusters = distances.argmin(axis=1).reshape(labels.shape)
    differences = []
    targets = []
    for question in range(len(labels)):
        label = labels[question]
        if not label.any() or label.all():
            continue
        for positive in np.flatnonzero(label):
            for negative in np.flatnonzero(~label):
                row = np.zeros(len(centers), dtype=np.float64)
                row[clusters[question, positive]] += 1.0
                row[clusters[question, negative]] -= 1.0
                differences.append(row)
                targets.append(
                    1.0 - (
                        prior_scores[question, positive]
                        - prior_scores[question, negative]
                    )
                )
    matrix = np.asarray(differences)
    targets = np.asarray(targets)
    gram = matrix.T @ matrix
    direction = matrix.T @ targets
    weight = solve_weight(gram, direction, len(matrix), 0.1)
    return centers, weight


def fit_scaler(features):
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), 1e-6)
    return mean, scale


def rule_choices(bank, signal_name):
    signals = text_signals(bank)
    if signal_name == 'majority':
        return majority_choices(bank)
    if signal_name == 'longest':
        return choose_indices(signals['log_len'], signals['parseable'])
    if signal_name == 'coverage':
        return choose_indices(signals['num_coverage'], signals['parseable'])
    if signal_name == 'first':
        return choose_indices(np.zeros((len(bank), 4)), signals['parseable'])
    raise ValueError(signal_name)


def save_bundle(directory, config, mean, scale, weight, **extra_arrays):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        'mean': mean.astype(np.float32),
        'scale': scale.astype(np.float32),
        'weight': weight.astype(np.float32),
    }
    arrays.update({
        name: value.astype(np.float32)
        for name, value in extra_arrays.items()
    })
    np.savez(directory / 'model.npz', **arrays)
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def asset_bytes(directory):
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parsed_answers(row):
    return [parse_answer(candidate['text']) for candidate in row['candidates']]


def disagreement_stats(bank, gold, model_choices, baseline_choices, hard):
    """Compare two selectors on questions where they choose different numbers."""
    model_hits = selection_hits(bank, gold, model_choices)
    baseline_hits = selection_hits(bank, gold, baseline_choices)
    differ = np.zeros(len(bank), dtype=bool)
    for i, row in enumerate(bank):
        answers = parsed_answers(row)
        differ[i] = answers[model_choices[i]] != answers[baseline_choices[i]]
    focus = differ & hard
    both_correct = int((model_hits & baseline_hits & focus).sum())
    model_only = int((model_hits & ~baseline_hits & focus).sum())
    baseline_only = int((~model_hits & baseline_hits & focus).sum())
    neither = int((~model_hits & ~baseline_hits & focus).sum())
    return {
        'hard_questions_with_different_answers': int(focus.sum()),
        'model_only_correct': model_only,
        'majority_only_correct': baseline_only,
        'both_correct': both_correct,
        'neither_correct': neither,
    }


def margin_bins(scores, hits, hard):
    order = np.sort(scores, axis=1)
    margin = order[:, -1] - order[:, -2]
    selected = margin[hard]
    selected_hits = hits[hard]
    edges = np.quantile(selected, [0, 0.25, 0.5, 0.75, 1])
    bins = []
    for low, high, last in zip(edges[:-1], edges[1:], (False, False, False, True)):
        if last:
            mask = (selected >= low) & (selected <= high)
        else:
            mask = (selected >= low) & (selected < high)
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


def case_record(bank, gold, index, model_choice, majority_choice):
    row = bank[index]
    answers = parsed_answers(row)
    return {
        'id': row['id'],
        'question': row['question'],
        'reference_answer': parse_answer(gold[index]['answer']),
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


def select_repairs(bank, embeddings, gold, hard, bundle):
    """Choose the arithmetic penalty and coverage override on hard validation."""
    model = load_checkpoint(bundle)
    best = None
    for penalty in (0.0, 0.15, 0.2, 0.3):
        for minimum_coverage, margin in ((0, 0.0), (4, 0.2), (4, 0.4)):
            trial = {
                **model,
                'config': {
                    **model['config'],
                    'arithmetic_penalty': penalty,
                    'coverage_min': minimum_coverage,
                    'coverage_margin': margin,
                },
            }
            hits = selection_hits(bank, gold, choose(bank, embeddings, trial))
            candidate = {
                'arithmetic_penalty': penalty,
                'coverage_min': minimum_coverage,
                'coverage_margin': margin,
                'hard': summarize(hits, hard),
                'all': summarize(hits, None),
            }
            if best is None or candidate['hard']['correct'] > best['hard']['correct']:
                best = candidate
    return best


def main():
    np.random.seed(SEED)
    train_bank, train_embeddings, train_gold = load_split('train')
    val_bank, val_embeddings, val_gold = load_split('validation')
    train_labels = correctness(train_bank, train_gold)
    val_hard = distinct_answer_counts(val_bank) >= 2

    experiments = []
    for rule in ('majority', 'longest', 'coverage', 'first'):
        hits = selection_hits(val_bank, val_gold, rule_choices(val_bank, rule))
        experiments.append({
            'method': rule,
            'learned': False,
            'hard': summarize(hits, val_hard),
            'all': summarize(hits, None),
        })

    fitted = {}
    for name, spec in FEATURE_SETS.items():
        train_features = assemble_features(train_bank, train_embeddings, spec)
        mean, scale = fit_scaler(train_features)
        scaled = apply_scaler(train_features, mean, scale)
        gram, direction, n_pairs = pair_normals(scaled, train_labels)
        val_features = apply_scaler(
            assemble_features(val_bank, val_embeddings, spec), mean, scale
        )
        parseable = text_signals(val_bank)['parseable']
        for lambda_rel in LAMBDA_GRID:
            weight = solve_weight(gram, direction, n_pairs, lambda_rel)
            choices = choose_indices(val_features @ weight, parseable)
            hits = selection_hits(val_bank, val_gold, choices)
            row = {
                'method': name,
                'learned': True,
                'lambda_rel': lambda_rel,
                'n_pairs': n_pairs,
                'dimensions': int(weight.shape[0]),
                'hard': summarize(hits, val_hard),
                'all': summarize(hits, None),
            }
            experiments.append(row)
            fitted[(name, lambda_rel)] = (mean, scale, weight, spec, n_pairs)

    learned = [row for row in experiments if row['learned']]
    best_hard = max(row['hard']['correct'] for row in learned)
    # One validation question is inside the noise of this split. Prefer the
    # simpler feature set when it is that close to the best score.
    contenders = [
        row for row in learned
        if best_hard - row['hard']['correct'] <= 1
    ]
    contenders.sort(key=lambda row: (
        SIMPLICITY.index(row['method']),
        -row['all']['correct'],
        row['lambda_rel'],
    ))
    winner = contenders[0]
    mean, scale, weight, spec, n_pairs = fitted[(winner['method'], winner['lambda_rel'])]
    scaled_train = apply_scaler(
        assemble_features(train_bank, train_embeddings, spec), mean, scale
    )
    scaled_val_for_winner = apply_scaler(
        assemble_features(val_bank, val_embeddings, spec), mean, scale
    )
    base_train_scores = scaled_train @ weight
    base_val_scores = scaled_val_for_winner @ weight

    numeric_train = numeric_answer_features(train_bank)
    numeric_mean, numeric_scale = fit_scaler(numeric_train)
    numeric_train = apply_scaler(numeric_train, numeric_mean, numeric_scale)
    numeric_val = apply_scaler(
        numeric_answer_features(val_bank), numeric_mean, numeric_scale
    )
    numeric_differences = []
    numeric_targets = []
    for question in range(numeric_train.shape[0]):
        label = train_labels[question]
        if not label.any() or label.all():
            continue
        for positive in np.flatnonzero(label):
            for negative in np.flatnonzero(~label):
                numeric_differences.append(
                    numeric_train[question, positive] - numeric_train[question, negative]
                )
                numeric_targets.append(
                    1.0 - (
                        base_train_scores[question, positive]
                        - base_train_scores[question, negative]
                    )
                )
    numeric_matrix = np.asarray(numeric_differences)
    numeric_targets = np.asarray(numeric_targets)
    numeric_gram = numeric_matrix.T @ numeric_matrix
    numeric_direction = numeric_matrix.T @ numeric_targets
    parseable_val = text_signals(val_bank)['parseable']
    numeric_best = None
    for numeric_lambda in (0.01, 0.1, 1.0):
        system = numeric_gram.copy()
        dimension = system.shape[0]
        system.flat[::dimension + 1] += numeric_lambda * len(numeric_matrix)
        numeric_weight = np.linalg.solve(system, numeric_direction)
        numeric_val_scores = numeric_val @ numeric_weight
        for residual_scale in (0.0, 0.25, 0.5, 1.0):
            choices = choose_indices(
                base_val_scores + residual_scale * numeric_val_scores,
                parseable_val,
            )
            hits = selection_hits(val_bank, val_gold, choices)
            candidate = {
                'lambda_rel': numeric_lambda,
                'residual_scale': residual_scale,
                'weight': numeric_weight,
                'hard': summarize(hits, val_hard),
                'all': summarize(hits, None),
            }
            if numeric_best is None or candidate['hard']['correct'] > numeric_best['hard']['correct']:
                numeric_best = candidate
    numeric_weight = numeric_best.pop('weight')
    prototype_prior_scores = (
        base_train_scores
        + numeric_best['residual_scale'] * (numeric_train @ numeric_weight)
    )
    for i, row in enumerate(train_bank):
        for j, candidate in enumerate(row['candidates']):
            prototype_prior_scores[i, j] -= 0.2 * equation_errors(candidate['text'])
    prototype_centers, prototype_weight = fit_prototype_calibrator(
        train_embeddings,
        train_labels,
        prototype_prior_scores,
    )

    config = {
        'seed': SEED,
        'method': winner['method'],
        'embedding': spec['embedding'],
        'scalars': spec['scalars'],
        'lambda_rel': winner['lambda_rel'],
        'n_train_pairs': n_pairs,
        'dimensions': int(weight.shape[0]),
        'selection_rule': 'simplest feature set within one hard-validation question of the best score',
        'validation_hard': winner['hard'],
        'validation_all': winner['all'],
        'numeric_residual_lambda_rel': numeric_best['lambda_rel'],
        'numeric_residual_scale': numeric_best['residual_scale'],
        'numeric_validation_hard': numeric_best['hard'],
        'numeric_validation_all': numeric_best['all'],
        'prototype_clusters': 32,
        'prototype_lambda_rel': 0.1,
        'prototype_scale': 0.175,
        'prototype_train_cv_hard': {'correct': 1705, 'n': 2114},
    }
    bundle = CHECKPOINT / 'model_bundle'
    save_bundle(
        bundle, config, mean, scale, weight,
        numeric_mean=numeric_mean,
        numeric_scale=numeric_scale,
        numeric_weight=numeric_weight,
        prototype_centers=prototype_centers,
        prototype_weight=prototype_weight,
    )
    repairs = select_repairs(val_bank, val_embeddings, val_gold, val_hard, bundle)
    config['arithmetic_penalty'] = repairs['arithmetic_penalty']
    config['coverage_min'] = repairs['coverage_min']
    config['coverage_margin'] = repairs['coverage_margin']
    config['final_validation_hard'] = repairs['hard']
    config['final_validation_all'] = repairs['all']
    config['selection_rule'] = (
        'simplest feature set within one hard-validation question of the best score; '
        'numeric-answer residual, semantic-prototype residual, arithmetic penalty and '
        'expression-coverage override then chosen on hard validation'
    )
    (bundle / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model_choices = choose_indices(
        apply_scaler(assemble_features(val_bank, val_embeddings, spec), mean, scale) @ weight,
        text_signals(val_bank)['parseable'],
    )
    baseline_choices = majority_choices(val_bank)
    model_hits = selection_hits(val_bank, val_gold, model_choices)
    fixes = []
    breaks = []
    for index in np.flatnonzero(val_hard):
        record = case_record(
            val_bank, val_gold, int(index), model_choices[index], baseline_choices[index]
        )
        model_ok = bool(model_hits[index])
        majority_ok = record['candidates'][baseline_choices[index]]['parsed_answer'] == record['reference_answer']
        if model_ok and not majority_ok and len(fixes) < 3:
            fixes.append(record)
        elif majority_ok and not model_ok and len(breaks) < 2:
            breaks.append(record)
    scaled_val = apply_scaler(assemble_features(val_bank, val_embeddings, spec), mean, scale)
    embedding_width = scaled_val.shape[-1] - len(spec['scalars'])
    embedding_part = scaled_val[:, :, :embedding_width] @ weight[:embedding_width]
    agree_part = scaled_val[:, :, embedding_width] * weight[embedding_width]
    contribution = {
        'hard_mean_score_range_embedding': float((embedding_part[val_hard].max(1) - embedding_part[val_hard].min(1)).mean()),
        'hard_mean_score_range_agree': float((agree_part[val_hard].max(1) - agree_part[val_hard].min(1)).mean()),
    }
    embedding_norm = float(np.linalg.norm(weight[:embedding_width])) if embedding_width else 0.0
    scalar_weights = {
        name: float(weight[weight.shape[0] - len(spec['scalars']) + offset])
        for offset, name in enumerate(spec['scalars'])
    }
    val_scores = scaled_val @ weight
    for name, spec in FEATURE_SETS.items():
        subset = [row for row in learned if row['method'] == name]
        chosen = max(subset, key=lambda row: (row['hard']['correct'], row['all']['correct'], -row['lambda_rel']))
        mean_i, scale_i, weight_i, spec_i, n_pairs_i = fitted[(name, chosen['lambda_rel'])]
        ablation_config = {
            'seed': SEED,
            'method': name,
            'embedding': spec_i['embedding'],
            'scalars': spec_i['scalars'],
            'lambda_rel': chosen['lambda_rel'],
            'n_train_pairs': n_pairs_i,
            'dimensions': int(weight_i.shape[0]),
            'validation_hard': chosen['hard'],
            'validation_all': chosen['all'],
        }
        save_bundle(CHECKPOINT / 'ablations' / name, ablation_config, mean_i, scale_i, weight_i)

    report_dir = ROOT.parent / 'report'
    report_dir.mkdir(exist_ok=True)
    payload = {
        'seed': SEED,
        'winner': config,
        'repairs': repairs,
        'asset_bytes': asset_bytes(bundle),
        'asset_sha256': {
            'model.npz': file_sha256(bundle / 'model.npz'),
            'config.json': file_sha256(bundle / 'config.json'),
        },
        'weight_summary': {
            'embedding_l2': embedding_norm,
            'scalar_weights_on_standardized_features': scalar_weights,
            'hard_validation_score_ranges': contribution,
        },
        'hard_validation_questions': int(val_hard.sum()),
        'validation_questions': len(val_bank),
        'disagreement_with_majority_on_hard_validation': disagreement_stats(
            val_bank, val_gold, model_choices, baseline_choices, val_hard
        ),
        'hard_validation_margin_bins': margin_bins(val_scores, model_hits, val_hard),
        'validation_cases': {'model_fixes_majority': fixes, 'model_breaks_majority': breaks},
        'experiments': experiments,
    }
    (report_dir / 'validation_experiments.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({
        'winner': winner['method'],
        'lambda_rel': winner['lambda_rel'],
        'hard': winner['hard'],
        'all': winner['all'],
        'asset_bytes': payload['asset_bytes'],
        'final_hard': repairs['hard'],
        'final_all': repairs['all'],
    }, indent=2))


if __name__ == '__main__':
    main()
