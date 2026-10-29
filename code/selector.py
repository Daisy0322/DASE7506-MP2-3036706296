"""Select one of four candidate solutions.

The method does not read reference answers. It compares the four fixed
embeddings and a few quantities computed from the candidate texts.
"""

from pathlib import Path
from fractions import Fraction
import json
import re

import numpy as np

from checks import equation_errors, expression_coverage
from common import parse_answer

NUMBER_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')

# Pre-specified feature sets. The training script picks one on validation.
FEATURE_SETS = {
    'agree': {'embedding': None, 'scalars': ['agree', 'parseable']},
    'embed': {'embedding': 'centered', 'scalars': []},
    'embed_agree': {'embedding': 'centered', 'scalars': ['agree', 'parseable']},
    'absolute_agree': {'embedding': 'absolute', 'scalars': ['agree', 'parseable']},
    'embed_agree_cov': {'embedding': 'centered', 'scalars': ['agree', 'parseable', 'num_coverage']},
    'embed_agree_len': {'embedding': 'centered', 'scalars': ['agree', 'parseable', 'log_len']},
}

# Simpler sets win a tie that is only one validation question wide.
SIMPLICITY = ['agree', 'embed', 'embed_agree', 'absolute_agree', 'embed_agree_cov', 'embed_agree_len']


def _number_set(text):
    found = set()
    for match in NUMBER_RE.findall(text):
        token = match.replace(',', '')
        if token.endswith('.0'):
            token = token[:-2]
        found.add(token)
    return found


def text_signals(bank):
    """Per-candidate signals that do not require a learned model.

    Returns float arrays of shape [N, 4]: agree, parseable, log_len, num_coverage.
    `agree` is the fraction of the four candidates that parse to the same number.
    Unparsed candidates score 0, because the official scorer cannot mark them correct.
    """
    agree = np.zeros((len(bank), 4), dtype=np.float64)
    parseable = np.zeros((len(bank), 4), dtype=np.float64)
    log_len = np.zeros((len(bank), 4), dtype=np.float64)
    coverage = np.zeros((len(bank), 4), dtype=np.float64)
    for i, row in enumerate(bank):
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts = {}
        for answer in answers:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        question_numbers = _number_set(row['question'])
        for j, candidate in enumerate(row['candidates']):
            answer = answers[j]
            parseable[i, j] = float(answer is not None)
            if answer is not None:
                agree[i, j] = counts[answer] / 4.0
            log_len[i, j] = np.log1p(len(candidate['text']))
            if question_numbers:
                overlap = len(question_numbers & _number_set(candidate['text']))
                coverage[i, j] = overlap / len(question_numbers)
    return {
        'agree': agree,
        'parseable': parseable,
        'log_len': log_len,
        'num_coverage': coverage,
    }


def numeric_answer_features(bank):
    """General plausibility features for each candidate's parsed answer."""
    features = np.zeros((len(bank), 4, 17), dtype=np.float64)
    for i, row in enumerate(bank):
        numbers = []
        for token in NUMBER_RE.findall(row['question']):
            try:
                numbers.append(abs(float(Fraction(token.replace(',', '')))))
            except (ValueError, ZeroDivisionError):
                continue
        positive = [value for value in numbers if value > 0]
        minimum = min(positive, default=1.0)
        maximum = max(positive, default=1.0)
        total = sum(positive) or 1.0
        product = float(np.prod(positive[:6])) if positive else 1.0
        answers = []
        for candidate in row['candidates']:
            parsed = parse_answer(candidate['text'])
            try:
                answers.append(Fraction(parsed) if parsed is not None else None)
            except (ValueError, ZeroDivisionError):
                answers.append(None)
        magnitudes = [abs(float(answer)) for answer in answers if answer is not None]
        for j, answer in enumerate(answers):
            if answer is None:
                continue
            value = float(answer)
            magnitude = abs(value)
            ratios = [
                np.log((magnitude + 1.0) / (reference + 1.0))
                for reference in (minimum, maximum, total, abs(product))
            ]
            relative = [
                abs(np.log((magnitude + 1.0) / (reference + 1.0)))
                for reference in positive
            ] or [0.0]
            rank = sum(other < magnitude for other in magnitudes) / max(1, len(magnitudes) - 1)
            features[i, j] = [
                np.log1p(magnitude),
                value < 0,
                value == 0,
                answer.denominator == 1,
                np.log1p(answer.denominator),
                *ratios,
                min(relative),
                rank,
                magnitude > maximum,
                magnitude < minimum,
                any(abs(magnitude - number) < 1e-9 for number in numbers),
                len(set(answers)) / 4.0,
                len(numbers) / 10.0,
                1.0,
            ]
    return features


def assemble_features(bank, embeddings, spec):
    """Build an [N, 4, D] feature array for one feature set."""
    parts = []
    if spec['embedding'] == 'centered':
        parts.append(embeddings - embeddings.mean(axis=1, keepdims=True))
    elif spec['embedding'] == 'absolute':
        parts.append(embeddings)
    elif spec['embedding'] is not None:
        raise ValueError(f"Unknown embedding mode: {spec['embedding']}")
    signals = text_signals(bank)
    for name in spec['scalars']:
        parts.append(signals[name][:, :, None])
    if not parts:
        raise ValueError('Feature set is empty')
    return np.concatenate(parts, axis=-1).astype(np.float64, copy=False)


def apply_scaler(features, mean, scale):
    return (features - mean) / scale


def scores_from_features(features, weight):
    return features @ weight


def choose_indices(scores, parseable):
    """Pick the highest score among candidates that have a parsed number."""
    adjusted = np.where(parseable > 0, scores, -1e9)
    # argmax breaks remaining ties toward the smallest index.
    selected = adjusted.argmax(axis=1)
    return [int(index) for index in selected]


def majority_choices(bank):
    """Select the most common parsed answer, breaking ties toward index 0."""
    signals = text_signals(bank)
    return choose_indices(signals['agree'], signals['parseable'])


def load_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    with np.load(checkpoint / 'model.npz', allow_pickle=False) as bundle:
        model = {
            'config': config,
            'mean': bundle['mean'].astype(np.float64),
            'scale': bundle['scale'].astype(np.float64),
            'weight': bundle['weight'].astype(np.float64),
        }
        for name in (
            'numeric_mean',
            'numeric_scale',
            'numeric_weight',
            'prototype_centers',
            'prototype_weight',
        ):
            if name in bundle:
                model[name] = bundle[name].astype(np.float64)
    return model


def choose(bank, embeddings, model):
    spec = {
        'embedding': model['config']['embedding'],
        'scalars': list(model['config']['scalars']),
    }
    features = assemble_features(bank, embeddings, spec)
    scaled = apply_scaler(features, model['mean'], model['scale'])
    scores = scores_from_features(scaled, model['weight'])
    config = model['config']
    numeric_scale = float(config.get('numeric_residual_scale', 0.0))
    if numeric_scale and 'numeric_weight' in model:
        numeric = numeric_answer_features(bank)
        numeric = apply_scaler(numeric, model['numeric_mean'], model['numeric_scale'])
        scores += numeric_scale * scores_from_features(numeric, model['numeric_weight'])
    prototype_scale = float(config.get('prototype_scale', 0.0))
    if prototype_scale and 'prototype_centers' in model:
        centered = embeddings - embeddings.mean(axis=1, keepdims=True)
        flat = centered.reshape(-1, centered.shape[-1]).astype(np.float64, copy=False)
        centers = model['prototype_centers']
        distances = (
            np.sum(flat * flat, axis=1)[:, None]
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * flat @ centers.T
        )
        clusters = distances.argmin(axis=1)
        prototype_scores = model['prototype_weight'][clusters].reshape(len(bank), 4)
        scores += prototype_scale * prototype_scores
    penalty = float(config.get('arithmetic_penalty', 0.0))
    if penalty:
        for i, row in enumerate(bank):
            for j, candidate in enumerate(row['candidates']):
                scores[i, j] -= penalty * equation_errors(candidate['text'])
    parseable = text_signals(bank)['parseable']
    choices = choose_indices(scores, parseable)
    minimum_coverage = int(config.get('coverage_min', 0))
    margin = float(config.get('coverage_margin', 0.0))
    if minimum_coverage:
        for i, row in enumerate(bank):
            answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
            distinct = {answer for answer in answers if answer is not None}
            if len(distinct) < 2:
                continue
            table = expression_coverage(row['question'])
            coverage = {}
            for answer in distinct:
                try:
                    coverage[answer] = table.get(Fraction(answer), 0)
                except (ValueError, ZeroDivisionError):
                    coverage[answer] = 0
            if not coverage:
                continue
            best = max(coverage.values())
            winners = [answer for answer, used in coverage.items() if used == best]
            if best < minimum_coverage or len(winners) != 1:
                continue
            indexes = [j for j, answer in enumerate(answers) if answer == winners[0]]
            replacement = indexes[int(np.argmax(scores[i, indexes]))]
            current = choices[i]
            if scores[i, replacement] < scores[i, current] - margin:
                continue
            choices[i] = replacement
    return choices
