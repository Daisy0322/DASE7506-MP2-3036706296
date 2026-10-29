"""Select one of four candidate solutions.

The method does not read reference answers. It scores candidates in two
stages. Stage 1 is a pairwise logistic ranker on the fixed embeddings
(centered within the question, and separately uncentered) plus final-answer
agreement. Stage 2 is a second pairwise logistic ranker that combines the
stage-1 scores with features computed from the candidate text: agreement of
intermediate numbers with the other three traces, counts of explicit false
equations, structure of the write-up, and plausibility of the parsed final
number relative to the numbers in the question.
"""

from pathlib import Path
from fractions import Fraction
import json
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this experiment's checks.py first
from checks import equation_errors  # noqa: E402
from common import parse_answer  # noqa: E402  (official parser from code/)

NUMBER_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')

# Word and symbol patterns counted in each trace (capped at five occurrences).
WORD_PATTERNS = {
    'wait': r'\bwait\b', 'but': r'\bbut\b', 'however': r'\bhowever\b', 'actually': r'\bactually\b',
    'recheck': r're-?check|double-?check|verify|let.s check|check(?:ing)?\b',
    'mistake': r'mistake|error|incorrect|wrong', 'assume': r'assum', 'approx': r'approx|round',
    'check_mark': r'✅|✔', 'cross': r'❌|✗', 'so': r'\bso\b', 'therefore': r'therefore|thus|hence',
    'final': r'final answer', 'boxed': r'\\boxed', 'hash': r'####', 'question_mark': r'\?',
    'not': r'\bnot\b', 'each': r'\beach\b', 'total': r'\btotal\b', 'remaining': r'remain|left',
    'per': r'\bper\b', 'step': r'step \d|^\s*\d+\.', 'bold': r'\*\*', 'dollar': r'\$',
    'frac': r'\\frac', 'times': r'\\times|×', 'percent': r'%|percent', 'hour': r'hour|minute',
    'half': r'half|twice|double|triple', 'dozen': r'dozen',
    'interpret': r'interpret|means that|this means|in other words',
    'alternatively': r'alternativ|another way|or\b', 'note': r'\bnote\b', 'since': r'\bsince\b',
    'because': r'\bbecause\b', 'given': r'\bgiven\b', 'we_need': r'we need|we must|we have to',
    'lets': r'let.s', 'x_var': r'\bx\b', 'equation': r'equation', 'no': r'\bno\b',
    'already': r'already', 'first': r'\bfirst\b', 'then': r'\bthen\b', 'after': r'\bafter\b',
    'before': r'\bbefore\b', 'including': r'includ', 'except': r'except|excluding|other than',
}
WORD_RE = {name: re.compile(pattern, re.I | re.M) for name, pattern in WORD_PATTERNS.items()}
BOXED_RE = re.compile(r'\\boxed\{[^{}]+\}')
WORD_TOKEN_RE = re.compile(r'\w+')
SENTENCE_END_RE = re.compile(r'[.!?]\s')


def _number_set(text):
    found = set()
    for match in NUMBER_RE.findall(text):
        token = match.replace(',', '')
        if token.endswith('.0'):
            token = token[:-2]
        found.add(token)
    return found


def _fractions(text):
    values = []
    for match in NUMBER_RE.findall(text):
        try:
            values.append(Fraction(match.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            continue
    return values


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
    """General plausibility features for each candidate's parsed answer (17 values)."""
    features = np.zeros((len(bank), 4, 17), dtype=np.float64)
    for i, row in enumerate(bank):
        numbers = [abs(float(value)) for value in _fractions(row['question'])]
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


def trace_features(bank):
    """Features computed from each candidate's text and from the other three traces.

    Returns (names, array of shape [N, 4, 73]). Values are small (most lie in
    [0, 1]); they are standardized by the training scaler before use.
    """
    names = None
    rows = []
    for row in bank:
        question = row['question']
        question_numbers = set(_fractions(question))
        question_words = set(WORD_TOKEN_RE.findall(question.lower()))
        texts = [candidate['text'] for candidate in row['candidates']]
        answers = [parse_answer(text) for text in texts]
        candidate_numbers = [_fractions(text) for text in texts]
        candidate_sets = [set(values) for values in candidate_numbers]
        lengths = np.array([len(text) for text in texts], dtype=np.float64)
        for j, text in enumerate(texts):
            feature = {}
            body = text.rsplit('####', 1)[0] if '####' in text else text
            body = BOXED_RE.sub(' ', body)
            body_numbers = _fractions(body)
            own = candidate_sets[j]
            novel = own - question_numbers
            others = set().union(*[candidate_sets[k] for k in range(4) if k != j])
            others_novel = others - question_numbers
            feature['log_len'] = np.log1p(len(text))
            feature['rel_len'] = np.log1p(len(text)) - np.log1p(lengths).mean()
            feature['len_rank'] = (lengths < len(text)).sum() / 3
            feature['n_nums'] = len(candidate_numbers[j]) / 10
            feature['n_distinct_nums'] = len(own) / 10
            feature['q_cov'] = len(own & question_numbers) / max(1, len(question_numbers))
            feature['novel_frac'] = len(novel) / max(1, len(own))
            feature['novel_shared'] = len(novel & others_novel) / max(1, len(novel))
            feature['novel_unshared_count'] = len(novel - others_novel) / 5
            jaccard = []
            for k in range(4):
                if k == j:
                    continue
                other_novel = candidate_sets[k] - question_numbers
                jaccard.append(len(novel & other_novel) / max(1, len(novel | other_novel)))
            feature['novel_jaccard_mean'] = float(np.mean(jaccard))
            feature['novel_jaccard_max'] = float(np.max(jaccard))
            errors = equation_errors(text)
            equals = text.count('=')
            feature['eq_errors'] = errors
            feature['eq_errors_any'] = float(errors > 0)
            feature['n_eq'] = equals / 10
            feature['eq_err_rate'] = errors / max(1, equals)
            answer_value = None
            if answers[j] is not None:
                try:
                    answer_value = Fraction(answers[j])
                except (ValueError, ZeroDivisionError):
                    answer_value = None
            if answer_value is not None:
                feature['ans_in_body'] = float(answer_value in set(body_numbers))
                feature['ans_in_question'] = float(answer_value in question_numbers)
                feature['ans_shared_cands'] = sum(
                    answer_value in candidate_sets[k] for k in range(4) if k != j
                ) / 3
                feature['ans_is_last_num'] = float(bool(body_numbers) and body_numbers[-1] == answer_value)
                feature['ans_count_in_text'] = min(
                    5, sum(value == answer_value for value in candidate_numbers[j])
                ) / 5
            else:
                for name in ('ans_in_body', 'ans_in_question', 'ans_shared_cands',
                             'ans_is_last_num', 'ans_count_in_text'):
                    feature[name] = 0.0
            feature['n_boxed'] = min(3, text.count('\\boxed')) / 3
            feature['n_hash'] = min(3, text.count('####')) / 3
            feature['n_lines'] = np.log1p(text.count('\n'))
            feature['n_sent'] = np.log1p(len(SENTENCE_END_RE.findall(text)))
            opening_words = set(WORD_TOKEN_RE.findall(text[:300].lower()))
            feature['has_question_restate'] = len(question_words & opening_words) / max(1, len(question_words))
            for name, pattern in WORD_RE.items():
                feature['w_' + name] = min(5, len(pattern.findall(text))) / 5
            if names is None:
                names = list(feature)
            rows.append([feature[name] for name in names])
    array = np.array(rows, dtype=np.float64).reshape(len(bank), 4, len(names))
    return names, array


def stage1_features(bank, embeddings, mode, signals=None):
    """[N, 4, 1026]: embedding (centered within question or raw) + agree + parseable."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if mode == 'centered':
        part = embeddings - embeddings.mean(axis=1, keepdims=True)
    elif mode == 'absolute':
        part = embeddings
    else:
        raise ValueError(f'Unknown embedding mode: {mode}')
    signals = signals or text_signals(bank)
    return np.concatenate(
        [part, signals['agree'][:, :, None], signals['parseable'][:, :, None]], axis=-1
    )


STAGE2_BLOCKS = ('trace', 'numeric')


def stage2_features(bank, stage1_scores, blocks=STAGE2_BLOCKS, signals=None, trace=None, numeric=None):
    """[N, 4, D]: stage-1 scores, agreement, then the requested blocks: trace
    features (raw and within-question centered) and numeric-answer features."""
    signals = signals or text_signals(bank)
    parts = [score[:, :, None] for score in stage1_scores]
    parts.append(signals['agree'][:, :, None])
    if 'trace' in blocks:
        if trace is None:
            trace = trace_features(bank)[1]
        parts.append(trace)
        parts.append(trace - trace.mean(axis=1, keepdims=True))
    if 'numeric' in blocks:
        if numeric is None:
            numeric = numeric_answer_features(bank)
        parts.append(numeric)
    return np.concatenate(parts, axis=-1)


def apply_scaler(features, mean, scale):
    return (features - mean) / scale


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
        model = {'config': config}
        for name in bundle.files:
            model[name] = bundle[name].astype(np.float64)
    return model


def stage1_scores(bank, embeddings, model, signals):
    scores = []
    for mode in model['config']['stage1_modes']:
        features = stage1_features(bank, embeddings, mode, signals)
        features = apply_scaler(features, model[f'{mode}_mean'], model[f'{mode}_scale'])
        scores.append(features @ model[f'{mode}_weight'])
    return scores


def choose(bank, embeddings, model):
    signals = text_signals(bank)
    scores = stage1_scores(bank, embeddings, model, signals)
    if model['config'].get('stage2', True):
        blocks = tuple(model['config'].get('stage2_blocks', STAGE2_BLOCKS))
        features = stage2_features(bank, scores, blocks, signals)
        features = apply_scaler(features, model['stage2_mean'], model['stage2_scale'])
        final = features @ model['stage2_weight']
    else:
        final = scores[0]
    return choose_indices(final, signals['parseable'])
