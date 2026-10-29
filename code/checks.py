"""Checks that do not need a learned model.

`equation_errors` counts arithmetic claims that are false.
`expression_coverage` asks whether a candidate's final number can be
computed from the question's own numbers.
"""

import ast
import re
from fractions import Fraction

NUMBER = r'[-+]?\d+(?:\.\d+)?'
EQUATION = re.compile(
    rf'({NUMBER}(?:\s*[+\-*/]\s*{NUMBER}){{1,8}})\s*=\s*({NUMBER})'
)
FRACTION_EQUATION = re.compile(
    rf'\\frac\s*\{{\s*({NUMBER})\s*\}}\s*\{{\s*({NUMBER})\s*\}}'
    rf'\s*=\s*\\frac\s*\{{\s*({NUMBER})\s*\}}\s*\{{\s*({NUMBER})\s*\}}'
)
FRACTION_VALUE = re.compile(
    rf'\\frac\s*\{{\s*({NUMBER})\s*\}}\s*\{{\s*({NUMBER})\s*\}}\s*=\s*({NUMBER})'
)
FRACTION = re.compile(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
QUESTION_NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')


def _evaluate(expr):
    tree = ast.parse(expr, mode='eval')

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError
            return left / right
        raise ValueError

    return walk(tree)


def _close(left, right):
    return abs(left - right) <= 1e-6 * max(1.0, abs(right))


def equation_errors(text):
    """How many explicit numeric equations in the solution are false."""
    errors = 0
    for numerator, denominator, other_num, other_den in FRACTION_EQUATION.findall(text):
        try:
            if float(denominator) == 0 or float(other_den) == 0:
                continue
            left = float(numerator) / float(denominator)
            right = float(other_num) / float(other_den)
            errors += int(not _close(left, right))
        except ValueError:
            continue
    for numerator, denominator, value in FRACTION_VALUE.findall(text):
        try:
            if float(denominator) == 0:
                continue
            errors += int(not _close(float(numerator) / float(denominator), float(value)))
        except ValueError:
            continue
    normalized = text.replace('\\times', '*').replace('\\div', '/').replace('\\cdot', '*')
    normalized = normalized.replace('×', '*').replace('÷', '/').replace('−', '-').replace('–', '-')
    normalized = FRACTION.sub(lambda match: f'({match.group(1)})/({match.group(2)})', normalized)
    normalized = re.sub(r'\\[a-zA-Z]+', ' ', normalized)
    normalized = normalized.replace('{', ' ').replace('}', ' ').replace('$', ' ')
    normalized = re.sub(r'(?<=\d),(?=\d)', '', normalized)
    for expr, rhs in EQUATION.findall(normalized):
        try:
            errors += int(not _close(_evaluate(expr), float(rhs)))
        except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
            continue
    return errors


def _question_fractions(question):
    values = []
    for match in QUESTION_NUMBER.findall(question):
        try:
            values.append(Fraction(match.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            continue
    return values[:6]


def expression_coverage(question):
    """Map each reachable value to the most question-numbers used to make it.

    Questions with more than six numbers use the first six. The search is
    capped so inference stays well inside the time limit.
    """
    numbers = _question_fractions(question)
    count = len(numbers)
    if count < 2:
        return {}
    states = [dict() for _ in range(1 << count)]

    def add(bucket, value, used):
        if value.denominator > 5000 or abs(value.numerator) > 10 ** 8:
            return
        previous = bucket.get(value)
        if previous is None or used > previous:
            bucket[value] = used

    for index, value in enumerate(numbers):
        add(states[1 << index], value, 1)
    for mask in range(1, 1 << count):
        submask = mask
        while True:
            submask = (submask - 1) & mask
            if submask == 0:
                break
            other = mask ^ submask
            if submask > other or not states[submask] or not states[other]:
                continue
            bucket = states[mask]
            for left, left_used in list(states[submask].items()):
                for right, right_used in states[other].items():
                    used = left_used + right_used
                    add(bucket, left + right, used)
                    add(bucket, left * right, used)
                    add(bucket, left - right, used)
                    add(bucket, right - left, used)
                    if right:
                        add(bucket, left / right, used)
                    if left:
                        add(bucket, right / left, used)
            if len(bucket) > 12:
                kept = sorted(bucket.items(), key=lambda item: (-item[1], item[0].denominator))[:12]
                states[mask] = dict(kept)
    best = {}
    for bucket in states:
        for value, used in bucket.items():
            if best.get(value, -1) < used:
                best[value] = used
    return best
