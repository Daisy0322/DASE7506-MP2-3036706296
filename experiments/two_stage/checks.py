"""Checks that do not need a learned model.

`equation_errors` counts explicit arithmetic claims in a trace that are false,
for example `60 + 20 = 90` or `\\frac{0.6}{9} = \\frac{2}{15}`.
"""

import ast
import re

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
