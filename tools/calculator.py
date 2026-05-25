"""Calculator tool — safely evaluates arithmetic expressions via AST walking.

We never call eval(). Instead we parse the expression with ast.parse() and
walk the tree, allowing only numeric literals and a known set of operators.
Anything else (attribute access, function calls, names, etc.) raises ValueError.
"""
from __future__ import annotations

import ast
import operator
from typing import Callable

from pydantic import BaseModel, Field

from .registry import register_tool


class CalculatorParams(BaseModel):
    expression: str = Field(
        description=(
            "A math expression using +, -, *, /, **, %, //, parentheses, and "
            "numeric literals. Example: '(4500 + 150 + 280) * 1.19'."
        ),
        min_length=1,
        max_length=200,
    )


@register_tool
def calculator(params: CalculatorParams) -> str:
    """Evaluates an arithmetic expression and returns the numeric result.

    Supports + - * / ** % //, unary minus, parentheses, integers and floats.
    Rejects anything else (variable names, function calls, attribute access).
    """
    return str(_safe_eval(params.expression))


_BINARY_OPERATORS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> float:
    return _eval_node(ast.parse(expression, mode="eval").body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))
    raise ValueError(f"Expression construct not allowed: {type(node).__name__}")