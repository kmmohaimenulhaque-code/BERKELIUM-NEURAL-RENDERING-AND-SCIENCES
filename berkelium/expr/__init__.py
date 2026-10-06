"""Berkelium expression language: typed AST, safe parser, dimension-checked evaluator.

Grammar (precedence low→high): ``or`` < ``and`` < ``not`` < comparisons (non-chaining)
< ``+ -`` < ``* /`` < unary ``- +`` < ``^`` (right-assoc). Literals may carry a unit:
``2.5[mm]``. Calls: see ``FUNCTIONS`` plus ``if(cond, a, b)``. Dotted names
(``gear_a.tip_diameter``) refer to parameters or measurements.
"""

from .ast import Binary, Bool, Call, Name, Node, Num, Unary, free_names, to_source
from .evaluator import CONSTANTS, FUNCTIONS, ExprError, Value, evaluate
from .parser import ExprSyntaxError, parse


def eval_str(src: str, env=None) -> Value:
    """Parse and evaluate in one step."""
    return evaluate(parse(src), env)


__all__ = ["Binary", "Bool", "Call", "Name", "Node", "Num", "Unary", "free_names", "to_source",
           "CONSTANTS", "FUNCTIONS", "ExprError", "Value", "evaluate", "ExprSyntaxError", "parse", "eval_str"]
