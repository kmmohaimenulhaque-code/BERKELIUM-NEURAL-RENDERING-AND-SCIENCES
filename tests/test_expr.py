import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from berkelium.expr import ExprError, ExprSyntaxError, eval_str, free_names, parse, to_source
from berkelium.units import Quantity


def test_precedence_and_associativity():
    assert eval_str("2 + 3 * 4").si == 14
    assert eval_str("2 ^ 3 ^ 2").si == 512           # right-assoc
    assert eval_str("-2 ^ 2").si == -4                # unary binds looser than ^
    assert eval_str("10 - 4 - 3").si == 3             # left-assoc
    assert eval_str("2 ^ -1").si == 0.5


def test_units_and_dimensions():
    assert eval_str("2[mm] + 3[mm]").to("mm") == pytest.approx(5)
    assert eval_str("1[in] + 1[mm]").to("mm") == pytest.approx(26.4)
    assert eval_str("sqrt(4[mm^2])").to("mm") == pytest.approx(2)
    with pytest.raises(ExprError):
        eval_str("1[mm] + 1[N]")
    with pytest.raises(ExprError):
        eval_str("sin(1[mm])")
    with pytest.raises(ExprError):
        eval_str("1[mm] ^ 0.5")


def test_constraint_style_expression():
    env = {"od": Quantity.of(40, "mm"), "bore": Quantity.of(10, "mm"), "min_wall": Quantity.of(3, "mm")}
    assert eval_str("od - bore >= 2*min_wall", env) is True
    assert eval_str("od - bore >= 20*min_wall", env) is False
    assert free_names(parse("od - bore >= 2*min_wall + pi")) == {"od", "bore", "min_wall"}


def test_involute_oracle():
    # KHK Gear Technical Reference Eq. 3.2: inv(20 deg) = 0.014904
    assert eval_str("inv(20[deg])").si == pytest.approx(0.0149043839, abs=1e-9)


def test_logic_and_if():
    assert eval_str("if(1 < 2, 3, 4)").si == 3
    assert eval_str("not (1 > 2) and true") is True
    with pytest.raises(ExprError):
        eval_str("if(1, 2, 3)")


def test_errors_are_typed():
    for bad in ["1 +", "a < b < c", "foo(", "1 $ 2", "__import__('os')"]:
        with pytest.raises((ExprSyntaxError, ExprError)):
            eval_str(bad)
    with pytest.raises(ExprError):
        eval_str("1/0")
    with pytest.raises(ExprError):
        eval_str("unknown_name + 1")
    with pytest.raises(ExprError):
        eval_str("sqrt(-1)")


def test_no_python_eval_reachable():
    # attribute access / subscripts are not part of the grammar at all
    for bad in ["().__class__", "x[0]", "lambda: 1"]:
        with pytest.raises(ExprSyntaxError):
            parse(bad)


def test_depth_limit():
    with pytest.raises(ExprSyntaxError):
        parse("(" * 100 + "1" + ")" * 100)


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), st.floats(min_value=-1e6, max_value=1e6,
                                                                            allow_nan=False))
def test_roundtrip_and_determinism(a, b):
    src = f"({a!r} - {b!r}) * 2 + max({a!r}, {b!r})"
    node = parse(src)
    assert parse(to_source(node)) == node
    assert eval_str(src).si == eval_str(src).si
    assert eval_str(src).si == pytest.approx((a - b) * 2 + max(a, b))
