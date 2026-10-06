import math

import pytest

from berkelium.units import DIMENSIONLESS, Quantity, UnitError, parse_unit


def test_basic_conversions():
    assert Quantity.of(25.4, "mm").to("in") == pytest.approx(1.0)
    assert Quantity.of(1, "kpsi").to("MPa") == pytest.approx(6.894757293168361)
    assert Quantity.of(180, "deg").to("rad") == pytest.approx(math.pi)
    assert Quantity.of(1, "hp").to("W") == pytest.approx(745.6998715822702)


def test_compound_units():
    f, d = parse_unit("N/mm^2")
    assert d == parse_unit("MPa")[1] and f == pytest.approx(1e6)
    assert Quantity.of(1, "N*m").to("N*mm") == pytest.approx(1000)
    assert Quantity.of(60, "rev/min").to("rad/s") == pytest.approx(2 * math.pi)


def test_errors():
    with pytest.raises(UnitError):
        parse_unit("furlong")
    with pytest.raises(UnitError):
        Quantity.of(1, "mm").to("N")
    with pytest.raises(UnitError):
        parse_unit("mm^x")


def test_angle_dimensionless():
    assert parse_unit("deg")[1] == DIMENSIONLESS
