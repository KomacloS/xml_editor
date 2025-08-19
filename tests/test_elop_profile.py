import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from rules_profiles import compute_new_tolerance_pct


def test_elop_resistor_rules():
    assert compute_new_tolerance_pct('ELOP (Resistors)', 10, 1) == 20.0
    assert compute_new_tolerance_pct('ELOP (Resistors)', 50, 1) == 10.0
    assert compute_new_tolerance_pct('ELOP (Resistors)', 1000, 1) == 5.0


def test_elop_capacitor_rules():
    assert compute_new_tolerance_pct('ELOP (Capacitors)', '10pF', 0) == 20.0
    assert compute_new_tolerance_pct('ELOP (Capacitors)', '50pF', 0) == 10.0
    assert compute_new_tolerance_pct('ELOP (Capacitors)', '1uF', 0) == 5.0


def test_elop_inductor_rules():
    assert compute_new_tolerance_pct('ELOP (Inductors)', '500uH', 0) == 15.0
    assert compute_new_tolerance_pct('ELOP (Inductors)', '10mH', 0) == 10.0
    assert compute_new_tolerance_pct('ELOP (Inductors)', '1H', 0) == 5.0

