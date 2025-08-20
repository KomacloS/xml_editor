import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from rules_profiles import compute_new_tolerance_pct_for_ref


def test_elop_resistor_rules():
    assert compute_new_tolerance_pct_for_ref('ELOP', 'R1', 10, 1) == 20.0
    assert compute_new_tolerance_pct_for_ref('ELOP', 'R2', 50, 1) == 10.0
    assert compute_new_tolerance_pct_for_ref('ELOP', 'R3', 1000, 1) == 5.0


def test_elop_capacitor_rules():
    assert compute_new_tolerance_pct_for_ref('ELOP', 'C1', '10pF', 0) == 20.0
    assert compute_new_tolerance_pct_for_ref('ELOP', 'C2', '50pF', 0) == 10.0
    assert compute_new_tolerance_pct_for_ref('ELOP', 'C3', '1uF', 0) == 5.0


def test_elop_inductor_rules():
    assert compute_new_tolerance_pct_for_ref('ELOP', 'L1', '500uH', 0) == 15.0
    assert compute_new_tolerance_pct_for_ref('ELOP', 'L2', '10mH', 0) == 10.0
    assert compute_new_tolerance_pct_for_ref('ELOP', 'L3', '1H', 0) == 5.0

