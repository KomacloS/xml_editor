from rules_profiles import (
    ProfileSet,
    ComponentRules,
    make_tol_rule,
    save_profile_set,
    load_profile_set,
    register_profile_set,
    compute_new_tolerance_pct_for_ref,
)


def test_profile_set_round_trip(tmp_path):
    resistor = ComponentRules(table_a=(make_tol_rule('0', '10k', 10, 20),))
    capacitor = ComponentRules(table_a=(make_tol_rule('0', '10u', 5, 10),))
    inductor = ComponentRules(table_a=(make_tol_rule('0', '10m', 5, 10),))

    ps = ProfileSet(name='Combo', resistor=resistor, capacitor=capacitor, inductor=inductor)

    p = tmp_path / 'combo.json'
    save_profile_set(ps, p)
    loaded = load_profile_set(p)
    register_profile_set(loaded)

    assert compute_new_tolerance_pct_for_ref('Combo', 'R1', '5k', 1) == 11.0
    assert compute_new_tolerance_pct_for_ref('Combo', 'C1', '5u', 1) == 6.0
    assert compute_new_tolerance_pct_for_ref('Combo', 'L1', '5m', 1) == 6.0
