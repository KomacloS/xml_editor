from rules_profiles import (
    Profile,
    make_tol_rule,
    save_profile,
    load_profile,
    register_profile,
    compute_new_tolerance_pct,
    parse_si_value,
)


def test_save_load_and_use_profile(tmp_path):
    rules = [
        make_tol_rule('0', '10k', 10, 20),
        make_tol_rule('10k', '100k', 5, 10),
    ]
    prof = Profile(
        name='TempProfile',
        unit='Ω',
        value_parser=parse_si_value,
        table_a=tuple(rules),
    )

    p = tmp_path / 'profile.json'
    save_profile(prof, p)
    loaded = load_profile(p)
    register_profile(loaded)

    assert compute_new_tolerance_pct('TempProfile', '5k', 1) == 11.0
    assert compute_new_tolerance_pct('TempProfile', '20k', 1) == 6.0

