import math

from rules_profiles import Profile, TolRule, compute_new_tolerance_pct, PROFILES


def test_edge_values_use_more_strict_rule():
    profile_name = "Edge Test Profile"
    profile = Profile(
        name=profile_name,
        unit="F",
        value_parser=float,
        table_a=(
            TolRule(0.0, 100.0, 10.0, 10.0),
            TolRule(100.0, 200.0, 5.0, 5.0),
        ),
    )
    PROFILES[profile_name] = profile
    try:
        result = compute_new_tolerance_pct(profile_name, 100.0, bom_tol_pct=1.0)
    finally:
        PROFILES.pop(profile_name, None)

    assert math.isclose(result, 5.0)
