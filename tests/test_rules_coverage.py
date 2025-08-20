from rules_profiles import make_tol_rule, check_rules_cover


def test_no_gaps():
    rules = [
        make_tol_rule("0", "10", 1, 1),
        make_tol_rule("10", "20", 1, 1),
    ]
    assert check_rules_cover(rules) == []


def test_detect_gap():
    rules = [
        make_tol_rule("0", "5", 1, 1),
        make_tol_rule("10", "15", 1, 1),
    ]
    errs = check_rules_cover(rules)
    assert any("gap" in e for e in errs)

