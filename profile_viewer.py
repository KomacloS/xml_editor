from __future__ import annotations
from typing import Iterable
from rules_profiles import PROFILE_SETS, TolRule


def _print_table(label: str, rules: Iterable[TolRule]):
    print(f"    {label}:")
    for r in rules:
        print(
            f"      {r.vmin:g}-{r.vmax:g} -> ext +{r.extension_allowed_pct}% max {r.max_allowed_pct}%"
        )


def main() -> None:
    for name, ps in PROFILE_SETS.items():
        print(f"Profile set: {name}")
        if ps.resistor:
            _print_table("Resistors", ps.resistor.table_a)
            if ps.resistor.table_b:
                _print_table("Resistors (B)", ps.resistor.table_b)
        if ps.capacitor:
            _print_table("Capacitors", ps.capacitor.table_a)
            if ps.capacitor.table_b:
                _print_table("Capacitors (B)", ps.capacitor.table_b)
        if ps.inductor:
            _print_table("Inductors", ps.inductor.table_a)
            if ps.inductor.table_b:
                _print_table("Inductors (B)", ps.inductor.table_b)
        print()


if __name__ == "__main__":
    main()
