from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple, Union
import json
import pathlib
import re

Number = Union[int, float]

# Registry of single-component profiles and profile sets
PROFILES: Dict[str, "Profile"] = {}
PROFILE_SETS: Dict[str, "ProfileSet"] = {}

@dataclass(frozen=True)
class TolRule:
    """Represents a single tolerance rule."""

    vmin: float
    vmax: float
    extension_allowed_pct: float   # percent number (e.g., 3 == 3%)
    max_allowed_pct: float         # percent number

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Union[str, float, int]]) -> "TolRule":
        """Create a rule converting ``vmin``/``vmax`` from strings with SI prefixes.

        Values may contain prefixes such as ``'5m'`` or ``'10k'`` which are
        interpreted according to the standard SI multipliers.
        """

        return TolRule(
            parse_si_value(data["vmin"]),
            parse_si_value(data["vmax"]),
            float(data["extension_allowed_pct"]),
            float(data["max_allowed_pct"]),
        )

@dataclass(frozen=True)
class Profile:
    """Complete tolerance profile.

    ``value_parser`` converts user supplied values to the base unit.
    For profiles loaded from disk a generic :func:`parse_si_value` parser is
    used so that units with prefixes (micro, nano, milli, ...) are easy to use.
    """

    name: str
    unit: str
    value_parser: Callable[[Union[str, Number]], float]
    table_a: Tuple[TolRule, ...]
    table_b: Optional[Tuple[TolRule, ...]] = None
    default_threshold_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        data = {
            "name": self.name,
            "unit": self.unit,
            "default_threshold_pct": self.default_threshold_pct,
            "table_a": [r.to_dict() for r in self.table_a],
        }
        if self.table_b is not None:
            data["table_b"] = [r.to_dict() for r in self.table_b]
        return data

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "Profile":
        table_a = tuple(TolRule.from_dict(r) for r in data.get("table_a", []))
        table_b_data = data.get("table_b")
        table_b = tuple(TolRule.from_dict(r) for r in table_b_data) if table_b_data else None
        return Profile(
            name=data["name"],
            unit=data.get("unit", ""),
            value_parser=parse_si_value,
            table_a=table_a,
            table_b=table_b,
            default_threshold_pct=data.get("default_threshold_pct"),
        )

def parse_res_value_to_ohms(value: Union[str, Number]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().upper()
    if not s:
        raise ValueError('Empty resistor value')
    for token in ('Ω', 'OHM', ' '):
        s = s.replace(token, '')
    s = s.replace('M', 'E6').replace('K', 'E3')
    if s.endswith('R'):
        s = s[:-1]
        if s.endswith('.'):
            s = s[:-1]
    elif 'R' in s:
        raise ValueError(f"Unsupported 'R' format in '{value}'. Expected suffix like '22.1R'.")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Cannot parse resistor value '{value}'")

_CAP_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([pPnNuUµmM]?)[fF]?")  # trailing F optional
def parse_cap_value_to_farads(value: Union[str, Number]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise ValueError('Empty capacitor value')
    m = _CAP_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse capacitor value '{value}'")
    mag = float(m.group(1))
    prefix = (m.group(2) or '').lower()
    mult = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'm': 1e-3}.get(prefix, 1.0)
    return mag * mult

_IND_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([pPnNuUµmM]?)[hH]?")
def parse_ind_value_to_henries(value: Union[str, Number]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise ValueError('Empty inductor value')
    m = _IND_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse inductor value '{value}'")
    mag = float(m.group(1))
    prefix = (m.group(2) or '').lower()
    mult = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'm': 1e-3}.get(prefix, 1.0)
    return mag * mult

_SI_PREFIXES = {
    'p': 1e-12,
    'n': 1e-9,
    'u': 1e-6,
    'µ': 1e-6,
    'm': 1e-3,
    'k': 1e3,
    'M': 1e6,
    'g': 1e9,
    'G': 1e9,
}

_SI_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([pPnNuUµmMkKgG]?)(?:[A-Za-zΩ]*)?\s*$")

def parse_si_value(value: Union[str, Number]) -> float:
    """Parse a number that may use an SI prefix.

    This helper allows users to enter values like ``'10k'`` or ``'5.6u'`` which
    are converted to base units.  The function is used by profiles loaded from
    files and by helper constructors.
    """

    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise ValueError("Empty value")
    m = _SI_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse value '{value}'")
    mag = float(m.group(1))
    prefix = (m.group(2) or '')
    mult = _SI_PREFIXES.get(prefix, 1.0)
    return mag * mult


def make_tol_rule(vmin: Union[str, Number],
                  vmax: Union[str, Number],
                  extension_allowed_pct: Union[str, Number],
                  max_allowed_pct: Union[str, Number]) -> TolRule:
    """Create a :class:`TolRule` from possibly prefixed values."""

    return TolRule(
        parse_si_value(vmin),
        parse_si_value(vmax),
        float(extension_allowed_pct),
        float(max_allowed_pct),
    )


@dataclass(frozen=True)
class ComponentRules:
    """Tolerance tables for a single component type."""

    table_a: Tuple[TolRule, ...]
    table_b: Optional[Tuple[TolRule, ...]] = None
    default_threshold_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        data = {
            "table_a": [r.to_dict() for r in self.table_a],
            "default_threshold_pct": self.default_threshold_pct,
        }
        if self.table_b is not None:
            data["table_b"] = [r.to_dict() for r in self.table_b]
        return data

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ComponentRules":
        table_a = tuple(TolRule.from_dict(r) for r in data.get("table_a", []))
        table_b_data = data.get("table_b")
        table_b = tuple(TolRule.from_dict(r) for r in table_b_data) if table_b_data else None
        return ComponentRules(
            table_a=table_a,
            table_b=table_b,
            default_threshold_pct=data.get("default_threshold_pct"),
        )


@dataclass(frozen=True)
class ProfileSet:
    """Collection of rules for resistors, capacitors and inductors."""

    name: str
    resistor: Optional[ComponentRules] = None
    capacitor: Optional[ComponentRules] = None
    inductor: Optional[ComponentRules] = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {"name": self.name}
        if self.resistor:
            data["resistor"] = self.resistor.to_dict()
        if self.capacitor:
            data["capacitor"] = self.capacitor.to_dict()
        if self.inductor:
            data["inductor"] = self.inductor.to_dict()
        return data

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ProfileSet":
        def _get(key: str) -> Optional[ComponentRules]:
            block = data.get(key)
            return ComponentRules.from_dict(block) if block else None

        return ProfileSet(
            name=data["name"],
            resistor=_get("resistor"),
            capacitor=_get("capacitor"),
            inductor=_get("inductor"),
        )


def save_profile(profile: Profile, path: Union[str, pathlib.Path]) -> None:
    """Serialize ``profile`` to ``path`` in JSON format."""

    p = pathlib.Path(path)
    with p.open('w', encoding='utf-8') as f:
        json.dump(profile.to_dict(), f, indent=2, sort_keys=True)


def load_profile(path: Union[str, pathlib.Path]) -> Profile:
    """Load a profile from ``path`` previously written by :func:`save_profile`."""

    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding='utf-8'))
    return Profile.from_dict(data)


def register_profile(profile: Profile) -> None:
    """Register ``profile`` so it can be used with :func:`compute_new_tolerance_pct`."""

    PROFILES[profile.name] = profile


def save_profile_set(profile: ProfileSet, path: Union[str, pathlib.Path]) -> None:
    """Serialize ``profile`` to ``path`` in JSON format."""

    p = pathlib.Path(path)
    with p.open('w', encoding='utf-8') as f:
        json.dump(profile.to_dict(), f, indent=2, sort_keys=True)


def load_profile_set(path: Union[str, pathlib.Path]) -> ProfileSet:
    """Load a profile set from ``path`` previously written by :func:`save_profile_set`."""

    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding='utf-8'))
    return ProfileSet.from_dict(data)


def register_profile_set(ps: ProfileSet) -> None:
    """Register the component profiles contained in ``ps``."""
    PROFILE_SETS[ps.name] = ps
    if ps.resistor:
        register_profile(
            Profile(
                name=f"{ps.name} (Resistors)",
                unit='Ω',
                value_parser=parse_res_value_to_ohms,
                table_a=ps.resistor.table_a,
                table_b=ps.resistor.table_b,
                default_threshold_pct=ps.resistor.default_threshold_pct,
            )
        )
    if ps.capacitor:
        register_profile(
            Profile(
                name=f"{ps.name} (Capacitors)",
                unit='F',
                value_parser=parse_cap_value_to_farads,
                table_a=ps.capacitor.table_a,
                table_b=ps.capacitor.table_b,
                default_threshold_pct=ps.capacitor.default_threshold_pct,
            )
        )
    if ps.inductor:
        register_profile(
            Profile(
                name=f"{ps.name} (Inductors)",
                unit='H',
                value_parser=parse_ind_value_to_henries,
                table_a=ps.inductor.table_a,
                table_b=ps.inductor.table_b,
                default_threshold_pct=ps.inductor.default_threshold_pct,
            )
        )

MABAT_A = (
    TolRule(0.005, 0.05, 19.9, 20.0),
    TolRule(0.051, 3.99, 14.9, 15.0),
    TolRule(4.0,   9.99,  9.9, 14.0),
    TolRule(10.0,  99.9,  4.9,  5.0),
    TolRule(100.0, 1999.0,2.9,  3.0),
    TolRule(2000.0,10000.0,0.9, 1.0),
)
MABAT_B = (
    TolRule(0.005, 0.05, 24.0, 25.0),
    TolRule(0.051, 3.99, 19.0, 20.0),
    TolRule(4.0,   9.99, 14.0, 15.0),
    TolRule(10.0,  99.9,  9.0, 10.0),
    TolRule(100.0, 1999.0,4.0,  5.0),
    TolRule(2000.0,10000.0,2.0, 3.0),
)

CAP_STD = (
    TolRule(1.00e-20, 4.99e-10, 20.0, 30.0),
    TolRule(5.00e-10, 9.99e-09, 15.0, 25.0),
    TolRule(1.00e-08, 9.99e-07,  5.0, 15.0),
    TolRule(1.00e-06, 1.00e+00,  3.0, 13.0),
)

IND_STD = (
    TolRule(1.00e-05, 1.00e-03, 15.0, 35.0),
    TolRule(1.00e-03, 1.00e+00,  5.0, 25.0),
)

ELOP_RES = (
    TolRule(100.0001, 1.00e+09, 5.0, 5.0),
    TolRule(33.0,    100.0,    10.0, 10.0),
    TolRule(0.0,      32.9999, 20.0, 20.0),
)

ELOP_CAP = (
    TolRule(1.00e-07, 1.00e+03, 5.0, 5.0),
    TolRule(3.00e-11, 9.99e-08,10.0,10.0),
    TolRule(0.0,      2.999e-11,20.0,20.0),
)

ELOP_IND = (
    TolRule(1.00e-01, 1.00e+09, 5.0, 5.0),
    TolRule(1.00e-03, 9.99e-02,10.0,10.0),
    TolRule(0.0,      9.99e-04,15.0,15.0),
)

# Register built-in profile sets
register_profile_set(
    ProfileSet(
        name='MABAT',
        resistor=ComponentRules(
            table_a=tuple(MABAT_A),
            table_b=tuple(MABAT_B),
            default_threshold_pct=0.1,
        ),
        capacitor=ComponentRules(table_a=tuple(CAP_STD)),
        inductor=ComponentRules(table_a=tuple(IND_STD)),
    )
)

register_profile_set(
    ProfileSet(
        name='ELOP',
        resistor=ComponentRules(table_a=tuple(ELOP_RES)),
        capacitor=ComponentRules(table_a=tuple(ELOP_CAP)),
        inductor=ComponentRules(table_a=tuple(ELOP_IND)),
    )
)

def _select_rule(x: float, table: Tuple[TolRule, ...]) -> TolRule:
    for r in table:
        if r.vmin <= x <= r.vmax:
            return r
    raise ValueError(f"value {x} is out of supported range ({table[0].vmin}–{table[-1].vmax}).")

def compute_new_tolerance_pct(profile_name: str,
                              value_text: Union[str, Number],
                              bom_tol_pct: float,
                              threshold_pct: Optional[float] = None) -> float:
    profile = PROFILES[profile_name]
    x = profile.value_parser(value_text)
    if profile.table_b is not None:
        thr = profile.default_threshold_pct if threshold_pct is None else float(threshold_pct)
        table = profile.table_a if bom_tol_pct <= thr else profile.table_b
    else:
        table = profile.table_a
    rule = _select_rule(x, table)
    return float(min(bom_tol_pct + rule.extension_allowed_pct, rule.max_allowed_pct))


def compute_new_tolerance_pct_for_ref(profile_set: str,
                                      reference: str,
                                      value_text: Union[str, Number],
                                      bom_tol_pct: float,
                                      threshold_pct: Optional[float] = None) -> float:
    """Like :func:`compute_new_tolerance_pct` but pick component rules by reference.

    ``reference`` is expected to start with ``R``/``C``/``L`` designator.
    """

    ps = PROFILE_SETS[profile_set]
    prefix = str(reference).strip().upper()[:1]
    if prefix == 'R' and ps.resistor:
        profile_name = f"{ps.name} (Resistors)"
    elif prefix == 'C' and ps.capacitor:
        profile_name = f"{ps.name} (Capacitors)"
    elif prefix == 'L' and ps.inductor:
        profile_name = f"{ps.name} (Inductors)"
        
    else:
        raise ValueError(f"Unsupported reference '{reference}' for profile set '{profile_set}'")
    return compute_new_tolerance_pct(profile_name, value_text, bom_tol_pct, threshold_pct)
