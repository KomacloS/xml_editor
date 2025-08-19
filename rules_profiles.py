from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union
import re

Number = Union[int, float]

@dataclass(frozen=True)
class TolRule:
    vmin: float
    vmax: float
    extension_allowed_pct: float   # percent number (e.g., 3 == 3%)
    max_allowed_pct: float         # percent number

@dataclass(frozen=True)
class Profile:
    name: str
    unit: str
    value_parser: Callable[[Union[str, Number]], float]
    table_a: Tuple[TolRule, ...]
    table_b: Optional[Tuple[TolRule, ...]] = None
    default_threshold_pct: Optional[float] = None

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

PROFILES = {
    'MABAT (Resistors)': ('Ω', parse_res_value_to_ohms, tuple(MABAT_A), tuple(MABAT_B), 0.1),
    'Capacitor (F)':     ('F', parse_cap_value_to_farads, tuple(CAP_STD), None, None),
    'Inductor (H)':      ('H', parse_ind_value_to_henries, tuple(IND_STD), None, None),
    'ELOP (Resistors)':  ('Ω', parse_res_value_to_ohms, tuple(ELOP_RES), None, None),
    'ELOP (Capacitors)': ('F', parse_cap_value_to_farads, tuple(ELOP_CAP), None, None),
    'ELOP (Inductors)':  ('H', parse_ind_value_to_henries, tuple(ELOP_IND), None, None),
}

def _select_rule(x: float, table: Tuple[TolRule, ...]) -> TolRule:
    for r in table:
        if r.vmin <= x <= r.vmax:
            return r
    raise ValueError(f"value {x} is out of supported range ({table[0].vmin}–{table[-1].vmax}).")

def compute_new_tolerance_pct(profile_name: str,
                              value_text: Union[str, Number],
                              bom_tol_pct: float,
                              threshold_pct: Optional[float] = None) -> float:
    unit, parser, table_a, table_b, default_thr = PROFILES[profile_name]
    x = parser(value_text)
    if table_b is not None:
        thr = default_thr if threshold_pct is None else float(threshold_pct)
        table = table_a if bom_tol_pct <= thr else table_b
    else:
        table = table_a
    rule = _select_rule(x, table)
    return float(min(bom_tol_pct + rule.extension_allowed_pct, rule.max_allowed_pct))
