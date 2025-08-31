import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

from tolerance_gui import build_df_from_xml
from rules_profiles import compute_new_tolerance_pct_for_ref


def test_build_df_from_xml_scientific(tmp_path: Path):
    root = ET.Element('Root')
    c1 = ET.SubElement(root, 'Component', Name='C1', TolP='5')
    ET.SubElement(c1, 'Parameter', Name='Value', Value='2.2E-05', Unit='F')
    tree = ET.ElementTree(root)
    xml_path = tmp_path / 'sample.xml'
    tree.write(xml_path)
    df = build_df_from_xml([xml_path])
    val = df.loc[df['Ref'] == 'C1', 'Value'].iloc[0]
    assert val == pytest.approx(2.2e-05)


@pytest.mark.parametrize('value', [2.2e-05, 9.6e-06, 4.7e-06, 3.6e-10])
def test_compute_cap_values(value):
    res = compute_new_tolerance_pct_for_ref('MABAT', 'C1', value, 5)
    assert isinstance(res, float)


@pytest.mark.parametrize('value', [2.2e-06])
def test_compute_ind_values(value):
    # Use a profile that covers very small inductance values
    res = compute_new_tolerance_pct_for_ref('ELOP', 'L1', value, 5)
    assert isinstance(res, float)
