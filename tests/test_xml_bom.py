import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from tolerance_gui import build_df_from_xml


def create_sample_xml(path: Path):
    root = ET.Element('Root')
    c1 = ET.SubElement(root, 'Component', Name='R1', TolP='1', TolN='-1')
    ET.SubElement(c1, 'Parameter', Name='Value', Value='10k')
    c2 = ET.SubElement(root, 'Component', Name='C1', TolP='5')
    ET.SubElement(c2, 'Parameter', Name='Value', Value='100n')
    tree = ET.ElementTree(root)
    tree.write(path)


def test_build_df_from_xml(tmp_path):
    xml_path = tmp_path / 'sample.xml'
    create_sample_xml(xml_path)
    df = build_df_from_xml([xml_path])
    assert list(df.columns) == ['Ref', 'Value', 'TolP', 'TolN']
    assert df.shape[0] == 2
    r1 = df[df['Ref'] == 'R1'].iloc[0]
    assert r1['Value'] == '10k'
    assert r1['TolP'] == '1'
    assert r1['TolN'] == '-1'
    c1 = df[df['Ref'] == 'C1'].iloc[0]
    assert c1['TolP'] == '5'
    assert c1['TolN'] == ''
