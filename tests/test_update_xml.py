import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import types

# Provide dummy PyQt6 modules so tolerance_gui can be imported without Qt
pyqt6 = types.ModuleType("PyQt6")
pyqt6.QtCore = types.ModuleType("PyQt6.QtCore")
setattr(pyqt6.QtCore, "Qt", types.SimpleNamespace(WindowModality=types.SimpleNamespace(ApplicationModal=0)))
pyqt6.QtWidgets = types.ModuleType("PyQt6.QtWidgets")
pyqt6.QtGui = types.ModuleType("PyQt6.QtGui")

# Dummy widget classes used only for class definitions
class _Dummy:  # pragma: no cover - simple stub
    def __init__(self, *args, **kwargs):
        pass

for name in [
    "QApplication",
    "QWizard",
    "QWizardPage",
    "QFileDialog",
    "QVBoxLayout",
    "QLabel",
    "QPushButton",
    "QListWidget",
    "QComboBox",
    "QTableView",
    "QMessageBox",
    "QProgressDialog",
]:
    setattr(pyqt6.QtWidgets, name, _Dummy)
for name in ["QStandardItemModel", "QStandardItem"]:
    setattr(pyqt6.QtGui, name, _Dummy)
sys.modules.setdefault("PyQt6", pyqt6)
sys.modules.setdefault("PyQt6.QtCore", pyqt6.QtCore)
sys.modules.setdefault("PyQt6.QtWidgets", pyqt6.QtWidgets)
sys.modules.setdefault("PyQt6.QtGui", pyqt6.QtGui)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tolerance_gui import update_xml

def create_xml(path: Path):
    content = '''<?xml version="1.0" encoding="Windows-1252"?>
<Components>
  <Component Name="R1" TolP="1" TolN="1">
    <Parameter Name="TolPos" Value="1"/>
    <Parameter Name="TolNeg" Value="1"/>
  </Component>
  <Component Name="R2" TolP="2" TolN="2">
    <Parameter Name="TolPos" Value="2"/>
    <Parameter Name="TolNeg" Value="2"/>
  </Component>
  <Component Name="R3" TolP="3" TolN="3">
    <Parameter Name="TolPos" Value="3"/>
    <Parameter Name="TolNeg" Value="3"/>
  </Component>
  <Component Name="R4" TolP="4" TolN="4">
    <Parameter Name="TolPos" Value="4"/>
    <Parameter Name="TolNeg" Value="4"/>
  </Component>
</Components>'''
    path.write_text(content, encoding="Windows-1252")


def test_update_xml(tmp_path: Path):
    xml_path = tmp_path / "test.xml"
    create_xml(xml_path)

    df = pd.DataFrame({
        "Ref": ["R1", "R2", "R3"],
        "Tol+": ["11", "", "13"],
        "Tol-": ["12", "", ""]
    })

    new_path, changed = update_xml(xml_path, "Ref", "Tol+", "Tol-", df)
    assert new_path.exists()
    assert set(changed) == {"R1", "R3"}

    tree = ET.parse(new_path)
    root = tree.getroot()

    comps = {c.get("Name"): c for c in root.iter("Component")}
    # R1 updated fully
    c1 = comps["R1"]
    assert c1.get("TolP") == "11"
    assert c1.get("TolN") == "12"
    # R2 unchanged
    c2 = comps["R2"]
    assert c2.get("TolP") == "2"
    assert c2.get("TolN") == "2"
    # R3 updated only positive
    c3 = comps["R3"]
    assert c3.get("TolP") == "13"
    assert c3.get("TolN") == "3"
    # R4 untouched
    c4 = comps["R4"]
    assert c4.get("TolP") == "4"
    assert c4.get("TolN") == "4"
