import pandas as pd
import sys
import types
from pathlib import Path

# Provide dummy PyQt6 modules so tolerance_gui can be imported without Qt
pyqt6 = types.ModuleType("PyQt6")
pyqt6.QtCore = types.ModuleType("PyQt6.QtCore")
setattr(pyqt6.QtCore, "Qt", types.SimpleNamespace(WindowModality=types.SimpleNamespace(ApplicationModal=0)))
pyqt6.QtWidgets = types.ModuleType("PyQt6.QtWidgets")
pyqt6.QtGui = types.ModuleType("PyQt6.QtGui")

class _Dummy:
    def __init__(self, *a, **kw):
        pass

for name in ["QStandardItemModel", "QStandardItem"]:
    setattr(pyqt6.QtGui, name, _Dummy)
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

sys.modules.setdefault("PyQt6", pyqt6)
sys.modules.setdefault("PyQt6.QtCore", pyqt6.QtCore)
sys.modules.setdefault("PyQt6.QtWidgets", pyqt6.QtWidgets)
sys.modules.setdefault("PyQt6.QtGui", pyqt6.QtGui)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tolerance_gui import filter_nonempty


def test_filter_nonempty_drops_blank_and_nan():
    df = pd.DataFrame({
        "Ref": ["A", "B", "C", "D"],
        "Tol": ["0.1", "", float('nan'), "  "]
    })
    filtered = filter_nonempty(df, "Tol")
    assert list(filtered["Ref"]) == ["A"]
