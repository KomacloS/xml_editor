from __future__ import annotations

"""GUI for building tolerance profile sets.

The window allows the user to define tolerance rules for resistors, capacitors
and inductors.  Values may be entered using standard SI prefixes (``m`` for
milli, ``u``/``µ`` for micro, etc.).  A simple coverage checker ensures that the
entered ranges cover the full span without gaps.
"""

import sys
import pathlib

from rules_profiles import (
    ComponentRules,
    ProfileSet,
    check_rules_cover,
    make_tol_rule,
    save_profile_set,
    register_profile_set,
)

try:  # pragma: no cover - the GUI is optional for tests
    from PyQt6.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QMessageBox,
        QFileDialog,
        QInputDialog,
        QLineEdit,
        QComboBox,
    )
    from PyQt6.QtCore import Qt, pyqtSignal
except Exception:  # pragma: no cover - allows import without PyQt installed
    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Dummy()

        def __call__(self, *a, **k):
            return _Dummy()

    class _Signal:
        def connect(self, *a, **k):
            pass

        def emit(self, *a, **k):
            pass

    def pyqtSignal(*a, **k):  # type: ignore
        return _Signal()

    QApplication = QWidget = QVBoxLayout = QHBoxLayout = QLabel = QPushButton = (
        QTableWidget
    ) = QTableWidgetItem = QTabWidget = QMessageBox = QFileDialog = QInputDialog = (
        QLineEdit
    ) = QComboBox = _Dummy
    Qt = type("Qt", (), {})


class ValueCell(QWidget):
    """Editor widget for a value with an SI prefix selector."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.prefix = QComboBox()
        self.prefix.addItems(["", "p", "n", "u", "µ", "m", "K", "M", "G"])
        self.prefix.setFixedWidth(45)
        layout.addWidget(self.edit)
        layout.addWidget(self.prefix)

    def text(self) -> str:
        return f"{self.edit.text().strip()}{self.prefix.currentText()}"


class RuleTable(QWidget):
    """Table widget used to enter rules for a single component type."""

    def __init__(self, unit: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.unit = unit
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Min", "Max", "Ext %", "Max %"])
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Row")
        self.btn_remove = QPushButton("Remove Row")
        self.btn_check = QPushButton("Check Coverage")
        self.btn_add.clicked.connect(self.add_row)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_check.clicked.connect(self.check)
        for w in (self.btn_add, self.btn_remove, self.btn_check):
            btn_row.addWidget(w)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

    def add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for c in (0, 1):
            self.table.setCellWidget(row, c, ValueCell())
        for c in (2, 3):
            self.table.setItem(row, c, QTableWidgetItem())

    def remove_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _cell_text(self, r: int, c: int) -> str:
        if c in (0, 1):
            widget = self.table.cellWidget(r, c)
            return widget.text().strip() if widget else ""
        item = self.table.item(r, c)
        return item.text().strip() if item else ""

    def get_rules(self):
        rules = []
        for r in range(self.table.rowCount()):
            vals = [self._cell_text(r, c) for c in range(4)]
            if any(v == "" for v in vals):
                raise ValueError(f"Row {r+1} has empty fields")
            rules.append(make_tol_rule(*vals))
        return rules

    def check(self) -> None:
        try:
            rules = self.get_rules()
        except Exception as e:  # pragma: no cover - GUI feedback
            QMessageBox.warning(self, "Invalid data", str(e))
            return
        errs = check_rules_cover(rules)
        if errs:  # pragma: no cover - GUI feedback
            QMessageBox.critical(self, "Coverage errors", "\n".join(errs))
        else:  # pragma: no cover - GUI feedback
            QMessageBox.information(self, "Coverage", "All ranges are contiguous.")


class ProfileConstructorWindow(QWidget):
    """Window that brings together rule tables for each component."""

    profile_saved = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Profile Constructor")

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.res_tab = RuleTable("Ω")
        self.cap_tab = RuleTable("F")
        self.ind_tab = RuleTable("H")
        self.tabs.addTab(self.res_tab, "Resistors")
        self.tabs.addTab(self.cap_tab, "Capacitors")
        self.tabs.addTab(self.ind_tab, "Inductors")
        layout.addWidget(self.tabs)

        self.btn_save = QPushButton("Save Profile Set…")
        self.btn_save.clicked.connect(self.save_profile)
        layout.addWidget(self.btn_save)

    def save_profile(self) -> None:
        try:
            r_rules = self.res_tab.get_rules()
            c_rules = self.cap_tab.get_rules()
            i_rules = self.ind_tab.get_rules()
        except Exception as e:  # pragma: no cover - GUI feedback
            QMessageBox.warning(self, "Invalid data", str(e))
            return

        for name, rules in ("Resistor", r_rules), ("Capacitor", c_rules), ("Inductor", i_rules):
            errs = check_rules_cover(rules)
            if errs:  # pragma: no cover - GUI feedback
                QMessageBox.critical(self, f"{name} errors", "\n".join(errs))
                return

        name, ok = QInputDialog.getText(self, "Profile name", "Profile set name:")
        name = name.strip()
        if not ok or not name:
            return

        prof = ProfileSet(
            name=name,
            resistor=ComponentRules(table_a=tuple(r_rules)) if r_rules else None,
            capacitor=ComponentRules(table_a=tuple(c_rules)) if c_rules else None,
            inductor=ComponentRules(table_a=tuple(i_rules)) if i_rules else None,
        )

        folder = pathlib.Path(__file__).resolve().parent / "profiles"
        folder.mkdir(exist_ok=True)
        default = folder / f"{name}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save profile set",
            str(default),
            "JSON files (*.json)",
        )
        if path:
            try:  # pragma: no cover - file IO
                save_profile_set(prof, path)
                register_profile_set(prof)
                self.profile_saved.emit(prof.name)
                QMessageBox.information(self, "Saved", f"Profile saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))


def main() -> None:  # pragma: no cover - manual GUI launch
    app = QApplication(sys.argv)
    w = ProfileConstructorWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()

