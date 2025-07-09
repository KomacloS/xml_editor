# Placeholder for tolerance_gui.py – insert full code here.
# tolerance_gui.py
"""
Graphical tool to patch tolerances in Seica XML component files using a BOM
---------------------------------------------------------------------------
July 2025 – **revision 3 (completed)**

Implements a wizard‑style GUI that:

1. Lets the user pick **one BOM** (CSV/Excel) and **any number of XML** files.
2. Maps columns: Reference, Tol + (mandatory, non‑blank), Tol – (optional).
3. Previews *all* BOM rows that will actually be applied (blank‑Tol rows are
   filtered out automatically).
4. Applies the new tolerances strictly to components whose ref **exists in
   the BOM** **and** whose Tol + is non‑blank.
5. Writes every XML back as `<original>_updated.XML` alongside the source and
   shows a detailed summary of every component updated.

Run:
```bash
pip install PyQt6 pandas openpyxl
python tolerance_gui.py
```
"""
from __future__ import annotations

import sys
import pathlib
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWizard,
    QWizardPage,
    QFileDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QComboBox,
    QTableView,
    QMessageBox,
    QProgressDialog,
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem

###############################################################################
# Helper functions
###############################################################################

def load_bom(path: pathlib.Path) -> pd.DataFrame:
    """Return BOM as DataFrame; supports CSV and Excel."""
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    return df


def filter_nonempty(df: pd.DataFrame, tol_col: str) -> pd.DataFrame:
    """Keep only rows where the *positive* tolerance column isn’t blank/NaN."""
    series = df[tol_col].astype(str).str.strip()
    return df[series.ne("") & series.notna()].copy()


def update_xml(
    xml_path: pathlib.Path,
    ref_col: str,
    tolp_col: str,
    toln_col: str,
    df: pd.DataFrame,
) -> tuple[pathlib.Path, List[str]]:
    """Patch tolerances in a single XML and return (new_path, list_of_refs_changed)."""
    # Build lookup dicts (references upper‑cased, stripped)
    refs = df[ref_col].astype(str).str.upper().str.strip()
    tol_pos_series = df[tolp_col]
    tol_neg_series = df[toln_col] if toln_col != tolp_col else tol_pos_series

    tol_pos_map: Dict[str, str] = dict(zip(refs, tol_pos_series))
    tol_neg_map: Dict[str, str] = dict(zip(refs, tol_neg_series))

    tree = ET.parse(xml_path)
    root = tree.getroot()
    updated: List[str] = []

    for comp in root.iter("Component"):
        name = comp.get("Name", "").upper().strip()
        if name not in tol_pos_map:
            continue  # not listed in BOM → leave untouched

        tol_p_raw = tol_pos_map[name]
        tol_p = "" if pd.isna(tol_p_raw) else str(tol_p_raw).strip()
        if tol_p == "":
            continue  # Tol+ blank in BOM → skip

        tol_n_raw = tol_neg_map.get(name)
        if pd.isna(tol_n_raw) or str(tol_n_raw).strip() == "":
            tol_n = None  # keep existing value
        else:
            tol_n = str(tol_n_raw).strip()

        # Update attributes
        comp.set("TolP", tol_p)
        if tol_n is not None:
            comp.set("TolN", tol_n)

        # Update nested parameters
        for prm in comp.iter("Parameter"):
            if prm.get("Name") == "TolPos":
                prm.set("Value", tol_p)
            elif prm.get("Name") == "TolNeg":
                if tol_n is not None:
                    prm.set("Value", tol_n)

        updated.append(name)

    new_path = xml_path.with_name(xml_path.stem + "_updated" + xml_path.suffix)
    tree.write(new_path, encoding="Windows-1252", xml_declaration=True)
    return new_path, updated

###############################################################################
# Wizard pages
###############################################################################

class FileSelectPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 1 – Select files")
        layout = QVBoxLayout(self)

        self.xml_list = QListWidget()
        self.bom_label = QLabel("<i>No BOM selected</i>")

        btn_add_xml = QPushButton("Add XML files…")
        btn_add_xml.clicked.connect(self.add_xml_files)
        btn_bom = QPushButton("Choose BOM file…")
        btn_bom.clicked.connect(self.choose_bom_file)

        layout.addWidget(btn_add_xml)
        layout.addWidget(self.xml_list)
        layout.addWidget(btn_bom)
        layout.addWidget(self.bom_label)

    # --------------------------- file selection helpers --------------------
    def add_xml_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Pick XML component files", "", "XML files (*.xml)")
        for f in files:
            p = pathlib.Path(f)
            if p.suffix.lower() == ".xml" and p not in self.xml_paths:
                self.xml_paths.append(p)
                self.xml_list.addItem(str(p))
        self.completeChanged.emit()

    def choose_bom_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Pick BOM file (CSV / Excel)",
            "",
            "BOM files (*.csv *.xlsx *.xlsm *.xls)",
        )
        if f:
            self.bom_path = pathlib.Path(f)
            self.bom_label.setText(str(self.bom_path))
            self.completeChanged.emit()

    # --------------------------- page‑lifecycle ----------------------------
    def initializePage(self):
        self.xml_paths: List[pathlib.Path] = []
        self.bom_path: pathlib.Path | None = None

    def isComplete(self) -> bool:
        return bool(self.xml_paths and self.bom_path)

    def validatePage(self) -> bool:
        try:
            df = load_bom(self.bom_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load BOM: {e}")
            return False
        self.wizard().df_raw = df
        self.wizard().xml_paths = self.xml_paths
        return True


class ColumnMapPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 2 – Map BOM columns")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select which BOM columns contain the component reference and tolerances:"))

        self.combo_ref = QComboBox()
        self.combo_pos = QComboBox()
        self.combo_neg = QComboBox()

        layout.addWidget(QLabel("Reference column:"))
        layout.addWidget(self.combo_ref)
        layout.addWidget(QLabel("Positive tolerance column:"))
        layout.addWidget(self.combo_pos)
        layout.addWidget(QLabel("Negative tolerance column (can be same):"))
        layout.addWidget(self.combo_neg)

    def initializePage(self):
        cols = list(self.wizard().df_raw.columns)
        for cb in (self.combo_ref, self.combo_pos, self.combo_neg):
            cb.clear(); cb.addItems(cols)

    def isComplete(self) -> bool:
        return bool(self.combo_ref.currentText() and self.combo_pos.currentText())

    def validatePage(self) -> bool:
        wiz = self.wizard()
        wiz.ref_col = self.combo_ref.currentText()
        wiz.tolp_col = self.combo_pos.currentText()
        wiz.toln_col = self.combo_neg.currentText()

        wiz.df = filter_nonempty(wiz.df_raw, wiz.tolp_col)
        if wiz.df.empty:
            QMessageBox.warning(self, "No data", "Selected positive‑tolerance column has no non‑blank values.")
            return False
        return True


class PreviewPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 3 – Preview components to be updated")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Rows with blank Tol+ are already filtered out:"))
        self.table_view = QTableView()
        layout.addWidget(self.table_view)

    def initializePage(self):
        wiz = self.wizard()
        df = wiz.df[[wiz.ref_col, wiz.tolp_col, wiz.toln_col]].copy()

        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels(df.columns.tolist())
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                model.setItem(r, c, QStandardItem(str(df.iat[r, c])))
        self.table_view.setModel(model)
        self.table_view.resizeColumnsToContents()
        self.table_view.horizontalHeader().setStretchLastSection(True)


class ApplyPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 4 – Apply & finish")
        self.setFinalPage(True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Click <b>Finish</b> to apply the tolerances and save updated XMLs."))

    def validatePage(self) -> bool:
        wiz = self.wizard()
        total_files = len(wiz.xml_paths)
        progress = QProgressDialog("Updating XML files…", "Cancel", 0, total_files, self)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)

        changelog: Dict[pathlib.Path, List[str]] = {}
        failures: List[tuple[pathlib.Path, str]] = []

        for i, xml_path in enumerate(wiz.xml_paths, 1):
            progress.setValue(i - 1)
            progress.setLabelText(f"Processing {xml_path.name} ({i}/{total_files})")
            QApplication.processEvents()
            try:
                new_path, changed = update_xml(
                    xml_path,
                    wiz.ref_col,
                    wiz.tolp_col,
                    wiz.toln_col,
                    wiz.df,
                )
                changelog[new_path] = changed
            except Exception as e:
                failures.append((xml_path, str(e)))
            if progress.wasCanceled():
                break
        progress.setValue(total_files)

        # Build summary message
        summary: List[str] = []
        for new_path, comps in changelog.items():
            summary.append(f"\n✔ {new_path.name}: {len(comps)} component(s) updated")
            if comps:
                summary.extend(f"   • {ref}" for ref in sorted(comps))
        if failures:
            summary.append("\nFailures:")
            summary.extend(f"✖ {p.name}: {err}" for p, err in failures)
        if not summary:
            summary = ["No files were processed."]

        QMessageBox.information(self, "Finished", "\n".join(summary))
        return True

###############################################################################
# Wizard driver & entry‑point
###############################################################################

class ToleranceWizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tolerance Updater Wizard")
        self.addPage(FileSelectPage())
        self.addPage(ColumnMapPage())
        self.addPage(PreviewPage())
        self.addPage(ApplyPage())
        self.resize(900, 600)


def main():
    app = QApplication(sys.argv)
    wiz = ToleranceWizard()
    wiz.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
