from __future__ import annotations
import sys
import pathlib
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import xml.etree.ElementTree as ET

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QWizard, QWizardPage, QFileDialog, QVBoxLayout, QLabel, QPushButton,
    QListWidget, QComboBox, QTableView, QMessageBox, QProgressDialog, QHBoxLayout
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem

from rules_profiles import compute_new_tolerance_pct

RULE_NONE  = 'None (Use BOM TOLs)'
RULE_MABAT = 'MABAT (R/C/L)'

def load_bom(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.csv', '.txt'}:
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=0, engine='openpyxl')

def str_is_nonempty(x: object) -> bool:
    s = str(x).strip()
    return s != '' and s.lower() != 'nan'

def try_float(x: object) -> Optional[float]:
    s = str(x).strip()
    if s == '' or s.lower() == 'nan':
        return None
    try:
        return float(s)
    except Exception:
        return None

_range_re = re.compile(r'^([A-Za-z]+)(\d+)\s*-\s*(?:([A-Za-z]+)?(\d+))$')

def expand_ref_expr(expr: object) -> List[str]:
    """Accepts 'R1,R3,R7' and ranges 'R1-R10'/'R1-10'."""
    if expr is None:
        return []
    text = str(expr).strip()
    if text == '':
        return []
    parts = [p.strip() for p in text.split(',')]
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        m = _range_re.match(p)
        if m:
            pfx1, start, pfx2, end = m.groups()
            if pfx2 and pfx2 != pfx1:
                out.append(p)  # keep literal if prefixes disagree
                continue
            a = int(start); b = int(end)
            rng = range(a, b+1) if a <= b else range(a, b-1, -1)
            out.extend([f'{pfx1}{i}' for i in rng])
        else:
            out.append(p)
    return out

def update_xml_with_map(xml_path: pathlib.Path, map_pos: Dict[str, float], map_neg: Dict[str, float]) -> Tuple[pathlib.Path, List[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    updated: List[str] = []
    for comp in root.iter('Component'):
        name = comp.get('Name', '').strip()
        if name in map_pos:
            pos = str(map_pos[name]).rstrip('0').rstrip('.') if isinstance(map_pos[name], float) else str(map_pos[name])
            negv = map_neg.get(name, map_pos[name])
            neg = str(negv).rstrip('0').rstrip('.') if isinstance(negv, float) else str(negv)
            comp.set('TolP', pos)
            comp.set('TolN', neg)
            for prm in comp.iter('Parameter'):
                if prm.get('Name') == 'TolPos':
                    prm.set('Value', pos)
                elif prm.get('Name') == 'TolNeg':
                    prm.set('Value', neg)
            updated.append(name)
    out = xml_path.with_name(xml_path.stem + '_updated' + xml_path.suffix)
    tree.write(out, encoding='Windows-1252', xml_declaration=True)
    return out, updated

# ----------------------------- Wizard Pages ----------------------------- #
class FileSelectPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle('Step 1 – Select files and profile')
        layout = QVBoxLayout(self)

        self.xml_list = QListWidget()
        btn_add_xml = QPushButton('Add XML files…')
        btn_add_xml.clicked.connect(self.add_xml_files)

        self.bom_label = QLabel('<i>No BOM selected</i>')
        btn_bom = QPushButton('Choose BOM file…')
        btn_bom.clicked.connect(self.choose_bom_file)

        rule_row = QHBoxLayout()
        rule_row.addWidget(QLabel('Profile:'))
        self.rule_combo = QComboBox()
        self.rule_combo.addItems([RULE_NONE, RULE_MABAT])
        rule_row.addWidget(self.rule_combo)

        self.threshold_note = QLabel('MABAT: auto-applies R/C/L rules; resistor threshold is fixed at 0.1%')
        self.threshold_note.setStyleSheet('color: gray;')

        layout.addLayout(rule_row)
        layout.addWidget(self.threshold_note)
        layout.addWidget(btn_add_xml)
        layout.addWidget(self.xml_list)
        layout.addWidget(btn_bom)
        layout.addWidget(self.bom_label)

        self.rule_combo.currentTextChanged.connect(self._rule_changed)
        self._rule_changed(self.rule_combo.currentText())

    def _rule_changed(self, txt: str):
        self.threshold_note.setVisible(txt == RULE_MABAT)

    def add_xml_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Pick XML component files', '', 'XML files (*.xml)')
        for f in files:
            p = pathlib.Path(f)
            if p.suffix.lower() == '.xml' and p not in self.xml_paths:
                self.xml_paths.append(p)
                self.xml_list.addItem(str(p))
        self.completeChanged.emit()

    def choose_bom_file(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Pick BOM file (CSV / Excel)', '', 'BOM files (*.csv *.xlsx *.xlsm *.xls)')
        if f:
            self.bom_path = pathlib.Path(f)
            self.bom_label.setText(str(self.bom_path))
            self.completeChanged.emit()

    def initializePage(self):
        self.xml_paths = []
        self.bom_path = None

    def isComplete(self) -> bool:
        return bool(self.xml_paths and self.bom_path)

    def validatePage(self) -> bool:
        try:
            self.wizard().df_raw = load_bom(self.bom_path)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load BOM: {e}')
            return False
        self.wizard().xml_paths = self.xml_paths
        self.wizard().rule = self.rule_combo.currentText()
        self.wizard().fixed_res_threshold = 0.1  # percent; fixed for MABAT resistors
        return True

class ColumnMapPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle('Step 2 – Map BOM columns')
        layout = QVBoxLayout(self)
        self.combo_ref = QComboBox()
        self.combo_val = QComboBox()
        self.combo_pos = QComboBox()
        self.combo_neg = QComboBox()

        layout.addWidget(QLabel('Reference column (supports lists and ranges):'))
        layout.addWidget(self.combo_ref)
        self.value_label = QLabel('Value column (required for MABAT):')
        layout.addWidget(self.value_label)
        layout.addWidget(self.combo_val)
        layout.addWidget(QLabel('Positive tolerance column (% number):'))
        layout.addWidget(self.combo_pos)
        layout.addWidget(QLabel('Negative tolerance column (% number, can match Pos):'))
        layout.addWidget(self.combo_neg)

    def initializePage(self):
        cols = list(self.wizard().df_raw.columns)
        for cb in (self.combo_ref, self.combo_val, self.combo_pos, self.combo_neg):
            cb.clear(); cb.addItems(cols)
        needs_value = self.wizard().rule == RULE_MABAT
        self.value_label.setEnabled(needs_value)
        self.combo_val.setEnabled(needs_value)

    def isComplete(self) -> bool:
        has_basic = bool(self.combo_ref.currentText() and self.combo_pos.currentText())
        needs_value = self.wizard().rule == RULE_MABAT
        return has_basic and (bool(self.combo_val.currentText()) if needs_value else True)

    def _pick_subprofile_for_ref(self, ref: str) -> tuple[str, Optional[float]]:
        prefix = str(ref).strip().upper()[:1]
        if prefix == 'R':
            return 'MABAT (Resistors)', 0.1
        if prefix == 'C':
            return 'Capacitor (F)', None
        if prefix == 'L':
            return 'Inductor (H)', None
        raise ValueError(f"Unsupported reference prefix for MABAT: '{ref}'")

    def validatePage(self) -> bool:
        wiz = self.wizard()
        wiz.ref_col = self.combo_ref.currentText()
        wiz.tolp_col = self.combo_pos.currentText()
        wiz.toln_col = self.combo_neg.currentText()
        wiz.val_col = self.combo_val.currentText() if wiz.rule == RULE_MABAT else None

        df = wiz.df_raw.copy()
        rows = []
        for _, row in df.iterrows():
            refs = expand_ref_expr(row[wiz.ref_col])
            if not refs:
                continue
            for r in refs:
                rows.append({
                    'Ref': str(r).strip(),
                    'Value': row[wiz.val_col] if wiz.rule == RULE_MABAT else None,
                    'TolP': row[wiz.tolp_col],
                    'TolN': row[wiz.toln_col],
                })
        df_exp = pd.DataFrame(rows)

        if wiz.rule == RULE_NONE:
            df_exp = df_exp[df_exp['TolP'].apply(str_is_nonempty)].copy()
            if df_exp.empty:
                QMessageBox.warning(self, 'No rows', 'No rows with non-blank positive tolerance were found.')
                return False
            df_exp['NewTolP'] = df_exp['TolP'].astype(str).str.strip()
            df_exp['NewTolN'] = df_exp['TolN'].astype(str).str.strip()
            null_neg = df_exp['NewTolN'].eq('') | df_exp['NewTolN'].str.lower().eq('nan')
            df_exp.loc[null_neg, 'NewTolN'] = df_exp.loc[null_neg, 'NewTolP']
            wiz.df_prepared = df_exp[['Ref', 'NewTolP', 'NewTolN']].copy()
            return True

        # MABAT composite: choose rules by reference prefix
        errors = []
        new_rows = []
        for _, r in df_exp.iterrows():
            ref = r['Ref']
            val = r['Value']
            tolp = try_float(r['TolP'])
            toln = try_float(r['TolN'])
            if tolp is None and toln is None:
                continue
            try:
                profile_name, thr = self._pick_subprofile_for_ref(ref)
                newp = compute_new_tolerance_pct(profile_name, val, tolp, thr) if tolp is not None else None
                newn = compute_new_tolerance_pct(profile_name, val, toln, thr) if toln is not None else newp
                if newp is None and newn is None:
                    continue
                new_rows.append({'Ref': ref, 'NewTolP': newp, 'NewTolN': newn})
            except Exception as e:
                errors.append(f"{ref}: {e}")

        if errors:
            QMessageBox.critical(self, 'Value errors',
                                 'Some rows could not be processed:\n- ' +
                                 '\n- '.join(errors[:30]) +
                                 ('' if len(errors) <= 30 else f"\n(+{len(errors)-30} more)"))
            return False

        wiz.df_prepared = pd.DataFrame(new_rows)
        if wiz.df_prepared.empty:
            QMessageBox.warning(self, 'No rows', 'No valid rows remained after applying MABAT.')
            return False
        return True

class PreviewPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle('Step 3 – Preview all changes')
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('This table lists every component that will be updated and the *new* tolerances (%):'))
        self.table = QTableView()
        layout.addWidget(self.table)

    def initializePage(self):
        wiz = self.wizard()
        df = wiz.df_prepared.copy()
        for col in ('NewTolP', 'NewTolN'):
            df[col] = df[col].apply(lambda x: '' if x is None or str(x).strip()=='' else str(float(x)).rstrip('0').rstrip('.'))
        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels(df.columns.tolist())
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                model.setItem(r, c, QStandardItem(str(df.iat[r, c])))
        self.table.setModel(model)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

class ApplyPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle('Step 4 – Apply & finish')
        self.setFinalPage(True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Click Finish to write the updated XML files.'))

    def validatePage(self) -> bool:
        wiz = self.wizard()
        df = wiz.df_prepared.copy()
        map_pos: Dict[str, float] = {}
        map_neg: Dict[str, float] = {}
        for _, r in df.iterrows():
            ref = str(r['Ref']).strip()
            pos = r['NewTolP']
            neg = r['NewTolN'] if r['NewTolN'] is not None else r['NewTolP']
            if str_is_nonempty(pos):
                map_pos[ref] = float(pos)
                map_neg[ref] = float(neg if neg is not None else pos)

        pbar = QProgressDialog('Updating XML files…', 'Cancel', 0, len(wiz.xml_paths), self)
        pbar.setWindowModality(Qt.WindowModality.ApplicationModal)

        report, fails = {}, []
        for i, xml in enumerate(wiz.xml_paths, 1):
            pbar.setValue(i-1)
            pbar.setLabelText(f'Processing {xml.name} ({i}/{len(wiz.xml_paths)})')
            QApplication.processEvents()
            try:
                new_path, changed = update_xml_with_map(xml, map_pos, map_neg)
                report[new_path] = changed
            except Exception as e:
                fails.append((xml, str(e)))
            if pbar.wasCanceled():
                break
        pbar.setValue(len(wiz.xml_paths))

        lines = []
        for new, comps in report.items():
            lines.append(f'✔ {new.name}: {len(comps)} updated')
            lines.extend(f'   • {c}' for c in sorted(comps))
        for xml, err in fails:
            lines.append(f'✖ {xml.name}: {err}')
        if not lines:
            lines.append('No files processed.')
        QMessageBox.information(self, 'Done', '\n'.join(lines))
        return True

class ToleranceWizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Tolerance Updater Wizard')
        self.addPage(FileSelectPage())
        self.addPage(ColumnMapPage())
        self.addPage(PreviewPage())
        self.addPage(ApplyPage())
        self.resize(1000, 650)

def main():
    app = QApplication(sys.argv)
    w = ToleranceWizard()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
