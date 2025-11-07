from __future__ import annotations
import sys
import pathlib
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import xml.etree.ElementTree as ET

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication, QWizard, QWizardPage, QFileDialog, QVBoxLayout, QLabel, QPushButton,
        QListWidget, QComboBox, QTableView, QMessageBox, QProgressDialog, QHBoxLayout,
        QCheckBox
    )
    from PyQt6.QtGui import QStandardItemModel, QStandardItem
except Exception:  # pragma: no cover - allows import without PyQt installed
    Qt = type('Qt', (), {'WindowModality': type('WM', (), {'ApplicationModal': None})})

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Dummy()

        def __call__(self, *a, **k):
            return _Dummy()

    QApplication = QWizard = QWizardPage = QFileDialog = QVBoxLayout = QLabel = QPushButton = (
        QListWidget
    ) = QComboBox = QTableView = QMessageBox = QProgressDialog = QHBoxLayout = QCheckBox = _Dummy
    QStandardItemModel = QStandardItem = _Dummy

from rules_profiles import compute_new_tolerance_pct_for_ref, PROFILE_SETS

RULE_NONE  = 'None (Use BOM TOLs)'

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


def parse_numeric_value(value: object, unit: str = '') -> Optional[float]:
    """Parse a numeric value from XML preserving scientific notation.

    ``value`` is typically a string from a ``Value`` attribute.  If it can be
    interpreted as a float the numeric value is returned; otherwise ``None`` is
    returned and the caller may keep the original string.  The optional ``unit``
    parameter is accepted for future use but currently does not affect the
    result.
    """

    s = str(value).strip()
    if s == '' or s.lower() == 'nan':
        return None
    try:
        return float(s)
    except Exception:
        return None

_range_re = re.compile(r'^([A-Za-z]+)(\d+)\s*-\s*(?:([A-Za-z]+)?(\d+))$')


def build_df_from_xml(xml_paths: List[pathlib.Path]) -> pd.DataFrame:
    """Build a BOM-like DataFrame from one or more XML component files."""

    rows: List[Dict[str, object]] = []
    for path in xml_paths:
        try:
            tree = ET.parse(path)
        except Exception:
            continue
        root = tree.getroot()
        for comp in root.iter('Component'):
            ref = comp.get('Name', '').strip()
            tolp = comp.get('TolP', '')
            toln = comp.get('TolN', '')
            val = None
            for prm in comp.iter('Parameter'):
                if prm.get('Name') == 'Value':
                    raw_val = prm.get('Value')
                    unit = prm.get('Unit', '') if prm is not None else ''
                    parsed = parse_numeric_value(raw_val, unit)
                    val = parsed if parsed is not None else raw_val
                    break
            rows.append({'Ref': ref, 'Value': val, 'TolP': tolp, 'TolN': toln})
    return pd.DataFrame(rows)

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

        # Ensure attributes exist before any slot runs
        self.xml_paths = []          # list[pathlib.Path]
        self.bom_path = None         # Optional[pathlib.Path]

        layout = QVBoxLayout(self)

        self.xml_list = QListWidget()
        btn_add_xml = QPushButton('Add XML files…')
        btn_add_xml.clicked.connect(self.add_xml_files)

        self.use_xml_bom = QCheckBox('Use XML files as BOM')
        self.use_xml_bom.stateChanged.connect(self._toggle_xml_bom)

        self.bom_label = QLabel('<i>No BOM selected</i>')
        self.btn_bom = QPushButton('Choose BOM file…')
        self.btn_bom.clicked.connect(self.choose_bom_file)

        rule_row = QHBoxLayout()
        rule_row.addWidget(QLabel('Profile:'))
        self.rule_combo = QComboBox()
        self.refresh_profiles()
        rule_row.addWidget(self.rule_combo)

        self.threshold_note = QLabel('MABAT: auto-applies R/C/L rules; resistor threshold is fixed at 0.1%')
        self.threshold_note.setStyleSheet('color: gray;')

        self.btn_profile_constructor = QPushButton('Profile Constructor…')
        self.btn_profile_constructor.clicked.connect(self.open_profile_constructor)

        layout.addLayout(rule_row)
        layout.addWidget(self.threshold_note)
        layout.addWidget(self.btn_profile_constructor)
        layout.addWidget(btn_add_xml)
        layout.addWidget(self.xml_list)
        layout.addWidget(self.use_xml_bom)
        layout.addWidget(self.btn_bom)
        layout.addWidget(self.bom_label)

        self.rule_combo.currentTextChanged.connect(self._rule_changed)
        self._rule_changed(self.rule_combo.currentText())

        # NOTE: removed reference to non-existent self.path_edit
        # (no connection here)


    def _on_path_changed(self, _=None) -> None:
        """Notify the wizard that page completeness may have changed."""
        self.completeChanged.emit()

    def refresh_profiles(self):
        self.rule_combo.clear()
        self.rule_combo.addItem(RULE_NONE)
        self.rule_combo.addItems(sorted(PROFILE_SETS.keys()))

    def _rule_changed(self, txt: str):
        self.threshold_note.setVisible(txt == 'MABAT')

    def _toggle_xml_bom(self):
        use_xml = self.use_xml_bom.isChecked()
        self.btn_bom.setEnabled(not use_xml)
        self.bom_label.setEnabled(not use_xml)
        self.completeChanged.emit()

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

    def open_profile_constructor(self):
        try:
            from profile_constructor_gui import ProfileConstructorWindow
        except Exception as e:  # pragma: no cover - import fallback
            QMessageBox.critical(self, 'Error', f'Profile constructor GUI not available: {e}')
            return
        self._constructor_win = ProfileConstructorWindow()
        try:
            self._constructor_win.profile_saved.connect(self.refresh_profiles)
        except Exception:
            pass
        try:
            self._constructor_win.destroyed.connect(self.refresh_profiles)
        except Exception:
            pass
        self._constructor_win.show()

    def initializePage(self):
        self.refresh_profiles()
        self.xml_paths = []
        self.bom_path = None
        self.use_xml_bom.setChecked(False)
        self.bom_label.setText('<i>No BOM selected</i>')
        self.btn_bom.setEnabled(True)
        self.bom_label.setEnabled(True)


    def isComplete(self) -> bool:
        """
        True if:
        - 'Use XML files as BOM' is checked AND at least one existing XML path is selected, OR
        - It is unchecked AND a valid BOM file has been chosen.
        """
        if self.use_xml_bom.isChecked():
            return bool(self.xml_paths) and all(getattr(p, "exists", lambda: False)() for p in self.xml_paths)
        return bool(self.bom_path and getattr(self.bom_path, "exists", lambda: False)())

    def validatePage(self) -> bool:
        wiz = self.wizard()
        try:
            if self.use_xml_bom.isChecked():
                df = build_df_from_xml(self.xml_paths)
                wiz.df_raw = df
                wiz.using_xml_bom = True
            else:
                wiz.df_raw = load_bom(self.bom_path)
                wiz.using_xml_bom = False
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load data: {e}')
            return False
        wiz.xml_paths = self.xml_paths
        wiz.rule = self.rule_combo.currentText()
        wiz.fixed_res_threshold = 0.1  # percent; fixed for MABAT resistors
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
        self.value_label = QLabel('Value column (required for selected profile):')
        layout.addWidget(self.value_label)
        layout.addWidget(self.combo_val)
        layout.addWidget(QLabel('Positive tolerance column (% number):'))
        layout.addWidget(self.combo_pos)
        layout.addWidget(QLabel('Negative tolerance column (% number, can match Pos):'))
        layout.addWidget(self.combo_neg)

    def initializePage(self):
        wiz = self.wizard()
        cols = list(wiz.df_raw.columns)
        for cb in (self.combo_ref, self.combo_val, self.combo_pos, self.combo_neg):
            cb.clear(); cb.addItems(cols)
        if getattr(wiz, 'using_xml_bom', False):
            for cb, col in zip(
                (self.combo_ref, self.combo_val, self.combo_pos, self.combo_neg),
                ('Ref', 'Value', 'TolP', 'TolN'),
            ):
                cb.setCurrentText(col)
                cb.setEnabled(False)
        needs_value = wiz.rule != RULE_NONE
        self.value_label.setEnabled(needs_value)
        self.combo_val.setEnabled(needs_value and not getattr(wiz, 'using_xml_bom', False))

    def isComplete(self) -> bool:
        wiz = self.wizard()
        if getattr(wiz, 'using_xml_bom', False):
            return True
        has_basic = bool(self.combo_ref.currentText() and self.combo_pos.currentText())
        needs_value = wiz.rule != RULE_NONE
        return has_basic and (bool(self.combo_val.currentText()) if needs_value else True)

    def validatePage(self) -> bool:
        wiz = self.wizard()
        if getattr(wiz, 'using_xml_bom', False):
            wiz.ref_col = 'Ref'
            wiz.tolp_col = 'TolP'
            wiz.toln_col = 'TolN'
            wiz.val_col = 'Value' if wiz.rule != RULE_NONE else None
        else:
            wiz.ref_col = self.combo_ref.currentText()
            wiz.tolp_col = self.combo_pos.currentText()
            wiz.toln_col = self.combo_neg.currentText()
            wiz.val_col = self.combo_val.currentText() if wiz.rule != RULE_NONE else None

        df = wiz.df_raw.copy()
        rows = []
        for _, row in df.iterrows():
            refs = expand_ref_expr(row[wiz.ref_col])
            if not refs:
                continue
            for r in refs:
                rows.append({
                    'Ref': str(r).strip(),
                    'Value': row[wiz.val_col] if wiz.rule != RULE_NONE else None,
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

        # Composite profiles: choose rules by reference prefix
        errors = []
        new_rows = []
        for _, r in df_exp.iterrows():
            ref = r['Ref']
            val_raw = r['Value']
            parsed_val = parse_numeric_value(val_raw)
            val = parsed_val if parsed_val is not None else val_raw
            tolp = try_float(r['TolP'])
            toln = try_float(r['TolN'])
            if wiz.rule == 'ELOP':
                tolp = 0.0 if tolp is None else tolp
                toln = tolp if toln is None else toln
            elif tolp is None and toln is None:
                continue
            try:
                newp = (
                    compute_new_tolerance_pct_for_ref(wiz.rule, ref, val, tolp)
                    if tolp is not None
                    else None
                )
                newn = (
                    compute_new_tolerance_pct_for_ref(wiz.rule, ref, val, toln)
                    if toln is not None
                    else newp
                )
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
            QMessageBox.warning(self, 'No rows', 'No valid rows remained after applying the profile.')
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
