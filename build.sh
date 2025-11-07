#!/usr/bin/env bash
set -euo pipefail

# Build the ToleranceWizard executable (icon included) via PyInstaller.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT/tolerance_gui.py"
SPEC_FILE="$ROOT/ToleranceWizard.spec"
ICON_PATH="$ROOT/icons/tolerance_icon.ico"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Cannot find tolerance_gui.py at $SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$ICON_PATH" ]]; then
  echo "Cannot find icon at $ICON_PATH" >&2
  exit 1
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Cannot find PyInstaller spec at $SPEC_FILE" >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
else
  PYTHON="python"
fi

"$PYTHON" -m PyInstaller --noconfirm --clean "$SPEC_FILE"

