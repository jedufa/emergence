#!/usr/bin/env bash
# =====================================================================
# tools/docs/build_docs.sh
#
# Purpose:
#   Build the MkDocs site for this repo into ./site with one command.
#
# Why this exists:
#   Humans keep moving tools around, and mkdocstrings then fails to import
#   modules unless PYTHONPATH is set correctly and the right Python is used.
#
# Usage:
#   ./tools/docs/build_docs.sh
#
# Options:
#   --serve     Start local dev server (mkdocs serve)
#   --open      Attempt to open the built site in the Windows browser
# =====================================================================

set -euo pipefail

want_serve="0"
want_open="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve) want_serve="1"; shift 1;;
    --open)  want_open="1"; shift 1;;
    -h|--help)
      sed -n '1,140p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

has_cmd() { command -v "$1" >/dev/null 2>&1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------
# Pick a Python that actually has your deps.
# Priority:
#   1) currently activated venv (python on PATH)
#   2) user-level venv at ~/venvs/emergence (your new canonical location)
#   3) system python fallback
# ---------------------------------------------------------------------
PY_BIN="python3"
if has_cmd python; then
  PY_BIN="python"
fi

if [[ -x "$HOME/venvs/emergence/bin/python" ]]; then
  # Only switch if the current python is NOT coming from an active venv.
  # (If you're already activated, leave it alone.)
  CUR_PY="$("$PY_BIN" -c 'import sys; print(sys.prefix)' 2>/dev/null || echo "")"
  if [[ -z "$CUR_PY" ]] || [[ "$CUR_PY" == "/usr" ]] || [[ "$CUR_PY" == "/usr/local" ]]; then
    PY_BIN="$HOME/venvs/emergence/bin/python"
  fi
fi

# mkdocstrings imports your repo code. Make sure the repo root is importable.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Install deps only if missing (keeps rebuilds snappy and avoids pip spam).
if ! "$PY_BIN" -c "import mkdocs, mkdocs_material, mkdocstrings, pymdownx" >/dev/null 2>&1; then
  "$PY_BIN" -m pip install -q --disable-pip-version-check mkdocs-material "mkdocstrings[python]" pymdown-extensions
fi


# ---------------------------------------------------------------------
# Auto-generate CLI argument reference (keeps docs honest)
# ---------------------------------------------------------------------
# IMPORTANT: MkDocs only includes files under docs_dir (./docs).
# Write generated outputs there so they render in the site.
GEN_OUT="$ROOT/docs/generated/cli_reference.md"
if "$PY_BIN" -m tools.docs.generate_cli_reference --out "$GEN_OUT" --python "$PY_BIN" >/dev/null 2>&1; then
  echo "[docs] Wrote CLI reference: $GEN_OUT"
else
  echo "[docs] WARNING: CLI reference generation failed (continuing)." >&2
fi

# ---------------------------------------------------------------------
# Auto-generate Simulation Quick Start (MD + DOCX)
# ---------------------------------------------------------------------
QS_MD="$ROOT/docs/generated/quick_start.md"
QS_DOCX="$ROOT/docs/generated/quick_start.docx"
if "$PY_BIN" -m tools.docs.generate_quick_start --out_md "$QS_MD" --out_docx "$QS_DOCX" --python "$PY_BIN" >/dev/null 2>&1; then
  echo "[docs] Wrote Quick Start: $QS_MD"
else
  echo "[docs] WARNING: Quick Start generation failed (continuing)." >&2
fi


# ---------------------------------------------------------------------
# Auto-generate harness + tooling pages (tests + helper scripts)
# ---------------------------------------------------------------------
H_OUT="$ROOT/docs/generated"
if "$PY_BIN" -m tools.docs.generate_harness_pages --repo_root "$ROOT" --out_dir "$H_OUT" >/dev/null 2>&1; then
  echo "[docs] Wrote harness pages under: $H_OUT"
else
  echo "[docs] WARNING: Harness pages generation failed (continuing)." >&2
fi


if [[ "$want_serve" == "1" ]]; then
  exec "$PY_BIN" -m mkdocs serve
fi

"$PY_BIN" -m mkdocs build --clean

echo ""
echo "Docs built: $ROOT/site/index.html"

if [[ "$want_open" == "1" ]]; then
  if has_cmd wslview; then
    wslview "$ROOT/site/index.html" >/dev/null 2>&1 || true
  else
    echo "NOTE: wslview not found. Install 'wslu' to enable --open."
  fi
fi
