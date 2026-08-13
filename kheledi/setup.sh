#!/usr/bin/env bash
# Build this directory's own virtualenv, so nothing here can disturb a working
# install. Everything kheledi/ needs lives under kheledi/, and deleting the
# directory removes every trace of it.
#
#   bash setup.sh            # pin 1.20.1, the version the packages ship
#   bash setup.sh 1.28.0     # a second env, to compare runtimes side by side
#   PYTHON=python3.11 bash setup.sh 1.20.1
#
# Each version gets its own venv (.venv-<version>), so both can exist at once
# and neither replaces the other.
#
# Interpreter choice is not cosmetic. onnxruntime 1.20.1 has no wheel for
# Python 3.13 or newer, so on a machine whose `python3` is 3.14 the pinned
# version simply cannot be installed and pip reports the oldest available as
# 1.24.1. Python 3.12 is the newest interpreter with wheels across the whole
# range this directory compares, so it is preferred when present.
set -euo pipefail
cd "$(dirname "$0")"

ORT_VERSION="${1:-1.20.1}"
VENV=".venv-${ORT_VERSION}"

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
else
  PY=python3
fi
echo "  interpreter  $PY ($($PY --version 2>&1))"

if [ ! -d "$VENV" ]; then
  echo "  creating $VENV"
  "$PY" -m venv "$VENV"
fi

echo "  installing onnxruntime==${ORT_VERSION}"
"$VENV/bin/pip" -q install --upgrade pip
"$VENV/bin/pip" -q install "onnxruntime==${ORT_VERSION}" numpy psutil

echo
"$VENV/bin/python3" - <<'PY'
import onnxruntime as ort, numpy, platform
print(f"  ready: onnxruntime {ort.__version__} · numpy {numpy.__version__} · "
      f"{platform.system()} {platform.machine()}")
PY
echo
echo "  run it:  $VENV/bin/python3 2_bench.py"
