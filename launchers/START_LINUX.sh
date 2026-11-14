#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements-local.txt
    .venv/bin/python -m pip install --no-deps wheels/quantstream_local-0.3.0-py3-none-any.whl
fi
exec .venv/bin/python run_local.py
