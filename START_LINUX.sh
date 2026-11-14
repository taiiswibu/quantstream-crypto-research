#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r launchers/requirements-local.txt
    .venv/bin/python -m pip install --no-deps -e .
fi
if [ ! -f workspace/artifacts/local_registry.json ]; then
    .venv/bin/python -m quantstream.terminal --root workspace setup --days 90
fi
exec .venv/bin/python launchers/run_local.py
