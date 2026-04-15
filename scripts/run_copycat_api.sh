#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec python3 -m uvicorn api.server:app --host 0.0.0.0 --port "${API_PORT:-8000}"
