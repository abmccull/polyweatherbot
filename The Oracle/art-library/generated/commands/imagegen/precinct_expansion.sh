#!/usr/bin/env bash
set -euo pipefail
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export IMAGE_GEN="$CODEX_HOME/skills/imagegen/scripts/image_gen.py"
python "$IMAGE_GEN" generate-batch --input 'art-library/generated/prompt-packs/imagegen/precinct_expansion.jsonl' --out-dir 'output/art/imagegen/precinct_expansion'
