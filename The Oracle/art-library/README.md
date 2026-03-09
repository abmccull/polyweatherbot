# The Oracle Asset Library

This folder is the tracked source-of-truth for the game's asset-production catalog.

## Structure

- `source/catalog.mjs` — editable catalog of every tracked asset family and prompt seed.
- `generated/asset-manifest.json` — machine-readable manifest.
- `generated/asset-manifest.csv` — spreadsheet-friendly manifest export.
- `generated/summary.md` — counts by family, wave, and provider lane.
- `generated/prompt-packs/` — batch-ready prompt packs for Meshy and OpenAI image generation.
- `generated/commands/` — generated shell scripts for running those prompt packs.

## Commands

- `pnpm art:library` — rebuild the generated manifest and prompt packs.
- `pnpm art:submit -- --provider meshy --wave core_slice --limit 5` — dry-run the first Meshy submissions.
- `pnpm art:submit -- --provider meshy --wave core_slice --limit 5 --submit` — actually submit those Meshy jobs.

## Notes

- The catalog currently covers buildings, props, terrain, flora, fauna, characters, portraits, and UI/world-map packs.
- Buildable sanctuary assets can now declare an `eraProfile` so the same footprint can ship with era-specific visual variants.
- `OPENAI_API_KEY` is still required before any OpenAI image generation batch can be executed.
- Meshy and pixel conversion are useful for geometry consistency and cleanup-friendly base assets, but final production sprites still need art review and hand cleanup.
