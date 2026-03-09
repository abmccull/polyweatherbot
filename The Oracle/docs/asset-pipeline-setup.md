# Asset Pipeline Setup

## Local Setup

1. Copy `.env.example` to `.env` if you have not already.
2. Set `MESHY_API_KEY` in `.env`.
3. Keep generated outputs under `output/art/`.

## Commands

- `pnpm art:library`
- `pnpm art:submit -- --provider meshy --wave core_slice --limit 5`
- `pnpm art:submit-pending -- --submit`
- `pnpm art:harvest -- --family building`
- `pnpm art:meshy help`
- `pnpm art:pixel help`
- `pnpm art:pixel buildings`
- `pnpm art:pixel-buildings`
- `pnpm art:pixel-flora`
- `pnpm art:pixel-props`
- `pnpm art:pixel-terrain`
- `pnpm art:pixel-walkers`
- `pnpm art:rename -- --jsonl <batch.jsonl> --dir <output-dir>`
- see `docs/blender-mcp-integration.md` for the Blender bridge and render stage

## Typical Flow

1. Rebuild the tracked asset catalog with `pnpm art:library`.
2. Confirm the style direction against `docs/art-style-bible.md`.
3. Dry-run or submit a wave with `pnpm art:submit`.
4. Bulk-submit any still-missing Meshy assets with `pnpm art:submit-pending -- --submit`.
5. Poll and download completed submissions with `pnpm art:harvest`.
6. Import the proxy into Blender MCP, normalize scale/origin, and render a fixed-camera source image.
7. If Meshy credits or Blender add-on generators are unavailable, switch to the manual Blender fallback:
   - open `output/art/blender-templates/oracle_isometric_template.blend`
   - run `scripts/art/blender/manual-blockout-asset.py`
   - render one asset per Blender MCP call
8. For the tracked building set, batch-export the final browser-ready PNGs with `pnpm art:pixel buildings`.
9. Use `pnpm art:pixel convert ...` only for one-off tests or non-building assets that do not yet have a dedicated batch exporter.
10. Review the result in the game preview scenes and the public runtime asset folder.
11. Rename generated outputs back to asset IDs with `pnpm art:rename` only when working from ad hoc generator batches that lost their tracked naming.
12. For buildables, produce era variants from the same scene setup using the asset's `eraProfile`.

## Meshy Examples

### Text to 3D Preview

```bash
pnpm art:meshy text-preview \
  --asset-id temple-of-apollo \
  --prompt "ancient greek hilltop temple, marble, bronze trim, isometric game asset, clean silhouette"
```

### Check Status

```bash
pnpm art:meshy status \
  --kind text-to-3d \
  --task-id YOUR_TASK_ID
```

### Download a GLB

```bash
pnpm art:meshy download \
  --kind text-to-3d \
  --task-id YOUR_TASK_ID \
  --format glb
```

## Pixel Conversion Example

```bash
pnpm art:pixel convert \
  --input output/art/renders/temple-of-apollo.png \
  --output output/art/pixel/temple-of-apollo.png
```

## Building Batch Export

```bash
pnpm art:pixel buildings
```

This command now converts the tracked building render set into the current final usable browser format:

- `output/art/pixel/buildings/`
- `apps/web/public/assets/precinct/buildings/`
- `apps/web/public/assets/precinct/building-art-manifest.json`
- `packages/content/src/generated/buildingArt.ts`

## Terrain Batch Export

```bash
pnpm art:pixel-terrain
```

This command exports the current browser terrain runtime set into:

- `output/art/pixel/terrain/`
- `apps/web/public/assets/precinct/terrain/`
- `apps/web/public/assets/precinct/terrain-art-manifest.json`
- `packages/content/src/generated/terrainArt.ts`

## Flora Batch Export

```bash
pnpm art:pixel-flora
```

This command exports the current browser flora/perimeter runtime set into:

- `output/art/pixel/flora/`
- `apps/web/public/assets/precinct/flora/`
- `apps/web/public/assets/precinct/flora-art-manifest.json`
- `packages/content/src/generated/floraArt.ts`

## Prop Batch Export

```bash
pnpm art:pixel-props
```

This command exports the current browser prop runtime set into:

- `output/art/pixel/props/`
- `apps/web/public/assets/precinct/props/`
- `apps/web/public/assets/precinct/prop-art-manifest.json`
- `packages/content/src/generated/propArt.ts`

The current prop runtime set now includes `13` exported assets, including the local Blender-fallback `purification_font`.

## Walker Batch Export

```bash
pnpm art:pixel-walkers
```

This command exports the current browser walker runtime set into:

- `output/art/pixel/walkers/`
- `apps/web/public/assets/precinct/walkers/`
- `apps/web/public/assets/precinct/walker-art-manifest.json`
- `packages/content/src/generated/walkerArt.ts`

## Notes

- Meshy credentials must stay local.
- The tracked asset catalog now lives under `art-library/`.
- The art direction source-of-truth lives in `docs/art-style-bible.md`.
- Building assets now carry an `eraProfile` so the same sanctuary can visually evolve over time.
- The pixel conversion command uses `proper-pixel-art` through `uvx`.
- The current runtime-ready export path now covers buildings, terrain accents, flora, walkers, and a first prop/clutter set; fauna, portraits, and UI packs still need matching runtime exporters.
- This setup intentionally separates concept generation, 3D proxy generation, and final cleanup.
- Blender can now be used as a fully local fallback for manual blockout assets when external generation is blocked.
- The manual Blender fallback now covers the full tracked building family except the three Meshy-first imports already rendered through the import helper.
