# Restart Context

This file is a quick handoff for the next Codex session working on `The Oracle`.

## Repo + Goal

- Repo path: `/Users/tsc-001/station_sniper/The Oracle`
- Current focus: move from graybox/vertical-slice implementation toward a full art + asset production pipeline with consistent **ancient Greece / Delphi oracle** vibes.
- For the full end-state visual roadmap, use:
  - `docs/visual-finish-plan.md`

## What Was Completed

### Codebase / design audit

- The codebase was thoroughly compared against the provided GDD.
- Conclusion: the repo is a strong **vertical slice**, not a full implementation.
- Follow-up planning docs were created:
  - `docs/parallel-implementation-plan.md`
  - `docs/full-release-follow-on-plan.md`

### Asset pipeline

- Added a full tracked asset library and prompt catalog:
  - `art-library/source/catalog.mjs`
  - `art-library/generated/asset-manifest.json`
  - `art-library/generated/summary.md`
- Added art tooling:
  - `scripts/art/build-library.mjs`
  - `scripts/art/submit-library.mjs`
  - `scripts/art/meshy.mjs`
  - `scripts/art/pixel-art.mjs`
  - `scripts/art/rename-generated-batch.mjs`
- Added package commands:
  - `pnpm art:library`
  - `pnpm art:submit`
  - `pnpm art:meshy`
  - `pnpm art:pixel`
  - `pnpm art:rename`

### Art direction tuning

- Added and tuned style docs:
  - `docs/art-style-bible.md`
  - `docs/asset-pipeline-setup.md`
  - `docs/meshy-art-pipeline.md`
  - `docs/blender-mcp-integration.md`
- Prompt builders were retuned to avoid:
  - medieval / bishop-like silhouettes
  - glossy casual-mobile / toy-diorama style
  - oversized imperial megastructures
- Building prompts now emphasize **compact sanctuary-scale** forms.

### Era variation support

- Buildable assets now carry `eraProfile` metadata.
- Era rules are defined in `docs/art-style-bible.md` for:
  - `archaic`
  - `classical`
  - `hellenistic`
  - `romanized`
- The intention is gradual precinct evolution, not full reskin swaps.

## Generated / validated outputs

### Live deploy

- A live Vercel deployment now exists for the current built web client:
  - `https://dist-murex-pi.vercel.app`
- The successful deploy path used the already-built static output from:
  - `apps/web/dist`
- A direct Vite/workspace deploy attempt from `apps/web` uploaded, but the hosted build failed because Vercel fell back to `npm install` against the workspace layout.
- If a proper repo-native Vercel deployment is needed later, add explicit monorepo/Vercel config instead of relying on auto-detection.

### Image generation

- Strong tuned image batch exists in:
  - `output/art/imagegen/core_slice_tuned/`
- Additional probes / batches exist in:
  - `output/art/imagegen/style-pass-ancient-greece-1/`
  - `output/art/imagegen/style-pass-terrain-2/`
  - `output/art/imagegen/characters-core/`
  - `output/art/imagegen/portraits-core/`
  - `output/art/imagegen/core_slice/`
- Quality takeaway:
  - buildings, portraits, and character sheets are now much closer to the target vibe
  - terrain improved significantly after retuning and is more extraction-friendly

### Meshy

- Meshy pipeline is wired and working through `.env`.
- Multiple preview tasks were submitted earlier.
- One completed proxy model was downloaded to:
  - `output/art/meshy/downloads/text-to-3d/sacred_way_kit.glb`
- New bulk helpers were added at:
  - `scripts/art/submit-pending-meshy.mjs`
  - `scripts/art/harvest-meshy-ledger.mjs`
- Package commands now include:
  - `pnpm art:submit-pending`
  - `pnpm art:harvest`
- New preview tasks were submitted for the first real asset batch:
  - `castalian_spring` -> `019cc4de-ca12-7440-8b55-72b63bab6cd6`
  - `eternal_flame_brazier` -> `019cc4de-ccaf-7cf9-9059-5ba6e40fdf4f`
  - `temple_of_apollo` -> `019cc4de-d5b4-7aeb-9445-aa49b02dde88`
- Follow-up polling completed and all three preview tasks reached:
  - `SUCCEEDED`
- A bulk pending submission run accepted `47` additional Meshy tasks before stopping on:
  - `402 Insufficient funds`
- The accepted task map now lives in:
  - `output/art/meshy/submission-ledger.jsonl`
- The first harvest pass now lives in:
  - `output/art/meshy/harvest-ledger.jsonl`
- Downloaded GLBs now exist at:
  - `output/art/meshy/downloads/text-to-3d/castalian_spring.glb`
  - `output/art/meshy/downloads/text-to-3d/eternal_flame_brazier.glb`
  - `output/art/meshy/downloads/text-to-3d/temple_of_apollo.glb`
  - `output/art/meshy/downloads/text-to-3d/gatehouse_entrance.glb`
  - `output/art/meshy/downloads/text-to-3d/purification_font.glb`
  - `output/art/meshy/downloads/text-to-3d/inner_sanctum.glb`
  - `output/art/meshy/downloads/text-to-3d/omphalos_stone.glb`
  - `output/art/meshy/downloads/text-to-3d/tholos.glb`
  - `output/art/meshy/downloads/text-to-3d/treasury_vault.glb`
  - `output/art/meshy/downloads/text-to-3d/priest_quarters.glb`
  - `output/art/meshy/downloads/text-to-3d/gymnasium.glb`
  - `output/art/meshy/downloads/text-to-3d/scriptorium.glb`
  - `output/art/meshy/downloads/text-to-3d/animal_pen.glb`
  - `output/art/meshy/downloads/text-to-3d/olive_press.glb`
  - `output/art/meshy/downloads/text-to-3d/incense_store.glb`
  - `output/art/meshy/downloads/text-to-3d/sacrificial_altar.glb`
  - `output/art/meshy/downloads/text-to-3d/temple_of_athena.glb`
  - `output/art/meshy/downloads/text-to-3d/temple_of_hermes.glb`
  - `output/art/meshy/downloads/text-to-3d/kitchen.glb`
  - `output/art/meshy/downloads/text-to-3d/storehouse.glb`
  - `output/art/meshy/downloads/text-to-3d/xenon.glb`
  - `output/art/meshy/downloads/text-to-3d/agora_market.glb`
- Additional prop downloads now exist at:
  - `output/art/meshy/downloads/text-to-3d/amphora_stack.glb`
  - `output/art/meshy/downloads/text-to-3d/grain_sacks.glb`
  - `output/art/meshy/downloads/text-to-3d/oil_jars.glb`
  - `output/art/meshy/downloads/text-to-3d/incense_censer.glb`
  - `output/art/meshy/downloads/text-to-3d/offering_bowl.glb`
  - `output/art/meshy/downloads/text-to-3d/laurel_wreath_set.glb`
  - `output/art/meshy/downloads/text-to-3d/scroll_shelves.glb`
  - `output/art/meshy/downloads/text-to-3d/writing_desk.glb`
  - `output/art/meshy/downloads/text-to-3d/market_stall_set.glb`
  - `output/art/meshy/downloads/text-to-3d/votive_statue_small.glb`
  - `output/art/meshy/downloads/text-to-3d/votive_statue_large.glb`
  - `output/art/meshy/downloads/text-to-3d/bronze_tripod.glb`
  - `output/art/meshy/downloads/text-to-3d/ritual_basin.glb`
  - `output/art/meshy/downloads/text-to-3d/stone_bench.glb`
  - `output/art/meshy/downloads/text-to-3d/scaffold_set.glb`
  - `output/art/meshy/downloads/text-to-3d/repair_tools.glb`
- Harvest ledger rows now recorded:
  - `44`
- Total downloaded Meshy GLBs now on disk:
  - `39`

### Blender MCP

- Blender MCP was installed as a **global** MCP setup, not project-local.
- Relevant global config locations:
  - `~/.mcp.json`
  - `~/Library/Application Support/Claude/claude_desktop_config.json`
- Codex global MCP server was added as `blender`.
- Blender add-on file staged at:
  - `~/Library/Application Support/Blender/5.0/scripts/addons/blender_mcp_addon.py`
  - `~/Downloads/blender_mcp_addon.py`
- Blender add-on was enabled and connected successfully.
- Blender panel showed:
  - `Disconnect from MCP server`
  - `Running on port 9876`

### Blender scene template

- The native Blender MCP tool surface is now usable in-session.
- A tracked bootstrap script was added at:
  - `scripts/art/blender/oracle-scene-template.py`
- A tracked Meshy import/render helper was added at:
  - `scripts/art/blender/render-meshy-asset.py`
- A tracked manual blockout helper was added at:
  - `scripts/art/blender/manual-blockout-asset.py`
- The first reusable scene template was generated and saved to:
  - `output/art/blender-templates/oracle_isometric_template.blend`
- Template metadata was written to:
  - `output/art/blender-templates/oracle_isometric_template.json`
- A validation render was written to:
  - `output/art/renders/template/oracle_template_probe.png`
- The template now includes:
  - fixed orthographic 3/4-isometric camera
  - fixed sun/shadow direction
  - transparent PNG render settings
  - sanctuary-safe material presets for limestone, marble, terracotta, and aged bronze
  - footprint guides for `1x1`, `2x2`, and `3x3`
  - a hidden reference shrine used for validation renders

### Manual Blender fallback outputs

- The Blender-side fallback path now works without any external generator.
- Current disabled add-on integrations inside Blender MCP:
  - Sketchfab
  - Polyhaven
  - Hyper3D
  - Hunyuan3D
- Because those integrations are disabled and Meshy credits ran out, the active fallback is:
  - open `output/art/blender-templates/oracle_isometric_template.blend`
  - run `scripts/art/blender/manual-blockout-asset.py` with one asset at a time
- Manual blockout renders now exist for the full tracked building family except the three imported Meshy-first assets:
  - `castalian_spring`
  - `eternal_flame_brazier`
  - `temple_of_apollo`
- Manual render count now on disk:
  - `32`
- Manual render directory:
  - `output/art/renders/manual/`
- Matching per-asset `.blend` and metadata `.json` files were saved under:
  - `output/art/blender-scenes/`
- The manual helper now covers:
  - sanctuary infrastructure
  - temples and monuments
  - utility, housing, and market buildings
  - world-ui support buildings
  - late-game excavation / vault structures
- Blender MCP note:
  - some single-asset runs still timed out at the tool layer even though the `.png`, `.blend`, and `.json` were written correctly; verify the filesystem before rerunning

### First imported asset outputs

- The first downloaded Meshy assets were imported into the template scene and rendered to transparent PNGs:
  - `output/art/renders/meshy/castalian_spring.png`
  - `output/art/renders/meshy/eternal_flame_brazier.png`
  - `output/art/renders/meshy/temple_of_apollo.png`
- Per-asset Blender scene files were saved to:
  - `output/art/blender-scenes/castalian_spring.blend`
  - `output/art/blender-scenes/eternal_flame_brazier.blend`
  - `output/art/blender-scenes/temple_of_apollo.blend`
- Per-asset normalization metadata was saved to:
  - `output/art/blender-scenes/castalian_spring.json`
  - `output/art/blender-scenes/eternal_flame_brazier.json`
  - `output/art/blender-scenes/temple_of_apollo.json`
- These are still first-pass proxy renders. The next cleanup pass should focus on:
  - material reassignment / texture cleanup
  - rotation/orientation tuning where silhouettes feel generic
  - `temple_of_apollo` `archaic` and `classical` variant staging

### Final usable browser-format assets

- The current final usable game format is now standardized for building art as:
  - transparent pixel PNGs under `apps/web/public/assets/precinct/buildings/`
  - mirrored review outputs under `output/art/pixel/buildings/`
  - a browser manifest at `apps/web/public/assets/precinct/building-art-manifest.json`
  - a typed code registry at `packages/content/src/generated/buildingArt.ts`
- The tracked building export path now runs through:
  - `scripts/art/export-building-pixel-assets.mjs`
  - `pnpm art:pixel buildings`
  - `pnpm art:pixel-buildings`
- Current converted runtime-ready count:
  - `39` building PNGs in `apps/web/public/assets/precinct/buildings/`
  - `39` manifest entries in `apps/web/public/assets/precinct/building-art-manifest.json`
- UI surfaces already consuming this registry:
  - `packages/ui/src/PrecinctArtThumb.tsx`
  - `packages/ui/src/BuildPalette.tsx`
  - `packages/ui/src/OracleHud.tsx`
  - `packages/ui/src/WorldMapPanel.tsx`
- The live precinct scene now also consumes runtime-supported building art through:
  - `apps/web/src/game/PrecinctScene.ts`
  - sprite preload + reconciliation keyed to the generated building-art registry
  - graybox fallback for runtime defs that still have no converted art
- The earlier live-runtime building art gap is now closed for:
  - `grain_field`
  - `olive_grove`
  - `incense_workshop`
  - `papyrus_reed_bed`
- The smoke harness was updated to the current UI model:
  - bottom-toolbar tool selectors instead of the removed always-open build palette
  - overlay-trigger ids instead of the removed sidebar tabs
  - `window.__oracleDebug.save/load()` instead of removed HUD save/load buttons
- The precinct smoke now also records scene art-debug output at:
  - `output/playwright/precinct-smoke/art-debug.json`
- Validation now completed for this state:
  - `pnpm typecheck`
  - `pnpm --filter @the-oracle/web build`
  - `pnpm smoke:precinct`
  - `pnpm smoke:worldmap`
  - `pnpm smoke:campaign`
  - screenshot: `output/playwright/precinct-smoke/precinct-smoke-full.png`
- Important limitation:
  - the Phaser precinct environment is still mixed-mode: buildings now use sprite art, Sacred Way roads now use first-pass sprite art, terrain now has localized runtime accents plus perimeter modules, walkers now use runtime sprites, and both props plus flora now have live runtime layers, but terrain is not yet a full atlas and the environment rules are still deterministic first-pass staging
- Current smoke-readability note:
  - the old tiny white-proxy building issue is fixed: the runtime bundle now prefers the stronger tuned imagegen building renders where available, keys their background to transparency, and the live smoke captures show materially stronger silhouettes for `priest_quarters`, `storehouse`, `castalian_spring`, and `eternal_flame_brazier`
  - there is now a wider terrain runtime path as well: `scripts/art/export-terrain-runtime-assets.mjs` exports `dry_earth`, `limestone_terrace`, `sacred_paving`, `retaining_wall`, `stone_stairs`, `cliff_edge`, and `spring_pool` into `apps/web/public/assets/precinct/terrain/`, with a generated registry at `packages/content/src/generated/terrainArt.ts`
  - that terrain runtime path now exports `15` PNGs across those `7` terrain families, because `dry_earth`, `limestone_terrace`, and `sacred_paving` now have multiple runtime variants instead of each family collapsing to one repeated crop
  - there is now a walker runtime path: `scripts/art/export-walker-runtime-assets.mjs` exports `priest`, `custodian`, `carrier`, and `pilgrim` into `apps/web/public/assets/precinct/walkers/`, with a generated registry at `packages/content/src/generated/walkerArt.ts`
  - `PrecinctScene` now uses those assets in a more deliberate way: limestone around occupied tiles, sacred paving only on exact road tiles, plus retaining-wall, cliff-edge, spring-pool, and stair accents around the active precinct, with sprite-backed walkers replacing the old debug circles
  - `PrecinctScene` now also chooses terrain variants deterministically by tile; the current starter precinct smoke uses all `4` terrace variants, all `4` dry-earth variants, and all `3` paving variants, and `window.__oracleDebug.getPrecinctArtDebug()` now includes terrain texture keys so you can verify the mix directly in `output/playwright/precinct-smoke-current/art-debug.json`
  - the scene automation got one important fix: precinct auto-framing now freezes after the first canvas click so `viewportForTile()` stays stable during browser smoke placement flows
  - the live terrain layer now also has a more deliberate sanctuary footprint: a stone core with a dry-earth apron, board-tint shaping, and a cleaner south approach, so the cluster no longer drops straight from focal assets into empty parchment
  - the live prop path now exports `13` runtime props after adding `purification_font`, and the current starter precinct smoke now renders `12` live prop sprites with a more authored south-approach composition instead of relying only on benches/statues plus incidental clutter
  - `PrecinctScene` framing was tightened again so it biases toward the sanctuary core, authored approach props, and apron anchors instead of giving equal weight to distant perimeter flora during auto-frame
  - the latest fresh validation artifacts from this exact pass are `output/playwright/precinct-smoke-current/precinct-smoke-full.png`, `output/playwright/precinct-smoke-current/art-debug.json`, and `output/playwright/campaign-smoke-current/built-full.png`
  - the next quality blocker is moving from this broader terrain layer to a more authored terrain/prop/flora atlas. Full-page precinct captures are better than before, the repeated single-crop terrain issue is now addressed, and the south approach is more intentional, but the scene still needs more varied terrain sources, perimeter dressing, and stronger environmental composition before it reads as ship-grade
- Additional scope limitation:
  - this final browser-format export path now covers buildings, terrain accents, flora, walkers, and a first prop/clutter set; fauna, portraits, deeper character packs, and UI/world packs still need their own rendered source set plus matching exporters

## Blender MCP smoke-test status

The MCP resource registry still returned an empty resource list for `blender`, but the native Blender MCP tool surface itself worked correctly in-session.

Native checks now completed successfully through the built-in Blender tools:

- scene read test passed
- viewport screenshot test passed
- code execution test passed
- template bootstrap + save test passed

Earlier direct socket smoke tests against port `9876` had already passed:

- scene read test passed
- viewport screenshot test passed
- code execution test passed
- cleanup test passed

Artifacts:

- `output/art/blender-smoke/mcp-smoke.png`
- `output/art/blender-smoke/mcp-smoke-results.json`
- `output/art/blender-smoke/mcp-write-test.json`
- `output/art/blender-templates/oracle_isometric_template.blend`
- `output/art/blender-templates/oracle_isometric_template.json`
- `output/art/renders/template/oracle_template_probe.png`

## Important local state

- `.env` exists and already contains local API keys for the art pipeline.
- Do **not** print or copy secret values into chat or docs.
- `.gitignore` was updated so local `.env` files remain untracked.

## Best next step from here

1. Treat `docs/precinct-art-integration-plan.md` as the live implementation plan for getting the converted art into the actual game.
2. Improve the mixed-mode precinct readability:
   - keep the stronger imagegen-backed building path as the runtime default where present
   - keep the first Sacred Way road sprite path in place
   - keep the new terrain runtime path and refine the current `limestone_terrace` / `sacred_paving` / `retaining_wall` / `stone_stairs` / `cliff_edge` / `spring_pool` mix into a fuller terrain atlas instead of isolated accent tiles
   - keep the expanded prop runtime path and expand beyond the current `stone_bench` / `amphora_stack` / `ritual_basin` / `bronze_tripod` / `market_stall_set` / `omphalos_stone` / `votive_offering_rack` / `grain_sacks` / `oil_jars` / `incense_censer` / `offering_bowl` / `votive_statue_small` set
   - next best visual move is refining the new flora / rocky perimeter pass into a more varied authored environment layer so the sanctuary stops reading like a small focal cluster with repeated edge dressing
   - use `output/playwright/precinct-smoke-current/art-debug.json` to verify live sprite counts and sizing while iterating
3. Keep using Blender as the source stage for non-building families:
   - `scripts/art/blender/manual-blockout-asset.py`
   - `scripts/art/blender/render-meshy-asset.py`
   - run one asset per Blender MCP call to avoid the bridge timing out on batched renders
4. Convert new rendered families into the same browser-ready pattern:
   - `output/art/pixel/<family>/`
   - `apps/web/public/assets/...`
   - public manifest
   - generated content registry
5. Expand UI/world usage beyond the current build-palette and selected-building surfaces:
   - world unlock previews are already live; extend the same treatment deeper into atlas/world panels
   - atlas/world panels
   - future upgrade or inspection dialogs

## Short version

The repo now has:

- a scoped implementation roadmap
- a full asset catalog
- a tuned Greek-oracle art direction
- OpenAI image generation integrated
- Meshy integrated
- Blender MCP installed, smoke-tested, and usable through the native tool surface
- a reusable Blender scene template checked into the workflow
- a reusable Meshy-to-Blender import/render helper checked into the workflow
- a reusable manual Blender blockout helper checked into the workflow
- reusable Meshy bulk-submit and harvest helpers checked into the workflow
- first-pass imports and renders for the first three world assets
- `36` manual Blender blockout renders now on disk, covering the remaining building family alongside the three Meshy-first imports
- `39` Meshy GLBs now downloaded locally across buildings and props
- `39` runtime-ready building pixel assets under `apps/web/public/assets/precinct/buildings/`
- a browser manifest for those building assets at `apps/web/public/assets/precinct/building-art-manifest.json`
- a typed runtime art registry at `packages/content/src/generated/buildingArt.ts`
- the runtime building bundle now prefers tuned imagegen building renders where present, with background-keyed transparency and trim metadata
- `7` runtime-ready terrain pixel assets under `apps/web/public/assets/precinct/terrain/`
- a browser manifest for those terrain assets at `apps/web/public/assets/precinct/terrain-art-manifest.json`
- a typed terrain art registry at `packages/content/src/generated/terrainArt.ts`
- `4` runtime-ready walker pixel assets under `apps/web/public/assets/precinct/walkers/`
- a browser manifest for those walker assets at `apps/web/public/assets/precinct/walker-art-manifest.json`
- a typed walker art registry at `packages/content/src/generated/walkerArt.ts`
- `12` runtime-ready prop pixel assets under `apps/web/public/assets/precinct/props/`
- a browser manifest for those prop assets at `apps/web/public/assets/precinct/prop-art-manifest.json`
- a typed prop art registry at `packages/content/src/generated/propArt.ts`
- `4` runtime-ready flora pixel assets under `apps/web/public/assets/precinct/flora/`
- a browser manifest for those flora assets at `apps/web/public/assets/precinct/flora-art-manifest.json`
- a typed flora art registry at `packages/content/src/generated/floraArt.ts`
- first-pass UI art usage in the build palette, selected-building card, and world unlock previews
- sprite-backed building rendering now active in the live `PrecinctScene` for runtime-supported building defs
- first-pass Sacred Way road sprites now active in the live `PrecinctScene`
- localized terrain-art accents now active in the live `PrecinctScene`
- a more deliberate stone-core / dry-earth precinct footprint now active in the live `PrecinctScene`, with a clearer south approach corridor and some of the footprint now carried by board tint instead of only repeated terrain overlays
- first retaining-wall, cliff-edge, spring-pool, and stair perimeter accents now active in the live `PrecinctScene`
- first deterministic prop/clutter sprites now active in the live `PrecinctScene`, with a denser second pass for storage clutter, ritual bowls/censers, and votive statuary
- first deterministic flora/perimeter sprites now active in the live `PrecinctScene`, with the current starter smoke now showing `15` live flora sprites after trimming both the initial too-dense pass and the approach-corridor clutter
- sprite-backed walker rendering now active in the live `PrecinctScene`
- a softer terrain pass in the live `PrecinctScene` so the board no longer reads as a hard debug grid
- a stability fix in the live `PrecinctScene` so auto-framing no longer fights browser smoke tile placement after the first canvas click
- precinct smoke now captures `art-debug.json` so sprite counts, scales, and texture keys are inspectable after each run
- the smoke harness is aligned with the current bottom-toolbar / overlay-trigger / debug-persistence model
- a tracked runtime incorporation plan at `docs/precinct-art-integration-plan.md`
- era-aware building variation rules

The old restart blocker is cleared and the asset queue is active. Meshy credit exhaustion still blocks more external generation after `47` additional accepted jobs, but it is no longer the main integration blocker. The current product blocker is no longer “get building art into the game”; that part is working. The current product blocker is turning the current terrain + prop + flora runtime paths and the existing building/road/walker art into a fuller scene stack, with better terrain source variety and more varied perimeter composition as the next highest-leverage pass so the precinct stops reading as strong focal assets sitting on top of a partially finished board.

## Latest visual polish checkpoint

- Portrait runtime plumbing is now live:
  - `pnpm art:pixel-portraits` now exports `8` runtime-ready portraits under `apps/web/public/assets/precinct/portraits/`
  - generated registry: `packages/content/src/generated/portraitArt.ts`
  - the final `3` portraits are currently fallback-backed aliases, so the runtime bundle is complete even though native source renders are still missing for `croesus_portrait`, `alexander_portrait`, and `roman_envoy_portrait`
- Consultation and oracle UI are materially upgraded:
  - consultation now uses portrait-backed supplicant and Pythia cards, a clearer right-rail verdict block, and stronger layered bronze/parchment card treatment
  - envoy-approach and Pythia surfaces in the oracle overlay now use runtime portraits instead of text-only slabs
  - world atlas hierarchy is cleaner, with a stronger selected-polis summary, improved panel layering, and a richer world-surface shell for atlas content
- World/faction presentation is materially upgraded:
  - `WorldOverlayPanel` now uses the portrait runtime set inside faction cards, so the atlas is no longer a pure text-and-metrics ledger beneath the map
  - the faction/world section now reads as a proper strategy surface instead of a stack of anonymous rows
- Precinct readability is materially improved:
  - `apps/web/src/game/PrecinctScene.ts` now frames the sanctuary much tighter, scales buildings more aggressively, lightens distant board noise, and keeps the hover readout off the sanctuary centerline
  - the scene now also carries lightweight ambient polish: spring shimmer, ritual haze, brazier ember pulse, and a subtler atmospheric treatment over the sanctuary core
  - the starter precinct now reads like an authored sanctuary cluster instead of a tiny asset knot on an oversized debug board
- Current validated artifacts after the polish pass:
  - `output/playwright/precinct-smoke-current/precinct-smoke-full.png`
  - `output/playwright/campaign-smoke-current/consultation-open-full.png`
  - `output/playwright/campaign-smoke-current/built-full.png`
  - `output/playwright/worldmap-smoke-current/world-map-lab-full.png`
- Current validation status:
  - `pnpm art:pixel-portraits`
  - `pnpm typecheck`
  - `pnpm --filter @the-oracle/web build`
  - `pnpm smoke:precinct -- --url http://127.0.0.1:4173 --out-dir output/playwright/precinct-smoke-current`
  - `pnpm smoke:campaign -- --url http://127.0.0.1:4173 --out-dir output/playwright/campaign-smoke-current`
  - `pnpm smoke:worldmap -- --url http://127.0.0.1:4173 --out-dir output/playwright/worldmap-smoke-current`
- validation note:
  - `packages/testkit/scripts/worldmap-smoke.mjs` was tightened so replay day assertions use a live state read instead of a stale snapshot, eliminating the `Day 72` vs `Day 73` false failure
- Remaining visual gap after this pass:
  - the live game is now at a credible polished vertical slice for precinct/world/consultation surfaces, but the full end-state still needs native source renders for the fallback-backed portraits, more authored terrain/prop variety, and finish-layer FX breadth to fully exhaust the visual plan
