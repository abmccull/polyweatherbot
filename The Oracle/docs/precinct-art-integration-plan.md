# Precinct Art Integration Plan

This plan is grounded in the current runtime code, not just the art docs.

For the full-game visual finish roadmap, see `docs/visual-finish-plan.md`.

## What Landed Now

- Converted the current building render set into runtime-loadable pixel PNGs under `apps/web/public/assets/precinct/buildings/`.
- Wrote a public manifest at `apps/web/public/assets/precinct/building-art-manifest.json`.
- Mirrored the same converted set under `output/art/pixel/buildings/` for pipeline review.
- Generated a typed runtime registry at `packages/content/src/generated/buildingArt.ts`.
- Upgraded `scripts/art/export-building-pixel-assets.mjs` so the runtime bundle prefers the stronger tuned imagegen building renders when present, keys out their flat background, and emits trim metadata used by the scene.
- Added `scripts/art/export-terrain-runtime-assets.mjs`, which extracts a first runtime terrain set from the tuned terrain sheets and generates `packages/content/src/generated/terrainArt.ts`.
- Expanded `scripts/art/export-terrain-runtime-assets.mjs` so the repeat-heavy terrain families now export multiple runtime variants instead of a single reused crop, giving the live precinct `4` `dry_earth` variants, `4` `limestone_terrace` variants, and `3` `sacred_paving` variants inside the same generated terrain registry.
- Added `scripts/art/export-flora-runtime-assets.mjs`, which extracts the first runtime flora/perimeter set and generates `packages/content/src/generated/floraArt.ts`.
- Added `scripts/art/export-walker-runtime-assets.mjs`, which extracts runtime walker sprites from the tuned character sheets and generates `packages/content/src/generated/walkerArt.ts`.
- Landed sprite-backed building rendering in `apps/web/src/game/PrecinctScene.ts` for runtime-supported building art, with graybox fallback when a building def still has no converted art.
- Added first live Sacred Way road sprites in `apps/web/src/game/PrecinctScene.ts`, using the existing `sacred_way_kit` art instead of flat brown road diamonds where runtime art is available.
- Added first localized terrain accents in `apps/web/src/game/PrecinctScene.ts` so built/road-adjacent precinct tiles can use extracted terrain art instead of relying only on the flat ground pass.
- Expanded that terrain layer with first perimeter-shaping modules so the live precinct can place `retaining_wall` and `stone_stairs` accents around the active cluster instead of only showing flat tile accents.
- Expanded the terrain runtime again so `cliff_edge` and `spring_pool` are also exported and can shape the live precinct more like a sanctuary terrace.
- Softened the precinct ground pass in `apps/web/src/game/PrecinctScene.ts` so the terrain no longer overpowers the art with hard debug-grid contrast.
- Added a first live flora/perimeter layer in `apps/web/src/game/PrecinctScene.ts`, with deterministic `cypress_tree`, `laurel_shrub`, `dry_grass_cluster`, and `rocky_outcrop` placements around the active sanctuary cluster.
- Added a first shaped ground-footprint pass in `apps/web/src/game/PrecinctScene.ts`, using a stone-core / dry-earth apron sequence, board-tint shaping, and an intentionally clearer south approach instead of letting the cluster dissolve directly into the parchment board.
- Added deterministic terrain-variant selection in `apps/web/src/game/PrecinctScene.ts`, so the live precinct no longer stamps a single terrace/dry-earth/paving texture everywhere it uses those families.
- Replaced the old walker-circle-only view with sprite-backed walker rendering for `priest`, `custodian`, `carrier`, and `pilgrim`.
- Added first-pass UI consumption for runtime-supported buildings in:
  - `packages/ui/src/BuildPalette.tsx`
  - `packages/ui/src/OracleHud.tsx`
  - `packages/ui/src/PrecinctArtThumb.tsx`
  - `packages/ui/src/WorldMapPanel.tsx`
- Expanded the tracked building catalog and export path to cover the live farming / workshop ids:
  - `grain_field`
  - `olive_grove`
  - `incense_workshop`
  - `papyrus_reed_bed`
- Refreshed the browser smoke harness so it drives the current bottom-toolbar, overlay-trigger, and debug save/load model instead of the removed sidebar/save-button selectors.
- Revalidated the current state with:
  - `pnpm typecheck`
  - `pnpm --filter @the-oracle/web build`
  - `pnpm smoke:precinct`
  - `pnpm smoke:worldmap`
  - `pnpm smoke:campaign`
  - `output/playwright/precinct-smoke/precinct-smoke-full.png`

## Current Final Usable Format

For the live browser client, the final usable art format is:

- transparent pixel PNGs served from `apps/web/public/assets/precinct/buildings/`
- transparent pixel PNGs served from `apps/web/public/assets/precinct/terrain/`
- transparent pixel PNGs served from `apps/web/public/assets/precinct/flora/`
- transparent pixel PNGs served from `apps/web/public/assets/precinct/props/`
- transparent pixel PNGs served from `apps/web/public/assets/precinct/walkers/`
- a browser-loadable manifest at `apps/web/public/assets/precinct/building-art-manifest.json`
- a browser-loadable manifest at `apps/web/public/assets/precinct/terrain-art-manifest.json`
- a browser-loadable manifest at `apps/web/public/assets/precinct/flora-art-manifest.json`
- a browser-loadable manifest at `apps/web/public/assets/precinct/prop-art-manifest.json`
- a browser-loadable manifest at `apps/web/public/assets/precinct/walker-art-manifest.json`
- a typed code registry at `packages/content/src/generated/buildingArt.ts`
- a typed code registry at `packages/content/src/generated/terrainArt.ts`
- a typed code registry at `packages/content/src/generated/floraArt.ts`
- a typed code registry at `packages/content/src/generated/propArt.ts`
- a typed code registry at `packages/content/src/generated/walkerArt.ts`

Current scope:

- `39` building PNGs are exported in the final browser format
- `15` terrain PNGs are exported in the final browser format across `7` terrain families
- `4` flora PNGs are exported in the final browser format
- `13` prop PNGs are exported in the final browser format
- `4` walker PNGs are exported in the final browser format
- the selected-building card and build palette can already consume them
- world unlock previews can now consume the same registry
- the Phaser precinct scene now consumes the runtime-supported building subset directly
- the Phaser precinct scene now also consumes runtime-supported Sacred Way road art directly
- the Phaser precinct scene now also consumes a first localized terrain-art subset around roads/buildings
- the Phaser precinct scene now also consumes first-pass perimeter-shaping terrain modules (`retaining_wall`, `stone_stairs`, `cliff_edge`, `spring_pool`)
- the Phaser precinct scene now also consumes a shaped terrain footprint under the active precinct cluster instead of only isolated tile accents, with part of that footprint carried by the board tint rather than only repeated terrain overlays
- the Phaser precinct scene now also distributes terrain-family variants deterministically by tile, and the starter smoke currently exercises all `4` terrace variants, all `4` dry-earth variants, and all `3` paving variants
- the Phaser precinct scene now also consumes deterministic flora/perimeter dressing around the active precinct cluster
- the Phaser precinct scene now also consumes deterministic prop/clutter dressing around roads, storage, spring, sanctum, altar, and market anchors
- the Phaser precinct scene now also uses `purification_font` as a runtime prop and stages a more deliberate south-approach composition instead of leaving the entry to generic road clutter alone
- the Phaser precinct scene now also consumes runtime-supported walker sprites for the current live roles
- fauna, portraits, characters, and UI packs are not yet converted through this same runtime path

## Current Runtime Seams

### 1. The live precinct is now mixed-mode, not fully art-driven yet

The live precinct scene in `apps/web/src/game/PrecinctScene.ts` now uses a mixed render stack:

- terrain diamonds via `terrainGraphics`, plus localized terrain-art accents and perimeter-shaping terrain modules around active precinct tiles
- Sacred Way road sprites via a dedicated sprite pool, with `roadGraphics` fallback
- flora/perimeter sprites via a deterministic placement layer keyed by placement id
- prop/clutter sprites via a deterministic placement layer keyed by placement id
- building sprites via the generated building-art registry
- graybox building fallback via `buildingGraphics` when no converted art exists
- walker sprites via the generated walker-art registry, with `walkerGraphics` fallback only if a future role has no converted art
- selection/inspection overlays via `overlayGraphics`

There is now:

- a Phaser `preload()` path for runtime-supported building art
- a Phaser `preload()` path for runtime-supported terrain, flora, props, and walker art
- a sprite pool keyed by live `building.id`
- a sprite pool keyed by deterministic flora placement ids
- a sprite pool keyed by deterministic prop placement ids
- a sprite pool keyed by live `walker.id`
- footprint-based size normalization so the source renders fit the current one-tile placement model

There is still no:

- full terrain atlas
- authored biome/perimeter variety beyond the current deterministic starter rules
- generalized world/portrait/UI art loader

### 2. Runtime building definitions still lag behind the expanded art catalog

The art library now tracks `39` building assets in `art-library/generated/asset-manifest.json`.

The actual playable building defs in `packages/content/src/buildings.ts` currently cover `20` ids:

- `sacred_way`
- `priest_quarters`
- `storehouse`
- `castalian_spring`
- `inner_sanctum`
- `eternal_flame_brazier`
- `sacrificial_altar`
- `animal_pen`
- `granary`
- `kitchen`
- `olive_press`
- `incense_store`
- `agora_market`
- `xenon`
- `grain_field`
- `olive_grove`
- `incense_workshop`
- `papyrus_reed_bed`
- `scriptorium`
- `library`

So the art pipeline is ahead of the gameplay model. The extra building art cannot appear in the actual precinct until matching runtime defs, unlock rules, and reducer support exist.

The earlier live-runtime art gap is now closed: every current gameplay building def has a matching converted building art entry in the generated runtime registry.

### 3. UI surfaces are mostly text/color-driven today

The main integration surfaces are:

- `packages/ui/src/BuildPalette.tsx`
- `packages/ui/src/OracleHud.tsx`
- `packages/ui/src/PrecinctOverviewPanel.tsx`
- `packages/ui/src/WorldOverlayPanel.tsx`
- `packages/ui/src/WorldMapPanel.tsx`

Before this pass, they relied on:

- building colors from `packages/content/src/buildings.ts`
- emoji glyphs
- text summaries
- CSS shapes / minimap SVGs

The build palette, selected-building card, and world unlock previews now consume converted art directly.

## Phase Plan

## Phase 1: Treat converted PNGs as the source of truth for runtime art

Status: landed for export, scene, first-pass UI, and current gameplay building coverage

Tasks:

- keep `apps/web/public/assets/precinct/buildings/` as the runtime asset root
- keep `apps/web/public/assets/precinct/building-art-manifest.json` as the loader manifest
- extend `packages/ui/src/PrecinctArtThumb.tsx` whenever a runtime building def needs to map to a non-identical art id, such as `sacred_way -> sacred_way_kit`

Result:

- art is now in a format the browser can load directly with static URLs
- the first UI surfaces can already show it
- the live precinct scene now consumes runtime-supported building art directly
- the live precinct scene now consumes first-pass Sacred Way road art directly

## Phase 2: Swap precinct building rendering from graphics to sprite-backed entities

Status: landed for the current gameplay building subset with sprite fallback still preserved

Target files:

- `apps/web/src/game/PrecinctScene.ts`
- `apps/web/src/game/createGame.ts`
- `apps/web/src/game/iso.ts`

Tasks:

- keep the Phaser `preload()` path bound to the generated building-art registry
- keep the sprite cache keyed by `building.id`
- tune sprite contrast/anchor polish now that the base reconcile path is live
- keep `overlayGraphics` for hover, selection halos, inspection glow, and site links
- keep `terrainGraphics` and `roadGraphics` temporarily until terrain/road art lands
- keep graybox fallback drawing when a building has no mapped sprite yet

Recommended implementation shape:

1. Keep the runtime art registry keyed by `BuildingDefId`.
2. Preserve sprite reconciliation for the current building subset.
3. Preserve the existing condition bar / priest pips as overlay graphics until dedicated sprite overlays exist.
4. Expand from buildings into terrain, roads, and props instead of reworking the building pass again.

## Phase 3: Expand UI art usage beyond the selected-building card

Status: partially landed

Target files:

- `packages/ui/src/BuildPalette.tsx`
- `packages/ui/src/OracleHud.tsx`
- `packages/ui/src/WorldMapPanel.tsx`
- `packages/ui/src/WorldOverlayPanel.tsx`

Tasks:

- keep thumbnails in build-palette list rows and the selected-building inspector
- keep unlock-preview art in world progression surfaces
- add building portrait art to any future upgrade dialogs
- reserve world-atlas art slots for region/node illustrations later, but keep that separate from precinct building art

## Phase 4: Close the content/runtime mismatch

Target files:

- `packages/content/src/buildings.ts`
- `packages/core/src/reducers/index.ts`
- `packages/core/src/state/initialState.ts`
- `packages/core/src/selectors/index.ts`

Tasks:

- add gameplay defs for the remaining art-library buildings that are currently art-only
- assign categories, costs, unlock tiers, staffing, recipes, and passive effects
- thread those defs through unlock flow, placement flow, and any selectors/HUD sections that assume the smaller current set

This is the real blocker to using all `39` tracked building sprites in the live precinct.

## Phase 5: Terrain, roads, and prop dressing

Status: partially landed for terrain accents, road art, walkers, deterministic prop/clutter, and a first flora/perimeter layer

Target next:

- fuller terrain tiling and better source variety beyond the current localized accent + shaped-footprint pass
- more varied flora / rocky perimeter dressing so the sanctuary edge stops repeating the same small set of silhouettes
- Sacred Way modular kit decomposition
- broader processional markers / prop clutter coverage
- prop-to-building dressing around high-value buildings like sanctum, spring, altar, and market

Without this phase, the scene will improve, but still read as "painted buildings over a prototype board."

## Current Blockers

- `scripts/art/pixel-art.mjs` was previously broken for direct conversion; it now works, but terrain/prop batching still needs its own command path.
- The runtime only knows `20` building defs, while the art catalog tracks `39`.
- `PrecinctScene` still uses graphics for the background ground board and graybox fallback buildings without art, even though buildings, roads, terrain accents, flora, props, and walkers now have sprite-backed runtime paths.
- The runtime now has first terrain, flora, and prop sets plus a shaped precinct footprint, but not a fuller authored environment atlas.
- The runtime now has first terrain, flora, and prop sets plus a shaped precinct footprint, and the terrain families no longer collapse to a single repeated sprite, but it still does not have a fuller authored environment atlas.
- The world/UI surfaces do not yet have a generalized art registry; the current thumbnail helper is intentionally narrow and building-specific.
- The live sprite pass is better but still not ship-grade: the cluster now has a clearer footprint, terrain variation, a stronger south approach, plus active prop/flora dressing, but it still needs more authored terrain source coverage, more silhouette variety, and stronger environmental layering before the precinct reads like a finished sanctuary.

## Recommended Next Implementation Order

1. Expand the environment layer: better terrain source variety, more flora/perimeter variety, more prop variants, and better composition around the active sanctuary cluster.
2. Improve precinct readability further: keep tuning sprite scale, anchoring, framing, and contrast so rendered assets stop disappearing into the parchment ground.
3. Expand world/UI art usage beyond the current unlock-preview surfaces.
4. Add missing runtime defs for the remaining art-only buildings.
