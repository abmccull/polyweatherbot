# The Oracle Visual Finish Plan

This is the end-state implementation plan for taking `The Oracle` from the current art-integrated alpha to a polished, cohesive, ship-ready visual game.

It is broader than `docs/precinct-art-integration-plan.md`. That file remains the tactical precinct/runtime art plan. This file is the full-game visual finish roadmap.

## Goal

Deliver a visually polished game where:

- the precinct reads as a sacred sanctuary, not a small asset cluster on an empty board
- the world map, prophecy, logistics, and overlay screens feel part of the same authored universe
- the game no longer relies on graybox-looking fallback surfaces in primary player flows
- all major asset families have a runtime path and are used intentionally in the live game
- the visual hierarchy, readability, and atmosphere feel like a premium historical strategy game rather than a strong prototype

## Current Baseline

### Catalog

- Total tracked art catalog: `138`
- Family counts:
  - `39` building
  - `25` prop
  - `11` terrain
  - `6` flora
  - `8` fauna
  - `30` character
  - `8` portrait
  - `11` UI

### Runtime-exported now

- Buildings: `39/39`
- Terrain: `15` files across `7/11` terrain families
- Props: `13/25`
- Flora: `4/6`
- Walkers: `4` exported runtime characters

### Quality state

- Strongest visual area: building art and sanctuary material language
- Strongest systemic trait: tonal consistency across bronze, parchment, limestone, ochre, and sacred glyph motifs
- Weakest area: the main precinct still reads too small and too detached from the board
- Overall visual verdict: `alpha`

## End-State Visual Bar

The finished game should meet all of these conditions:

1. The precinct fills the player’s attention with authored terrain, roads, structures, props, flora, walkers, and effects.
2. The world map feels diagrammatic and sacred, not like a generic modal with nodes on a dark card.
3. Consultation, logistics, records, lineage, legacy, and world overlays all share one visual system and have clear information hierarchy.
4. Character-facing systems use finished portraits and role art instead of text-first abstraction.
5. Visual states communicate sanctity, strain, prosperity, scarcity, pressure, ritual readiness, and political danger without relying on raw numbers alone.
6. The UI passes the readability bar in `docs/ui-ux-design-standards.md`.
7. The game reads as a coherent Delphi-inspired universe in every major screen, not just in its building renders.

## Visual Principles To Preserve

Use `docs/art-style-bible.md` and `docs/ui-ux-design-standards.md` as hard constraints.

Do not regress into:

- generic fantasy UI
- mobile-casual saturation or toy-like proportions
- Roman-imperial monumentality too early
- unreadable dark-on-dark information panels
- placeholder geometry or empty boards in primary views

## Workstreams

## 1. Precinct Environment Finish

This is the highest-priority workstream because it is the game’s most exposed visual surface and currently the weakest in composition.

### Objectives

- make the precinct look inhabited, sacred, and spatially grounded
- remove the “small cluster on giant beige board” read
- replace mixed-mode presentation with authored world dressing

### Deliverables

- full terrain family coverage in runtime form
- final Sacred Way modular road set in live runtime usage
- broad prop coverage for sanctuary, storage, ritual, hospitality, production, and prestige zones
- full flora/perimeter coverage with stronger silhouette diversity
- tuned camera/framing so the precinct reads as a place, not a debug board
- stronger interaction overlays that complement, not obscure, the environment

### Implementation tasks

#### Terrain

- Export the remaining terrain families from the catalog into runtime form.
- Extend `scripts/art/export-terrain-runtime-assets.mjs` beyond accent tiles and current variants.
- Move from localized terrain accents toward a more authored atlas behavior in `apps/web/src/game/PrecinctScene.ts`.
- Introduce distinct sanctuary zones:
  - processional paving
  - consecrated terrace stone
  - dry hillside shoulder
  - sacred spring/wet stone zone
  - cliff/perimeter edge
- Replace more of the flat board-color logic with grounded terrain composition while preserving clarity.

#### Roads and circulation

- Break `sacred_way_kit` into readable runtime modules:
  - straight
  - turn
  - junction
  - ceremonial node
  - edge/border pieces
- Give the entry approach a consistent axial read.
- Add route-side markers, low walls, votive posts, and ritual procession cues.

#### Props

- Expand prop runtime export from `13` toward the full `25` prop family.
- Prioritize the missing high-value props already present in the library:
  - `scroll_shelves`
  - `writing_desk`
  - `treasury_chest`
  - `festival_banner_set`
  - `sacred_boundary_marker`
  - `sacrifice_table`
  - `caravan_cart`
  - `archive_scroll_crate`
  - `debate_podium`
  - `laurel_wreath_set`
  - `votive_statue_large`
  - `marble_column_fragment`
- Add per-building decor recipes in `apps/web/src/game/PrecinctScene.ts` so props reinforce function:
  - spring = basins, offerings, ritual apparatus
  - sanctum = omphalos, incense, votive clusters
  - storehouse/granary = sacks, jars, crates
  - xenon/gatehouse = benches, carts, markers
  - scriptorium/library = desks, shelves, scroll crates
  - altar/brazier = tables, tripods, sacred implements

#### Flora

- Export the remaining flora families and add more shape variety.
- Tune placement rules to frame the sanctuary instead of creating symmetric repetition.
- Use flora to establish thresholds, approaches, terraces, and sacred boundaries rather than generic scatter.

#### Camera and composition

- Continue tightening `framePrecinct()` in `apps/web/src/game/PrecinctScene.ts`.
- Reduce visible dead board space.
- Bias auto-frame to authored anchors, major buildings, and processional paths.
- Ensure the selected-building and selected-walker states do not cause awkward zoom/centering shifts.

### Acceptance criteria

- The starter precinct no longer reads empty at first glance.
- The player can visually distinguish sacred core, approach, storage, and perimeter zones.
- The board no longer dominates the scene more than the actual sanctuary.
- The live precinct screenshot reads as alpha-to-beta quality environment work, not prototype dressing.

## 2. Building Family Completion In Runtime

The building library is exported, but gameplay/runtime only uses the currently defined subset and still leaves the broader catalog stranded.

### Objectives

- close the gap between the art library and the playable precinct universe
- use the full building family intentionally, not just store it

### Implementation tasks

- Add gameplay defs for remaining art-only buildings in:
  - `packages/content/src/buildings.ts`
  - `packages/core/src/reducers/index.ts`
  - `packages/core/src/state/initialState.ts`
  - `packages/core/src/selectors/index.ts`
- Assign each building:
  - category
  - cost
  - staffing
  - unlock tier
  - production/support logic
  - sanctity/approach/political role
- Add building-specific environment rules so each new structure meaningfully changes the visual precinct.

### Acceptance criteria

- Every major buildable in the visual catalog can appear in the live precinct with intentional art support.
- No exported building art sits permanently outside the playable game universe.

## 3. Character, Walker, and Portrait Completion

The current runtime uses `4` walker sprites against a `30`-character family and `8` portrait assets.

### Objectives

- give people, roles, and narrative actors a finished visual identity
- stop relying on abstract text-only role presence

### Deliverables

- expanded walker/runtime role set
- portrait integration in narrative and management surfaces
- role-specific visual differentiation for core priesthood, staff, and envoy presence

### Implementation tasks

- Expand `scripts/art/export-walker-runtime-assets.mjs` to support more role families.
- Introduce portrait runtime registry and export path for the `8` portrait assets.
- Add portrait usage to:
  - consultation screens
  - priest panels
  - rival/origin/world-event panels
  - lineage/legacy surfaces where appropriate
- Ensure all key human roles read as Hellenic ritual staff, not generic clergy.

### Acceptance criteria

- Core human-facing systems have portrait support.
- Walkers are visually differentiated enough to read functionally during play.
- Character presence increases emotional and narrative weight across systems.

## 4. UI And HUD Polish

The HUD is coherent but too flat, too dense, and still visually under-resolved.

### Objectives

- move from themed prototype UI to premium strategy UI
- make hierarchy and readability match the theme’s seriousness

### Primary surfaces

- `packages/ui/src/OracleHud.tsx`
- `packages/ui/src/WorldMapPanel.tsx`
- `packages/ui/src/WorldOverlayPanel.tsx`
- `packages/ui/src/PrecinctOverviewPanel.tsx`
- `packages/ui/src/BuildPalette.tsx`
- `packages/ui/src/BottomToolbar.tsx`
- `packages/ui/src/OverlayTriggerStrip.tsx`

### Implementation tasks

#### Hierarchy

- Reduce the number of equal-weight panels on screen at once.
- Make one focal layer dominant per screen:
  - world view
  - selected entity
  - modal decision surface
- Increase contrast separation between:
  - primary action
  - selected state
  - secondary detail
  - ambient chrome

#### Typography

- Apply the scale and floor rules from `docs/ui-ux-design-standards.md`.
- Reduce small serif overload in dense panels.
- Use stronger numeric emphasis and better label/value separation.

#### Controls

- Redesign the right-rail overlay strip so it reads intentional, not tiny and incidental.
- Refine button shapes, states, spacing, and active markers.
- Remove lingering generic web-UI affordances that break the bronze-tablet tone.

#### Visual language

- Introduce more sacred/bronze ornament where it clarifies hierarchy.
- Remove decorative weight where it muddies information.
- Keep the interface warm, authoritative, and readable.

### Acceptance criteria

- No major panel feels like a dark brown text slab.
- Resource bars, modal choices, map nodes, and selected-state panels are readable at a glance.
- The HUD feels integrated with the universe rather than layered on top of it.

## 5. World Map, Atlas, And Campaign-Scope Visuals

The world-map direction is promising, but it needs to become a finished strategic layer rather than just the cleanest existing modal.

### Objectives

- make the world layer feel sacred, geopolitical, and authored
- connect it visually to the precinct and prophecy systems

### Implementation tasks

- Add art-backed faction/node markers and region-state cues.
- Improve node-link differentiation by route type.
- Add stronger map background treatment and composition.
- Expand use of building and landmark art in unlock, dedication, and milestone flows.
- Give pressures, rival influence, and crises clearer visual signatures.

### Acceptance criteria

- The world map feels like part of the same game universe as the precinct.
- A screenshot of the world layer is immediately legible and visually distinctive.

## 6. Consultation, Prophecy, Logistics, And Narrative Surfaces

These are currently serviceable but too dense and too heavy.

### Objectives

- make these panels feel ceremonial, high-stakes, and readable
- deliver more visual payoff for prophecy and ritual systems

### Implementation tasks

- Consultation:
  - improve hierarchy between supplicant, priest voices, selected fragments, and action area
  - add portrait support where appropriate
  - introduce clearer sacred/divine emphasis for reading-quality states
- Logistics:
  - clarify flow, queues, and movement states with cleaner status design
  - make the logistics view feel like a real management layer, not a debug derivative of the precinct
- Record/legacy/lineage:
  - add visual anchors and stronger panel organization
  - integrate portrait/emblem/icon usage where it clarifies lineage, reputation, or sacred history

### Acceptance criteria

- These screens read as designed game features, not system-heavy overlays.
- The prophecy loop feels visually special enough to justify the theme.

## 7. Effects, Motion, And State Feedback

This is the bridge from “solid art” to “alive game.”

### Objectives

- give sacred systems visible feedback
- create ritual atmosphere without noise

### Effects to implement

- brazier/fire flicker
- spring shimmer and water highlight
- incense/smoke drift
- sanctity halo / consecration pulse
- build placement affirmation
- selected-building and selected-walker feedback
- world pressure / crisis emphasis

### Motion principles

- meaningful, sparse, and readable
- no generic micro-animation spam
- motion should reinforce state, not decorate emptiness

### Acceptance criteria

- Players can feel the difference between a cold prototype board and a living sacred site.
- Ritual and political states produce meaningful visual response.

## 8. Remaining Asset Family Export Plan

To finish the visual universe, each family needs the same structure that buildings now have:

- source render
- runtime export
- public manifest
- generated typed registry
- live integration path

### Required remaining families

- remaining terrain families
- remaining props
- remaining flora
- portraits
- broader character set
- fauna
- UI-family art

### Pipeline files likely required

- new/expanded exporters in `scripts/art/`
- new generated registries in `packages/content/src/generated/`
- new public bundles in `apps/web/public/assets/`
- scene/panel integration in `apps/web/src/game/` and `packages/ui/src/`

## Execution Order

This is the recommended finish order.

## Phase 0: Lock the visual bar

- Treat `docs/art-style-bible.md` and `docs/ui-ux-design-standards.md` as hard constraints.
- Use this file as the end-state roadmap.
- Reject any asset or UI change that improves quantity while lowering cohesion.

## Phase 1: Finish the precinct read

- environment atlas
- road kit
- prop expansion
- flora expansion
- camera/framing
- precinct overlay cleanup

This is the highest-value phase.

## Phase 2: Finish the strategic and narrative UI

- world map
- consultation
- logistics
- record/legacy/lineage panels
- stronger HUD hierarchy

## Phase 3: Finish the human layer

- walkers
- characters
- portraits
- narrative art integration

## Phase 4: Close catalog-to-runtime gaps

- remaining buildable defs
- remaining export families
- generalized art-loader support where needed

## Phase 5: Effects and ship polish

- sacred FX
- motion polish
- visual QA sweeps
- readability pass
- screenshot review pass across all major states

## Definition Of Done

The visual finish work is done only when all of the following are true:

- the precinct no longer looks like a small cluster on an empty board
- all major game surfaces feel like one world, not separate systems with separate quality levels
- building, environment, character, portrait, and UI art all have real runtime integration
- the UI hierarchy is readable and calm under real gameplay load
- the game looks materially closer to beta/near-ship than alpha in fresh screenshots
- smoke captures across precinct, world map, consultation, logistics, and campaign progression are visually publishable

## Practical Next Step

The immediate next implementation sequence should be:

1. expand environment props/flora/terrain until the precinct board stops reading empty
2. tighten precinct framing and hierarchy again after that asset pass
3. redesign world/consultation hierarchy using the existing standards doc
4. add portraits and broader character runtime support
5. finish the remaining export families and effects layer
