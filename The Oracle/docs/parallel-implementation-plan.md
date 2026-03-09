# The Oracle Parallel Implementation Plan

## Purpose

This document turns the current GDD gap analysis into a parallel execution plan that can be distributed across the locally available Codex agent roles, with the top-level `default` agent acting as lead integrator. The goal is to maximize implementation throughput in a single coordinated burst without causing avoidable merge conflicts in the core simulation.

## Current Baseline

The current repo is a strong vertical slice, not the full GDD:

- Browser-first Phaser + React workspace is working.
- Deterministic daily sim, logistics, consultations, consequence resolution, monthly politics, save/load, and smoke coverage already exist.
- The implemented content footprint is still narrow: 6 building defs, 5 resources, 3 omen families, 3 consultation question templates, 1 Pythia, 8 factions, and no true world-map UI.
- The biggest missing pieces are the expanded precinct economy, full omen/Pythia depth, campaign progression, world map/crisis chains, and production-quality UI/art/audio systems.

## Planning Assumptions

- The top-level `default` agent is the **lead integrator / master orchestrator** for task assignment, parallel launches, monitoring handoffs, and final merge sequencing.
- Because local agent config sets `agents.max_depth = 1`, do not assume a spawned child agent can become the orchestrator of other child agents. Keep orchestration in the parent session.
- The practical concurrency ceiling is lower than the raw `agents.max_threads = 6`: keep one slot free for the orchestrator and one for QA/monitoring or emergency follow-up, so Wave 1 should usually run at most 3 implementation tracks at once.
- Specialty sub agents should work on **separated surfaces** after the schema freeze to reduce conflicts.
- The current smoke tests remain the required regression baseline:
  - `pnpm smoke:precinct`
  - `pnpm smoke:consultation`
  - `pnpm smoke:logistics`
  - `pnpm smoke:campaign`
- Every feature branch should add or extend:
  - deterministic debug scenarios
  - state surfaced through `render_game_to_text`
  - a targeted test or smoke path

## Available Local Agent Roles

| Agent Role | Best Use | Use In This Plan |
|---|---|---|
| `default` | Lead coordination across specialist agents | Owns `INT-0`, `INT-1`, launch sequencing, dependency management, and merge train decisions |
| `gameplay_engineer` | Deterministic game systems, progression loops, tuning hooks, state machines | Primary owner for economy, consultation semantics, and campaign-state expansion |
| `web_game_engineer` | Browser-rendered game implementation with the `develop-web-game` skill | Phaser/React integration, scene wiring, and rendered validation for game-facing changes |
| `ui_craftsman` | Frontend and Electron UI implementation | HUD, panel, inspector, minimap, and readability/polish work |
| `playwright_tester` | Browser and Electron testing with screenshots and repro evidence | Smoke expansion, interactive browser debugging, and capture workflows |
| `monitor` | Waiting on long-running jobs and repeated status checks | Watching dev servers, long smoke suites, and artifact generation while the orchestrator keeps moving |
| `security_reviewer` | Read-only AppSec review | Electron/persistence/mod-surface review and remediation guidance |
| `electron_engineer` | Desktop app behavior, preload, IPC, packaging | Follow-up fixes if `SEC-1` finds Electron-specific issues |
| `worker` | Generic bounded implementation work | Workbook creation via the `spreadsheet` skill, asset manifests, and one-off utility tasks that do not need a domain specialist |
| `game_design_reviewer` | Read-only game design critique | Optional sanity pass on progression clarity, pacing, and reward loops before final merge |

Skills remain part of the workflow, but they are not launchable agent roles. In this plan, `develop-web-game`, `imagegen`, `playwright`, `playwright-interactive`, `spreadsheet`, `screenshot`, and `security-best-practices` should be invoked by the owning agent above rather than treated as standalone sub agents.

## Delivery Strategy

Use four waves:

1. **Wave 0 — Shared foundation freeze**
2. **Wave 1 — Parallel gameplay implementation**
3. **Wave 2 — QA/debug integration sweep**
4. **Wave 3 — Final merge and stabilization**

The top-level `default` orchestrator handles Wave 0 and Wave 3. Specialist agents do most of Wave 1 and Wave 2.

Recommended live mix:

1. `default` orchestrator kept free for launches, conflict resolution, and integration
2. Up to 3 implementation agents in parallel
3. 1 QA or monitoring sidecar (`playwright_tester` or `monitor`)

Do not fill all available slots with code-writing agents; the plan needs capacity for QA, waiting, and follow-up fixes.

## Shared-Core Conflict Hotspots

These files should have a single owner at a time:

- `packages/core/src/state/gameState.ts`
- `packages/core/src/state/initialState.ts`
- `packages/content/src/schema.ts`
- `packages/content/src/data.ts`
- `packages/persistence/src/schema.ts`
- `packages/core/src/simulation/updateDay.ts`
- `packages/core/src/simulation/events.ts`
- `packages/ui/src/OracleHud.tsx`
- `packages/ui/src/ConsultationOverlay.tsx`
- `apps/web/src/game/PrecinctScene.ts`

To parallelize safely, the lead integrator should first split large registries and systems into smaller modules before feature agents branch.

---

## Wave 0 — Shared Foundation Freeze

### `INT-0` Lead Integrator — Data-Driven Core Expansion

**Owner:** `default`

**Why this must land first**

Almost every major GDD feature requires widening core unions and removing hardcoded special cases. If this work is not done first, every parallel branch will collide in the same files.

**Tasks**

- Expand core IDs and registries:
  - widen `ResourceId`
  - widen `BuildingDefId`
  - widen `PriestRole`
  - add scenario/reputation/world-map state types
- Split content data into domain files:
  - `buildings.ts`
  - `resources.ts`
  - `omens.ts`
  - `wordTiles.ts`
  - `factions.ts`
  - `scenarios.ts`
- Replace building-specific processing with a generic model where possible:
  - input/output recipes
  - staffing requirements
  - storage caps
  - passive effects / upkeep hooks
  - unlock tier metadata
- Add save-schema migration support for newly added fields.
- Add reputation tier state and unlock hooks, even if content is not fully wired yet.
- Add new debug scenario scaffolds to support downstream agents.

**Expected Deliverables**

- stable shared types for Wave 1
- modular content files
- migration-safe persistence updates
- no functional regressions in current smoke coverage

**Validation**

- `pnpm typecheck`
- `pnpm test`
- `pnpm smoke:precinct`
- `pnpm smoke:consultation`
- `pnpm smoke:logistics`
- `pnpm smoke:campaign`

**Branching Rule**

No Wave 1 feature branch should start from `main`; they should start from the branch or commit containing `INT-0`.

---

## Wave 1 — Parallel Gameplay Implementation

### `DVG-1` Precinct Economy Agent

**Owner:** `gameplay_engineer`

**Goal**

Expand the sacred precinct from a 6-building slice into a real temple-management economy loop.

**Primary Scope**

- Add these buildings first:
  - `sacrificial_altar`
  - `animal_pen`
  - `granary`
  - `kitchen`
  - `olive_press`
  - `incense_store`
  - `agora_market`
  - `xenon`
- Add these resources first:
  - `sacred_animals`
  - `bread`
  - `olives`
  - `papyrus`
  - `scrolls`
- Implement the first full production chains:
  - imported grain -> granary -> kitchen -> bread / feeding
  - olives -> olive press -> oil -> brazier
  - animal pen -> altar -> omen-quality input
  - imported incense -> incense store -> sanctum / ritual use
- Add building-specific demand and logistics jobs through the generic system from `INT-0`.
- Add visitor-facing economy hooks:
  - inn capacity
  - market income
  - donation modifiers from precinct quality

**Secondary Scope**

- Add sacred animal walkers.
- Add localized UI support in the HUD for the expanded economy.
- Add one deterministic debug scenario for each new chain.

**Files Likely Touched**

- `packages/content/src/buildings.ts`
- `packages/content/src/resources.ts`
- `packages/core/src/simulation/updateDay.ts`
- `packages/core/src/reducers/index.ts`
- `packages/ui/src/OracleHud.tsx`
- `apps/web/src/game/PrecinctScene.ts`
- `packages/core/tests/logistics.test.ts`
- `packages/testkit/scripts/`

**Avoid**

- consultation UI
- world map implementation
- trait logic outside economy impact
- broad HUD regrouping and polish that belongs in `DVG-4`

**Acceptance Criteria**

- precinct can support more than one meaningful economy chain
- resource shortages create visible operational consequences
- new buildings are placeable and inspectable
- smoke or test coverage proves new chains function deterministically

**Validation**

- unit tests for recipes/logistics
- new `economy-smoke` or expanded `campaign-smoke`
- screenshots/artifacts under `output/playwright/`

---

### `DVG-2` Consultation & Pythia Agent

**Owner:** `gameplay_engineer`

**Goal**

Turn the existing consultation slice into the real centerpiece mechanic described in the GDD.

**Primary Scope**

- Add remaining omen families:
  - `extispicy`
  - `oneiromancy`
  - `astromancy`
- Add priest roles needed to source them:
  - `augur`
  - `sacrificial_priest`
  - `dream_priest`
  - `astronomer`
  - `scholar`
- Expand consultation question content substantially by faction/domain.
- Expand word-tile library and semantic tags.
- Improve prophecy assembly UX:
  - category slots, or
  - drag/reorder if stable in Phaser/DOM overlay
- Add variable due horizons and stronger consequence interpretation rules.
- Wire Pythia traits into real gameplay modifiers:
  - reliability
  - clarity/value/risk tuning
  - overwork penalties
  - delivery modifiers

**Secondary Scope**

- Add pilgrimage action groundwork.
- Add second-Pythia slot groundwork, even if full roster UI is deferred.
- Surface more consultation internals in `render_game_to_text` for QA.

**Files Likely Touched**

- `packages/content/src/omens.ts`
- `packages/content/src/wordTiles.ts`
- `packages/content/src/questions.ts`
- `packages/core/src/simulation/events.ts`
- `packages/core/src/reducers/index.ts`
- `packages/core/src/selectors/index.ts`
- `packages/ui/src/ConsultationOverlay.tsx`
- `packages/core/tests/consultation.test.ts`
- `packages/core/tests/consequence.test.ts`

**Avoid**

- precinct logistics changes outside omen sourcing
- world-map view implementation
- broad HUD polish that belongs in `DVG-4`

**Acceptance Criteria**

- at least 6 omen families are live
- trait effects measurably alter outcomes
- consultation UI can support richer prophecy construction
- consequence text is more varied and aligned to prophecy semantics

**Validation**

- updated consultation tests
- expanded `smoke:consultation`
- one full consultation round-trip through save/load

---

### `DVG-3` Politics, Reputation, and Campaign Agent

**Owner:** `gameplay_engineer`

**Goal**

Build the first real campaign/meta layer around the precinct.

**Primary Scope**

- Add reputation tiers and unlock rules.
- Add the first real world-map state model:
  - city nodes
  - trade links
  - conflict/league markers
  - consultation pressure indicators
- Define stable selectors and view-model contracts for a world-map UI panel or scene.
- Add treasury dedication logic and patron milestones.
- Add multi-step crisis-chain support for political events.
- Implement `Rising Oracle` scenario progression and win condition.

**Secondary Scope**

- Add rival-site groundwork.
- Add hooks for philosopher/backlash/defilement threats, even if only one threat is fully playable.

**Files Likely Touched**

- `packages/content/src/factions.ts`
- `packages/content/src/scenarios.ts`
- `packages/content/src/politicalEvents.ts`
- `packages/core/src/simulation/events.ts`
- `packages/core/src/state/gameState.ts`
- `packages/core/src/selectors/index.ts`
- `packages/core/tests/politics.test.ts`

**Avoid**

- consultation tile-builder mechanics
- core logistics recipe work outside political effects
- direct HUD / panel implementation that belongs in `DVG-4`

**Acceptance Criteria**

- campaign progression state is stable and queryable through selectors
- reputation tiers unlock content
- at least one scenario has a defined progression path and completion condition

**Validation**

- politics tests expanded for tiers and crises
- new `worldmap-smoke` or expanded `campaign-smoke`

---

### `DVG-4` Precinct UI, Scene, and Feedback Agent

**Owner:** `web_game_engineer`

**Goal**

Close the gap between functional graybox and the GDD’s intended management-game readability.

**Primary Scope**

- Add a minimap panel.
- Add the first world-map panel and HUD integration on top of `DVG-3` selectors.
- Add placement ghost previews and clearer invalid/valid footprints.
- Add building labels / icons / role hints for new content.
- Add simple particles and feedback:
  - brazier flicker
  - spring shimmer
  - incense drift
  - donation popups
- Add advisor polish:
  - stronger warnings
  - event-linked callouts
  - better grouping in HUD
- Improve selected-entity inspector for buildings, carriers, priests, and Pythia.

**Secondary Scope**

- Add day/night tint pass if it does not destabilize visibility.
- Improve walker/building placeholder rendering while remaining art-agnostic.

**Files Likely Touched**

- `apps/web/src/game/PrecinctScene.ts`
- `apps/web/src/styles.css`
- `packages/ui/src/OracleHud.tsx`
- new `packages/ui/src/WorldMapPanel.tsx`
- new UI components under `packages/ui/src/`

**Avoid**

- changing economic rules unless needed for display
- changing consequence logic
- reshaping campaign-state schemas that belong in `DVG-3`

**Acceptance Criteria**

- newly added systems are readable without raw debug JSON
- map interactions remain automation-friendly
- UI remains stable under save/load and open overlay states

**Validation**

- visual screenshots
- targeted browser checks
- no regressions to existing selectors / smoke hooks
- optional `ui_craftsman` review on HUD and panel hierarchy before merge

---

### `ART-1` Meshy-Assisted Art Pipeline Agent

**Owner:** `web_game_engineer`

**Goal**

Turn the current graybox slice into the start of a real production art pipeline without blocking gameplay implementation.

**Primary Scope**

- create the first dedicated art-production lane using `imagegen`, Meshy, and `proper-pixel-art`
- replace the current graybox slice with a small but production-directed visual kit
- establish the conversion and QA workflow so later art batches can run in parallel with systems work

**Tasks**

- maintain `docs/meshy-art-pipeline.md` and `docs/asset-pipeline-setup.md`
- create the first asset manifest workbook
- generate anchor concepts for the live vertical slice
- add preview surfaces for building silhouettes, footprint overlays, and character readability
- integrate the first non-graybox visual upgrade for the live slice
- add automated visual gates with Playwright screenshots

**Dependencies**

- `INT-0` should land first so content IDs are stable enough for asset metadata
- `DVG-4` should coordinate any HUD or selection-outline changes that affect readability checks

**Guardrails**

- do not block gameplay merges on perfect art
- do not attempt full production animation sheets in the first pass
- do not store Meshy credentials in the repo
- keep generated assets modular and replaceable until the style bible is locked

**Acceptance Criteria**

- the repo has a documented, repeatable art-production workflow
- the current vertical slice no longer depends entirely on graybox placeholders
- the team can validate new visual assets in-engine before approving batches

---

### `SHEET-1` Balance Workbook Agent

**Owner:** `worker` using the `spreadsheet` skill

**Goal**

Produce a balancing workbook that lets the implementation agents stop guessing numbers.

**Primary Scope**

- Create a workbook under `docs/balance/` with tabs for:
  - building costs and upkeep
  - resource production/consumption rates
  - envoy payments and credibility deltas
  - Pythia fatigue/purification pacing
  - reputation-tier unlock thresholds
  - `Rising Oracle` scenario pacing
- Include recommended target values for:
  - first 90 in-game days
  - first tier unlock
  - first treasury dedication
  - first crisis chain
- Export machine-readable tables if useful for content seeding.

**Files Likely Touched**

- `docs/balance/oracle-balance.xlsx`
- optional `docs/balance/README.md`

**Acceptance Criteria**

- agents can pull agreed numbers from one place
- economy and consultation tuning are internally consistent

**Validation**

- manual workbook review
- optional import/export spot-checks if generated

---

### `SEC-1` Electron, Persistence, and Mod-Surface Review Agent

**Owner:** `security_reviewer`

**Goal**

Review the project before deeper desktop, Steam, and future mod/workshop work lands.

**Primary Scope**

- Review Electron boot/preload boundaries.
- Review save-path and snapshot loading behavior.
- Review any future scenario/mod ingestion plan for:
  - untrusted JSON
  - file-system access
  - SQLite misuse
  - unsafe IPC expansion
- Produce a short remediation plan before Steam/cloud/workshop work begins.

**Files Likely Touched**

- `apps/desktop/src/main.ts`
- `apps/desktop/src/preload.ts`
- `packages/persistence/src/`
- `docs/` review output

**Acceptance Criteria**

- explicit safe-default recommendations exist before desktop scope expands
- no insecure preload/IPC pattern is introduced during later work
- any code fixes required by the review are handed back to `electron_engineer` or `worker`, not made inside the read-only review pass

**Validation**

- review memo in `docs/`
- optional checklist integrated into future PR template

---

## Wave 2 — QA / Debug Parallel Sweep

### `PLAY-1` Regression Automation Agent

**Owner:** `playwright_tester`

**Goal**

Keep the rapidly expanding feature set shippable while other agents are still landing code.

**Tasks**

- Extend smoke coverage for:
  - new economy chains
  - new consultation builder behavior
  - tier unlock progression
  - world-map interactions
  - treasury dedication / patron events
- Add one smoke flow per new debug scenario.
- Expand saved-state comparisons to cover newly added world state.
- Keep selectors and artifact output consistent.

**Suggested Deliverables**

- `packages/testkit/scripts/economy-smoke.mjs`
- `packages/testkit/scripts/worldmap-smoke.mjs`
- `packages/testkit/scripts/progression-smoke.mjs`
- updated `campaign-smoke.mjs`

**Dependencies**

- starts as soon as each feature branch exposes stable selectors/hooks

---

### `PLAYI-1` Interactive UI Debug Agent

**Owner:** `playwright_tester` using the `playwright-interactive` skill when a persistent session is needed

**Goal**

Fix the hard parts that usually break in management-game UIs: canvas hitboxes, drag/drop builders, overlays, and multi-scene interactions.

**Tasks**

- Debug prophecy drag/reorder interactions.
- Debug minimap click-to-jump.
- Debug build ghost placement and selection hit areas.
- Debug overlay + save/load + pause edge cases.
- Debug world-map to precinct transitions if added.

**Best Time To Run**

- after `DVG-2` and `DVG-4` first compile
- again after `DVG-3` world-map UI lands

---

### `SHOT-1` Visual Artifact Agent

**Owner:** `playwright_tester` with `worker` fallback using the `screenshot` skill

**Goal**

Provide visual proof when Playwright canvas captures are unreliable.

**Tasks**

- Capture headed/full-window screenshots for:
  - consultation builder layout
  - minimap visibility
  - tooltip/advisor layering
  - world-map panel readability
  - particle visibility
- Record exact bad states for handoff back to `DVG-4` and `PLAYI-1`.

**Use This Agent When**

- Playwright screenshots show black/blank canvas
- z-index or overlay issues only reproduce in a real desktop window

---

## Wave 3 — Final Merge and Stabilization

### `INT-1` Lead Integrator — Merge Train

**Owner:** `default`

**Tasks**

- Merge in this order unless blocked:
  1. `INT-0` foundation
  2. `SHEET-1` balance workbook
  3. `DVG-1` precinct economy
  4. `DVG-2` consultation + Pythia
  5. `DVG-3` politics + progression
  6. `DVG-4` UI/scene polish
  7. `ART-1` Meshy-assisted art pipeline
  8. `SEC-1` review-driven fixes
  9. `PLAY-1` final automation pass
- Resolve shared hot-file conflicts centrally.
- Standardize IDs, selectors, and scenario names.
- Run full validation.

**Final Validation Suite**

- `pnpm typecheck`
- `pnpm test`
- `pnpm --filter @the-oracle/web build`
- `pnpm smoke:precinct`
- `pnpm smoke:consultation`
- `pnpm smoke:logistics`
- `pnpm smoke:campaign`
- any new smoke scripts added by `PLAY-1`

---

## Recommended Task Board

### Must-Do First

- `INT-0` Shared foundation freeze by `default`
- `SHEET-1` Balance workbook start by `worker` using `spreadsheet`
- `ART-1` Art pipeline prep by `web_game_engineer`
- `SEC-1` Security review start by `security_reviewer`

### Parallel Block A (launch up to 3 concurrently)

- `DVG-1` Precinct economy by `gameplay_engineer`
- `DVG-2` Consultation + Pythia by `gameplay_engineer`
- `DVG-3` Politics + progression systems by `gameplay_engineer`
- `DVG-4` UI + scene readability by `web_game_engineer`
- `ART-1` Meshy-assisted art pipeline by `web_game_engineer`

### Parallel Block B

- `PLAY-1` regression automation against each branch via `playwright_tester`
- `PLAYI-1` interactive debugging for unstable UI flows via `playwright_tester`
- `SHOT-1` visual capture support where needed via `playwright_tester` or `worker`

### Suggested Launch Order Under Current Agent Limits

- Start `INT-0`, `SHEET-1`, and `SEC-1` first.
- Once `INT-0` lands, launch at most 3 implementation tracks concurrently.
- Prefer `DVG-1` + `DVG-2` + `DVG-4` as the first trio, because `DVG-3` depends on the same shared campaign-state surfaces and can start once selectors/contracts are stable.
- Start `ART-1` as soon as `INT-0` stabilizes content IDs and `DVG-4` exposes preview surfaces; it should run alongside gameplay work rather than waiting for final polish.
- Keep `PLAY-1` or `monitor` available as the fourth child slot instead of saturating all slots with feature agents.

---

## Concrete Definition of “Accomplish as Much as Possible in One Go”

If this whole plan lands successfully, the repo should end the burst with:

- 10+ meaningful buildings instead of 6
- multiple real production chains instead of the current narrow sacred-resource loop
- at least 6 omen families and materially deeper prophecy construction
- real Pythia trait effects and first roster expansion groundwork
- a visible world-map/progression layer with reputation tiers and one scenario arc
- expanded smoke coverage for new systems
- a balancing workbook and a desktop/security review ready for the next milestone

That would move the project from a **vertical slice** toward a genuine **M1/M2 hybrid prototype** aligned with the GDD.

## Notes for the Lead Integrator

- Do not let multiple agents widen shared unions independently.
- Prefer adding new files over concentrating everything back into `data.ts` and `updateDay.ts`.
- Require every branch to surface its new state in `render_game_to_text` before asking QA to automate it.
- Keep scenario/debug hooks deterministic; they are the backbone of the current repo’s velocity.
- Do not start audio/Steam/workshop implementation until precinct, consultation, and campaign loops are functionally broader.
