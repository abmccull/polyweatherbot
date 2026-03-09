# The Oracle Remaining Full-Release Execution Plan

## Purpose

This document turns the remaining gaps after the recent full-release implementation waves into an executable program.

It is a sequel to:

- `docs/parallel-implementation-plan.md`
- `docs/full-release-follow-on-plan.md`

Unlike those broader planning docs, this file is grounded in the repo as it exists now and focuses only on what is still required to reach a real full-release build.

## Verified Current State

The project is no longer a narrow vertical slice. It now has:

- seeded run setup and origin-based world generation
- first-pass world atlas and replay/timeline surfaces
- campaign progression and visible world pressure
- first-pass philosopher threats
- Sacred Record foundations and prophecy-depth summaries
- first-pass precinct sacred geography / precinct soul surfacing
- priest-politics foundations
- rival-oracle and espionage foundations
- recurring named-character scaffolding
- deterministic tests and smoke coverage for origins, world map, consultation, precinct, logistics, campaign, and replay-debug flows

The project is still not at the full-release target described in `docs/full-release-follow-on-plan.md`. The biggest confirmed gaps are:

- the world simulation is still a first pass rather than a full faction web
- prophecy, precinct, priest, and rival systems are foundations rather than long-horizon mastery systems
- there is no true late-game lifecycle yet
- lineage, burdens, endless mode, and challenge-seed loops are not playable
- content breadth is still narrow
- scenario tooling / workshop / text generation / ship systems are still missing

## Milestone Status

### `FR-A` Replayable Prototype

**Status:** effectively landed

This milestone is covered by the currently shipped origin system, world generation, first-pass world pressure systems, and deterministic validation.

### `FR-B` Living History Build

**Status:** partial

The repo has Sacred Record, priest politics, rival pressure, and first-pass sacred-geography support, but not the full long-horizon depth originally planned.

### `FR-C` Long Campaign Build

**Status:** not started in gameplay terms

There is no fully playable Age system, legendary consultation framework, decline mode, or end-of-run history artifact pipeline.

### `FR-D` Infinite Replay Build

**Status:** scaffolding only

Profile persistence types exist, but lineage, burdens, endless mode, and challenge-seed loops are not active systems.

### `FR-E` Content & Ship Build

**Status:** largely open

The repo still has a small scenario roster, no editor/workshop path, no full consequence/textgen layer, and no final polish/ship pass.

## Orchestration Model

The top-level `default` agent remains the master orchestrator.

Responsibilities:

- own launch order and dependency enforcement
- keep one slot free for integration / emergency fixes
- decide when a subsystem is stable enough for downstream work
- run combined validation after every major wave

Recommended parallel shape:

- `default` as lead integrator
- up to 3 implementation lanes in parallel
- 1 QA sidecar or monitor lane

Do not saturate all 6 agent threads with implementation work. The remaining work is too coupled for that to be efficient.

## Real Agent Roles to Use

| Agent Role | Primary Use in This Plan |
|---|---|
| `default` | orchestration, shared-core integration, merge gating, migration decisions |
| `gameplay_engineer` | deterministic systems, progression, late-game, sim logic, tuning |
| `web_game_engineer` | Phaser/React integration, in-run game surfaces, sandbox flows |
| `ui_craftsman` | large UI panels, inspectors, navigation, readability, polished overlays |
| `playwright_tester` | deterministic smoke expansion, visual evidence, repro capture |
| `worker` | bounded implementation slices, content wiring, utilities, docs/build tooling |
| `design_director` | review-only passes on readability, hierarchy, and feature clarity |
| `security_reviewer` | workshop/editor/profile/save security review before exposure |
| `electron_engineer` | desktop, Steam, workshop, and packaging integration |
| `monitor` | dev servers, smoke waits, repeated status polling |

## Shared-Core Conflict Hotspots

These surfaces should not have more than one active code-writing agent at a time without explicit coordination:

- `packages/core/src/state/gameState.ts`
- `packages/core/src/state/initialState.ts`
- `packages/core/src/simulation/updateDay.ts`
- `packages/core/src/simulation/events.ts`
- `packages/core/src/selectors/index.ts`
- `packages/content/src/schema.ts`
- `packages/content/src/data.ts`
- `packages/persistence/src/schema.ts`
- `packages/ui/src/RightSidebar.tsx`
- `packages/ui/src/WorldMapPanel.tsx`
- `packages/ui/src/WorldTimelinePanel.tsx`
- `packages/ui/src/OracleHud.tsx`
- `apps/web/src/app.tsx`
- `apps/web/src/runtime.ts`

Before each major wave, the `default` agent should split shared logic into new domain modules so parallel lanes can own narrower write sets.

## Program Structure

The remaining work should be executed in five major stages:

1. `Stage R0` Close the incomplete foundations behind `FR-B`
2. `Stage R1` Build the long-campaign lifecycle (`FR-C`)
3. `Stage R2` Build infinite replay and meta systems (`FR-D`)
4. `Stage R3` Build the content/editor/text generation layer (`FR-E` core)
5. `Stage R4` Ship hardening, visual scale, and release systems

Each stage below defines:

- the exact missing problems to solve
- recommended ownership by real local agent roles
- expected module boundaries
- validation and merge gates

---

## Stage `R0` — Finish the Incomplete Foundations

### Goal

Upgrade the newly landed systems from first-pass foundations into stable long-horizon systems that can support late game, lineage, and large content authoring.

### Why This Stage Comes First

The later phases depend on:

- richer history artifacts
- more expressive world-state outputs
- stable character memory and political simulation
- more complete precinct and prophecy surfaces

If those are still shallow, the late-game and meta layers will become wrappers around incomplete systems.

### `R0-INT` Shared-Core Completion

**Owner:** `default`

**Primary Scope**

- Split remaining monolithic logic into domain files if needed:
  - `packages/core/src/state/prophecy.ts`
  - `packages/core/src/state/ages.ts`
  - `packages/core/src/state/legacy.ts`
  - `packages/core/src/state/lineage.ts`
  - `packages/core/src/state/endless.ts`
  - `packages/core/src/state/challenges.ts`
  - `packages/core/src/state/worldHistory.ts`
  - `packages/content/src/legendaryConsultations.ts`
  - `packages/content/src/ages.ts`
  - `packages/content/src/burdens.ts`
  - `packages/content/src/scenarioPacks.ts`
- Formalize history artifact schemas:
  - Sacred Record entries
  - world-history events
  - named-character biographies
  - age summaries
  - end-of-run legacy summaries
- Wire profile persistence into app/runtime boundaries rather than leaving it as an isolated repository package.
- Add content validation and seed validation entrypoints that later editor/workshop code can reuse.
- Establish explicit test/debug hooks for new long-horizon time domains:
  - month
  - year
  - age
  - end-of-run

**Supporting Roles**

- `worker` for bounded refactors and validation tooling
- `playwright_tester` for smoke break detection if startup flows change

**Exit Criteria**

- long-horizon systems no longer widen `gameState.ts` ad hoc
- profile, legacy, and history data have stable schema paths
- later stages can add features without reopening every central file

### `R0-WORLD` Faction-Web Completion

**Owner:** `gameplay_engineer`

**Current Status**

Partially landed through first-pass politics, philosopher pressure, world atlas, and crisis chains.

**Missing Capabilities**

- alliance formation and fracture logic
- hegemon emergence over time
- memory of oracle intervention / neglect by faction and region
- revolutions and regime changes
- colonisation / client-state emergence
- philosophy / belief spread across routes and blocs
- trade-route opening and closure consequences
- pilgrim traffic and resource pricing swings

**Implementation Tasks**

- Create a dedicated faction-web state model with:
  - alliances
  - blocs
  - regime types
  - hegemon score
  - oracle memory
  - route state
  - regional philosophy vectors
- Advance the world sim on a monthly timeline and add age-aware modifiers later.
- Feed world results back into precinct play:
  - donation and envoy cadence
  - pilgrim flow
  - trade modifiers
  - scarcity pricing
  - consultation question weighting
- Replace purely card-like summaries with structured world-history event generation.
- Extend replay/timeline output to distinguish:
  - diplomatic shifts
  - revolutions
  - hegemon changes
  - oracle-driven interventions

**Likely Files**

- `packages/core/src/simulation/events.ts`
- `packages/core/src/state/campaign.ts`
- `packages/core/src/state/worldHistory.ts` (new)
- `packages/core/src/selectors/index.ts`
- `packages/content/src/factions.ts`
- `packages/content/src/politicalEvents.ts`
- `packages/ui/src/WorldMapPanel.tsx`
- `packages/ui/src/WorldTimelinePanel.tsx`

**Validation**

- add `tests/faction-web.test.ts`
- add `smoke:faction-web`
- expose structured world-history deltas in `render_game_to_text`

**Exit Criteria**

- the outside world can change materially without direct player input
- long campaigns produce distinct, readable geopolitical histories
- precinct economy and consultations respond to world-state changes

### `R0-PROPHECY` Prophecy Full-Depth Completion

**Owner:** `gameplay_engineer`

**Current Status**

Sacred Record and prophecy-depth summaries are in place, but the full interpretation and contradiction game is not.

**Missing Capabilities**

- real interpretation phase after delivery
- internal-faction interpretation branches
- multipart prophecies over multiple years
- contradiction detection across time
- rival-oracle same-question resolution
- prestige swings tied to prophecy outcomes
- searchable / queryable Sacred Record usage

**Implementation Tasks**

- Create explicit prophecy arc state:
  - prophecy ID
  - originating client
  - interpretation branches
  - follow-up obligations
  - contradiction markers
  - payoff / collapse state
- Extend consultation resolution to branch into:
  - delivered text
  - interpreted political usage
  - delayed outcome windows
- Add a Sacred Record browser that supports:
  - filtering by patron
  - filtering by domain
  - filtering by contradiction status
  - finding unresolved long-tail prophecies
- Add rival-oracle hooks so a major question can acquire competing interpretations.

**Likely Files**

- `packages/core/src/reducers/index.ts`
- `packages/core/src/state/prophecy.ts` (new)
- `packages/core/src/selectors/index.ts`
- `packages/ui/src/ConsultationOverlay.tsx`
- `packages/ui/src/SacredRecordPanel.tsx`
- `packages/ui/src/RightSidebar.tsx`
- `packages/testkit/scripts/consultation-smoke.mjs`

**Validation**

- expand `tests/prophecy-depth.test.ts`
- add `tests/prophecy-arc.test.ts`
- extend `smoke:consultation` for multipart history and contradiction cases

**Exit Criteria**

- top-tier prophecies feel like multi-year strategic commitments
- prophecy history can be browsed and used as a real planning tool

### `R0-PRECINCT` Precinct Full-Depth Completion

**Owner:** `web_game_engineer`

**Current Status**

Sacred geography and precinct-soul surfacing are present, but excavation and full sacred-site depth are not.

**Missing Capabilities**

- excavation layers
- chambers / relic finds
- site-history reveals
- legitimacy complications from discovered ruins
- stronger geography constraints on placement
- deeper precinct-soul consequences

**Implementation Tasks**

- Add an excavation subsystem with:
  - diggable tiles
  - depth layers
  - relic tables
  - reveal states
  - site-history fragments
- Add sacred-site constraints to building placement and omen quality.
- Make precinct soul a persistent system with:
  - hidden factors
  - visible summaries
  - late-game gates
  - links to legendary visit eligibility
- Add visual overlays for:
  - sacred lines
  - relic zones
  - excavated chambers
  - precinct integrity / disharmony

**Likely Files**

- `apps/web/src/game/PrecinctScene.ts`
- `packages/core/src/state/gameState.ts`
- `packages/core/src/simulation/updateDay.ts`
- `packages/ui/src/OracleHud.tsx`
- `packages/ui/src/PrecinctOverviewPanel.tsx`
- `apps/web/src/styles.css`

**Validation**

- add `tests/precinct-soul.test.ts`
- add `tests/excavation.test.ts`
- extend `smoke:precinct`
- capture headed screenshots for excavation overlays

**Exit Criteria**

- precinct layout is strategic in sacred, historical, and spatial terms
- site history and excavation create run identity, not just extra resources

### `R0-SOCIETY` Priest Politics and Espionage Completion

**Owner:** two-lane block

- `gameplay_engineer` for priest society and corruption systems
- `worker` for bounded espionage operation wiring if the write sets are separated cleanly

**Current Status**

Priest politics and rival-oracle activity exist, but neither is yet a deep social or espionage game.

**Missing Priest-System Capabilities**

- hidden secrets
- rivalries / friendships / attractions
- corruption and omen-tilt investigation
- succession contests
- elder-advisor return arcs

**Missing Rival-System Capabilities**

- true spymaster network
- more operations from the original plan
- sabotage and protection loops
- exposure and diplomatic blowback
- persistent espionage assets, not just rival activity summaries

**Implementation Tasks**

- Extend named-character memory and relationship state for priest-level secrets and rivalries.
- Add investigation actions and corruption evidence.
- Add succession state:
  - candidates
  - blocs
  - campaign events
  - transition outcomes
- Add espionage asset model:
  - agents
  - cover
  - local networks
  - intel quality
  - counterintelligence
- Implement operations from the original plan:
  - intercept rival prophecy
  - plant false omen report
  - recruit envoy secretary
  - sabotage eternal flame
  - seed philosopher
  - protect your oracle

**Likely Files**

- `packages/core/src/state/priestPolitics.ts`
- `packages/core/src/state/rivalOracles.ts`
- `packages/core/src/state/characters.ts`
- `packages/core/src/simulation/updateDay.ts`
- `packages/core/src/simulation/events.ts`
- `packages/content/src/characters.ts`
- `packages/content/src/rivalOracles.ts`
- `packages/ui/src/RightSidebar.tsx`
- `packages/ui/src/WorldMapPanel.tsx`

**Validation**

- add `tests/priest-succession.test.ts`
- add `tests/espionage.test.ts`
- extend world-map smoke for exposure/blowback states

**Exit Criteria**

- priest politics can materially destabilize the sanctuary over time
- espionage becomes a real parallel strategy layer
- rival oracles feel like long-term competitors rather than pressure tags

### Stage `R0` Launch Order

1. `R0-INT`
2. In parallel, up to three lanes:
   - `R0-WORLD`
   - `R0-PROPHECY`
   - `R0-PRECINCT`
3. After those stabilize, launch `R0-SOCIETY`
4. Finish with a combined QA pass:
   - `playwright_tester`
   - `design_director`

### Stage `R0` Milestone Gate

At the end of `R0`, the project should fully satisfy `FR-B`:

- Sacred Record is strategic and historical, not just informational
- precinct depth creates long-run site identity
- priest and rival systems are social systems, not only summary widgets
- the world sim is materially deeper and feeds the core loop

---

## Stage `R1` — Long Campaign Build

### Goal

Deliver the full campaign lifecycle: rise, maturity, transformation, decline, and remembered legacy.

### `R1-AGE` Age System

**Owner:** `gameplay_engineer`

**Missing Capability**

The game does not currently transform historically over long runs.

**Implementation Tasks**

- Add age progression model:
  - Archaic Age
  - Classical Age
  - Hellenic Age
  - Hellenistic Age
  - Roman Shadow
- Make ages change:
  - faction prominence
  - prestige expectations
  - consultation pool weighting
  - philosopher prevalence
  - trade behavior
  - rival-oracle style
- Add age summaries and transition events.
- Feed age identity into world, UI, and future legendary content.

**Likely Files**

- `packages/core/src/state/ages.ts` (new)
- `packages/core/src/simulation/updateDay.ts`
- `packages/content/src/ages.ts` (new)
- `packages/ui/src/WorldTimelinePanel.tsx`
- `packages/ui/src/OracleHud.tsx`

**Validation**

- add `tests/ages.test.ts`
- add `smoke:ages`

### `R1-LEGENDARY` Legendary Consultation Framework

**Owner:** two-lane block

- `gameplay_engineer` for scenario logic
- `ui_craftsman` for bespoke UI and presentation

**Missing Capability**

There is no authored framework for legendary visitors or multi-stage landmark consultations.

**Implementation Tasks**

- Add a scripted legendary-consultation framework with:
  - staged questions
  - age gates
  - named-character memory hooks
  - world-state prerequisites
  - branchable consequence outcomes
- Implement first legendary set:
  - Croesus
  - Themistocles
  - Spartan King
  - Philip
  - Alexander
  - Roman General
- Add special presentation rules distinct from normal envoy consultations.

**Likely Files**

- `packages/content/src/legendaryConsultations.ts` (new)
- `packages/core/src/state/characters.ts`
- `packages/core/src/reducers/index.ts`
- `packages/ui/src/ConsultationOverlay.tsx`
- `packages/ui/src/RightSidebar.tsx`

**Validation**

- add `tests/legendary-consultations.test.ts`
- add `smoke:legendary-consultation`

### `R1-LEGACY` Decline Arc and End-of-Run History

**Owner:** `gameplay_engineer`

**Missing Capability**

There is no emotionally coherent late-game contraction state or shareable end-of-run artifact.

**Implementation Tasks**

- Add decline-state entry conditions and gameplay:
  - prestige collapse
  - resource contraction
  - forced prioritization
  - comeback gambles
  - institution triage
- Add legacy score calculation and category breakdown.
- Generate end-of-run artifact:
  - major prophecies
  - major patrons
  - world turning points
  - named figures
  - precinct peak
  - decline years
- Wire artifact into profile persistence.

**Likely Files**

- `packages/core/src/state/legacy.ts` (new)
- `packages/persistence/src/ProfileRepository.ts`
- `packages/persistence/src/profileSchema.ts`
- `apps/web/src/app.tsx`
- `packages/ui/src/RightSidebar.tsx`
- future `packages/textgen/` if created earlier than Stage `R3`

**Validation**

- add `tests/legacy-score.test.ts`
- add `tests/decline.test.ts`
- add `smoke:decline`

### Stage `R1` Milestone Gate

At the end of `R1`, the project should satisfy `FR-C`:

- long runs feel historically transformed
- legendary consultations exist as authored peaks
- decline is a valid final act
- every run can end with a strong historical artifact

---

## Stage `R2` — Infinite Replay and Meta Systems

### Goal

Make the game continue to matter after one completed run.

### `R2-LINEAGE` Oracle Lineage and Meta Profile

**Owner:** `worker` with `default` integration

**Current Status**

Profile repository types exist but are not yet a true gameplay system.

**Implementation Tasks**

- Add profile boot/load/save in runtime and UI.
- Implement lineage data:
  - lineage name
  - inherited history markers
  - unlocked origins / burdens / challenge tiers
  - completed legacy records
- Add optional lineage carryover benefits or references that can be disabled.
- Surface lineage in new-run setup and end-of-run transition flow.

**Likely Files**

- `packages/persistence/src/WebProfileRepository.ts`
- `packages/persistence/src/SqliteProfileRepository.ts`
- `apps/web/src/runtime.ts`
- `apps/web/src/app.tsx`
- `packages/ui/src/RunSetupPanel.tsx`

**Validation**

- add `tests/lineage.test.ts`
- add `smoke:lineage`

### `R2-ENDLESS` Endless Mode and Burdens

**Owner:** `gameplay_engineer`

**Implementation Tasks**

- Add endless mode rule set with longer pacing and continued age cycling.
- Add burdens:
  - Long Memory
  - Silenced God
  - Rival's Revenge
  - Debt
  - Iron Oracle
  - Sceptic Age
- Support burden stacking and separate prestige tracking.
- Add Oracle Council or equivalent late-endless pressure layer.

**Likely Files**

- `packages/core/src/state/endless.ts` (new)
- `packages/content/src/burdens.ts` (new)
- `packages/core/src/state/gameState.ts`
- `packages/core/src/state/initialState.ts`
- `packages/ui/src/RunSetupPanel.tsx`

**Validation**

- add `tests/burdens.test.ts`
- add `tests/endless-mode.test.ts`
- add `smoke:endless`

### `R2-CHALLENGE` Challenge Seeds and Reproducibility

**Owner:** `playwright_tester`

**Implementation Tasks**

- Add fixed challenge-seed plumbing.
- Validate:
  - same challenge seed reproduces same starting conditions
  - burden stacks are deterministic
  - lineage-on vs lineage-off states are distinguishable and stable
- Add local scoreboard scaffolding even if online boards are deferred.

**Likely Files**

- `packages/testkit/scripts/`
- `apps/web/src/dev/SeedReplayInspector.tsx`
- `packages/persistence/src/ProfileRepository.ts`
- `apps/web/src/app.tsx`

**Validation**

- add `smoke:challenge`
- add reproducibility artifact dumps

### Stage `R2` Milestone Gate

At the end of `R2`, the project should satisfy `FR-D`:

- repeat runs build a player-specific lineage
- endless and burden runs are viable
- challenge seeds are deterministic and testable

---

## Stage `R3` — Content, Authoring, and Text Systems

### Goal

Make the game authorable and expandable at full-release scale.

### `R3-CONTENT` Scenario and Event Expansion

**Owner:** `worker` for content wiring, `gameplay_engineer` for system-sensitive authored logic

**Current Status**

The scenario roster is still tiny and the content footprint remains far below full-release scale.

**Implementation Tasks**

- Expand scenario roster toward:
  - Croesus Gamble
  - Philosopher's Athens
  - Macedonian Horizon
  - Successor Wars
  - Roman Question
  - Legendary Age
  - Unknown Oracle
- Establish content taxonomy:
  - common
  - uncommon
  - rare
  - unique
  - callback
  - chain
- Scale named-character roster toward historical breadth.
- Set explicit content production quotas per milestone instead of vague totals.

**Validation**

- content validators
- scenario snapshot tests
- event-bundle linting

### `R3-EDITOR` Scenario Editor and Workshop Foundation

**Owner:** two-lane block

- `ui_craftsman` for editor UI
- `worker` for package import/export and schema plumbing

**Implementation Tasks**

- Build internal editor first:
  - map editor
  - faction/personality editor
  - event graph editor
  - building override editor
  - origin editor
  - legendary consultation script editor
- Add package import/export format.
- Add strict validation and sandbox-safe parsing.
- Defer player-facing workshop exposure until after security review passes.

**Recommended New Package**

- `packages/editor/`

**Validation**

- add `smoke:editor`
- schema round-trip tests
- package import rejection tests

### `R3-TEXTGEN` Full Consequence and Procedural Text Generation

**Owner:** `gameplay_engineer` plus `worker` if a separate text package is introduced

**Implementation Tasks**

- Upgrade consequence resolution toward city-state-level long-horizon simulation.
- Add templated procedural text for:
  - consequence reports
  - Sacred Record commentary
  - age summaries
  - legacy histories
  - selected events
- Keep text generation deterministic and template-driven rather than opaque freeform generation.

**Recommended New Package**

- `packages/textgen/`

**Validation**

- add deterministic text snapshot tests
- add history-generation tests
- expose generated summaries in `render_game_to_text`

### Stage `R3` Milestone Gate

At the end of `R3`, the project should satisfy the core of `FR-E`:

- the content envelope can scale
- scenarios can be authored through tools
- generated history text is coherent and reproducible

---

## Stage `R4` — Performance, Visual Scale, Security, and Ship Layer

### Goal

Finish the project as a shippable product rather than a systems prototype.

### `R4-PERF` Performance, Accessibility, and Ship Systems

**Owner:** three-lane block

- `web_game_engineer` for performance and browser-side optimization
- `ui_craftsman` for accessibility and UI polish
- `electron_engineer` for desktop / Steam / workshop / cloud integration

**Implementation Tasks**

- hit scale targets for walkers, city-states, event chains, named characters, and save size
- improve accessibility:
  - text sizing
  - contrast options
  - motion reduction
  - input clarity
- add localization readiness:
  - string extraction
  - format boundaries
  - layout resilience
- add controller support
- complete desktop / Steam wiring where appropriate

**Validation**

- performance profiling captures
- long-run soak tests
- accessibility checklist pass
- desktop packaging checks

### `R4-ART` Visual Asset Factory and Readability QA

**Owner:** `worker` for tooling/docs, `ui_craftsman` for in-game integration, `design_director` for review

**Implementation Tasks**

- scale the art pipeline already documented in:
  - `docs/meshy-art-pipeline.md`
  - `docs/art-style-bible.md`
- build asset-family production batches for:
  - precinct buildings
  - terrain
  - characters
  - portraits
  - age emblems
  - legendary consultation assets
- add visual QA rules for:
  - readability
  - clutter
  - z-sort
  - minimap clarity
  - late-game screen density

### `R4-SEC` Security and Exposure Review

**Owner:** `security_reviewer`

**Implementation Tasks**

- review:
  - profile persistence
  - save import/export
  - editor package loading
  - workshop package parsing
  - Electron boundaries
- produce remediation plan before player-facing workshop exposure

### `R4-QA` Final Validation Matrix

**Owner:** `playwright_tester`

**Required Smoke Families**

- `smoke:origins`
- `smoke:faction-web`
- `smoke:consultation`
- `smoke:legendary-consultation`
- `smoke:precinct`
- `smoke:ages`
- `smoke:decline`
- `smoke:lineage`
- `smoke:challenge`
- `smoke:editor`

### Stage `R4` Milestone Gate

At the end of `R4`, the build should be feature-complete, scalable, secure enough for editor/workshop exposure, and ready for a proper release-candidate pass.

---

## Detailed Launch Map

### Block `A` — Close `FR-B`

1. `default`: `R0-INT`
2. `gameplay_engineer`: `R0-WORLD`
3. `gameplay_engineer` or `worker`: `R0-PROPHECY`
4. `web_game_engineer`: `R0-PRECINCT`
5. `playwright_tester`: world/consultation/precinct smoke expansion

After `R0-WORLD`, `R0-PROPHECY`, and `R0-PRECINCT` stabilize:

1. `gameplay_engineer`: priest-society completion
2. `worker`: espionage operation expansion if isolated enough
3. `ui_craftsman`: social/espionage panel cleanup and discoverability

### Block `B` — Build `FR-C`

1. `gameplay_engineer`: `R1-AGE`
2. `gameplay_engineer` + `ui_craftsman`: `R1-LEGENDARY`
3. `gameplay_engineer`: `R1-LEGACY`
4. `playwright_tester`: age / legendary / decline smoke families

### Block `C` — Build `FR-D`

1. `worker`: `R2-LINEAGE`
2. `gameplay_engineer`: `R2-ENDLESS`
3. `playwright_tester`: `R2-CHALLENGE`

### Block `D` — Build `FR-E` Core

1. `worker`: content validators and data import/export
2. `ui_craftsman`: editor shell
3. `gameplay_engineer`: text/consequence expansion
4. `default`: content schema integration and migration management

### Block `E` — Ship Layer

1. `web_game_engineer`: performance and browser-side scale
2. `ui_craftsman`: accessibility and final presentation
3. `electron_engineer`: desktop/Steam/workshop wiring
4. `security_reviewer`: editor/workshop/profile review
5. `playwright_tester`: final regression matrix

## Validation Requirements

Every major subsystem added in this remaining plan must provide:

- deterministic seed coverage
- targeted tests for the new state domain
- `render_game_to_text` exposure for critical state
- at least one smoke path if there is UI
- artifact dumps or screenshots for visually dense surfaces

For long-horizon systems, add one more requirement:

- a debug scenario or time-skip path that reaches the new system quickly

## Stop Conditions

Do not begin these until their prerequisites are real, not partial:

- do not start legendary consultations until age transitions and deeper prophecy arcs are stable
- do not start lineage until end-of-run history artifacts are wired
- do not expose workshop packages until editor import/export and security review are complete
- do not rely on procedural textgen before history schemas settle
- do not begin final ship polish until the gameplay/content core is stable enough that the UI will not be reworked again immediately

## Recommended Immediate Next Move

The next practical implementation wave should target completion of `FR-B`, not jump forward to late-game or workshop work.

Recommended next launch:

1. `default` closes `R0-INT`
2. `gameplay_engineer` deepens the faction web
3. `web_game_engineer` completes excavation / precinct full-depth
4. `gameplay_engineer` or `worker` deepens prophecy arcs depending on current conflict surfaces
5. `playwright_tester` expands smoke coverage in parallel

That is the shortest path to turning the current strong prototype into a stable base for the long-campaign and meta-progression phases.
