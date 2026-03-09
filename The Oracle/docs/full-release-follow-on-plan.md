# The Oracle Full Release Follow-On Plan

## Purpose

This document is the sequel to `docs/parallel-implementation-plan.md`.

It assumes the earlier implementation plan has already landed, meaning the project now has:

- a data-driven core simulation foundation
- an expanded precinct economy and logistics layer
- deeper consultation/Pythia gameplay
- a visible political world / progression layer
- stronger UI readability and smoke coverage
- baseline balancing and security review work

From that assumed baseline, this plan describes how to reach the **ideal full-release state** outlined in `TheOracle_FullRelease.md`.

## What Changes at This Stage

The previous plan moves the game from a vertical slice toward a robust prototype. This plan moves the game from a robust prototype toward a **true full-release replayable simulation**.

The key shift is:

- from **playable systems** to **interlocking systems at scale**
- from **one strong campaign loop** to **many distinct run identities**
- from **authored scenarios with limited variance** to **high-replayability worlds with authored peaks**
- from **single-run progression** to **meta-progression, legacy, and community replayability**

## Full Release Success Definition

At the end of this program, the game should support:

- meaningfully different runs through **Origins**, **world seeds**, and **procedural faction setups**
- a world that evolves independently through a proper **Faction Web**
- a full **prophecy metagame** with interpretation, multipart arcs, rival contradictions, and long-term record-keeping
- a deeply character-driven **priest / philosopher / rival / named character** layer
- a real **late game** through Ages, Legendary Consultations, Decline, and Legacy
- long-tail retention through **Lineage**, **Challenge Oracles**, **Endless Mode**, **New Game Plus**, and **Workshop**
- performance, tooling, and content throughput sufficient for a 200-hour strategy sim rather than a premium vertical slice

## Program Principles

1. **Replayability beats raw quantity.** Every major addition must create different runs, not just longer runs.
2. **The world must stop feeling like a backdrop.** External simulation must become consequential, not decorative.
3. **Every system must produce artifacts.** Sacred records, histories, lineages, seeds, save data, and end-of-run summaries are core value, not garnish.
4. **Generated systems need authored peaks.** Procedural worlds create variance; legendary consultations, decline arcs, and named figures create emotional memory.
5. **Determinism remains mandatory.** All procedural content, agent simulation, and text outputs need reproducible seeds and validation hooks.
6. **Tooling is part of the feature.** Scenario editing, content validation, prose templating, and debugging tools are required for this scale.

---

## Starting Assumption Checklist

Do not begin this plan until the previous plan has delivered at least:

- generic content/schema expansion for buildings/resources/roles
- multiple production chains and 10+ meaningful buildings
- 6+ omen families and a richer consultation layer
- a first world-map / progression implementation
- one fully playable campaign scenario with win logic
- Playwright smoke coverage that can validate precinct, consultation, logistics, politics, and progression

If any of those are not true, finish the earlier plan first.

---

## Program Structure

The full-release path should be executed in eight major phases:

1. **FR-0 — Platform Hardening & Authoring Infrastructure**
2. **FR-1 — Replayability Engine**
3. **FR-2 — Living World Simulation**
4. **FR-3 — Deep Core Systems**
5. **FR-4 — Character Society & Shadow Systems**
6. **FR-5 — Late Game, Endgame, and Historical Arc**
7. **FR-6 — Meta-Progression, Endless Mode, and Community Systems**
8. **FR-7 — Full Release Content, Scale, and Ship Layer**

Each phase has a lead dependency chain plus parallelizable workstreams for the locally available specialty sub agents.

---

## Available Local Specialty Sub Agents for This Stage

| Agent / Skill | Primary Responsibility in This Plan |
|---|---|
| `develop-web-game` | Core implementation across gameplay, UI, simulation, world-map, sandbox, and editor features |
| `playwright` | Deterministic regression automation, seed reproducibility tests, scenario/editor smoke coverage |
| `playwright-interactive` | Complex UI debugging: world map, drag/drop prophecy building, scenario editor, minimap, overlays |
| `spreadsheet` | Balance models for Origins, Ages, faction weights, burdens, economies, and score tuning |
| `security-best-practices` | Save/profile/workshop/editor review, Electron hardening, untrusted content boundaries |
| `screenshot` | Headed visual review for Phaser/canvas, editor UI, world-map readability, and artifact gaps |
| `imagegen` | Style anchors, concept batches, portraits, environment kits, and character turnaround packs that feed the asset pipeline |

## New Conflict Hotspots

As the game grows, these surfaces need explicit ownership during merge windows:

- `packages/core/src/state/`
- `packages/core/src/simulation/`
- `packages/core/src/selectors/`
- `packages/content/src/`
- `packages/persistence/src/`
- `packages/ui/src/`
- `apps/web/src/game/`
- any future `packages/textgen/` or `packages/editor/`

The lead integrator should continue splitting monolithic files as soon as a subsystem becomes shared by more than one active branch.

---

## FR-0 — Platform Hardening & Authoring Infrastructure

### Goal

Create the internal architecture needed to build the rest of full release without drowning in content and merge debt.

### Why This Must Come First

The full-release spec introduces:

- procedural world generation
- long-run historical simulation
- many more character types
- meta-progression persistence
- procedural text generation
- scenario editor / workshop ingestion

Those are infrastructure problems before they are feature problems.

### `FR-INT-0` Lead Integrator — Simulation and Data Platform

**Owner:** Lead integrator

**Tasks**

- Refactor simulation into clear domains:
  - precinct systems
  - faction/world simulation
  - characters / social graphs
  - consultation / prophecy systems
  - history / record systems
  - meta-profile systems
- Add canonical seeded RNG utilities with subsystem-specific streams.
- Introduce explicit long-horizon time domains:
  - tick
  - day
  - month
  - year
  - age
- Create stable schema packages / modules for:
  - generated worlds
  - characters
  - legacies
  - lineages
  - challenge seeds
  - editor-authored content
- Create history artifact storage models:
  - Sacred Record entries
  - named-character biographies
  - age summaries
  - end-of-run legacy histories
- Extend persistence for:
  - run save data
  - profile/meta data
  - lineage data
  - editor drafts
- Add validation / linting for content packs and seeds.
- Establish module boundaries for future Workshop content ingestion.

**Deliverables**

- modular data layer that can support generated worlds and authored events together
- profile persistence layer separate from run saves
- deterministic seed model for world, events, text generation, and challenges
- editor-safe content schema contracts

**Exit Criteria**

- downstream teams can add new generated systems without widening central unions repeatedly
- content packs and seeds can be validated automatically
- save/profile schema can evolve without breaking old runs

---

### `FR-DVG-AUTH` Content Tooling Agent

**Owner:** `develop-web-game`

**Goal**

Build lightweight internal authoring tools before content volume explodes.

**Tasks**

- Add content preview pages/dev panels for:
  - events
  - omen templates
  - word tiles
  - legendary consultation scripts
  - generated history templates
- Add schema-driven validators surfaced in dev UI or CLI.
- Add a seed inspector / replay inspector.
- Add debugging views for faction graphs, character graphs, and history timelines.

**Why This Matters**

Without this, generated content, 500+ events, and text systems will become too expensive to debug.

---

## FR-1 — Replayability Engine

### Goal

Ensure the second, fifth, and twentieth runs feel structurally different.

### Core Features from the Ideal-State Doc

- Oracle Origin System
- Procedural Political Generation
- World Seed System

### `FR-DVG-1` Origins & World Generation Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Implement the **Origin System** with at least these first-class starts:
  - Ancient Spring
  - Upstart Shrine
  - Cursed Oracle
  - War Oracle
  - God's Favourite
  - Merchant Oracle
  - Exile's Oracle
- For each Origin, support:
  - starting state overrides
  - unique mechanics
  - locked/disabled systems
  - bespoke tutorial/advisor messaging
  - score modifiers or challenge tags
- Build **Procedural World Generation** with seeded creation of:
  - city-state count
  - political climate
  - oracle density
  - economic climate
  - divine mood
  - faction personality matrices
- Add **World Seed** support to:
  - new game flow
  - save/load metadata
  - challenge mode compatibility
  - debugging UI and export/copy path

**Secondary Scope**

- Add world preview panel during game setup.
- Add seed re-roll and origin compare UI.

**Dependencies**

- `FR-INT-0`

**Exit Criteria**

- multiple new runs can begin from genuinely different conditions
- same seed recreates the same world reliably
- origins meaningfully alter the first 2–5 in-game years

---

### `FR-SHEET-1` Replayability Balance Model Agent

**Owner:** `spreadsheet`

**Scope**

- Build spreadsheets for:
  - origin difficulty curves
  - world-gen weight distributions
  - faction disposition frequencies
  - climate modifiers
  - divine mood effects
- Recommend safe ranges so generated worlds stay interesting without becoming unwinnable.

**Output**

- `docs/balance/full-release-replayability.xlsx`

---

### `FR-PLAY-1` Seed Reproducibility Agent

**Owner:** `playwright`

**Scope**

- Add deterministic smoke coverage proving that:
  - the same origin + seed produces the same starting world
  - different origins on the same seed still produce distinct opening conditions
  - world preview data matches in-run world state
- Add artifact dumps for generated world graphs.

---

## FR-2 — Living World Simulation

### Goal

Make the outside world feel alive, reactive, and dangerous even when the player does nothing.

### Core Features from the Ideal-State Doc

- Faction Web
- deeper rival presence
- philosopher threat as a persistent system
- cultural spread, revolutions, colonisation, hegemony drift

### `FR-DVG-2` Faction Web Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Replace the first-pass monthly politics layer with a fuller **Faction Web** simulation:
  - alliance formation and breakdown
  - hegemon emergence over long timescales
  - memory of oracle help / neglect
  - revolutions and government changes
  - colonisation and new client emergence
  - belief / philosophy spread across connected states
- Build world-sim outputs that affect precinct play:
  - trade route openings/closures
  - pilgrim traffic shifts
  - envoy frequency and stakes
  - resource pricing changes
- Expose a proper graph/world-map view rather than only side cards.

**Secondary Scope**

- Add “world pressure” summaries for advisors.
- Add historical replay/timeline view for major geopolitical changes.

**Dependencies**

- `FR-INT-0`
- `FR-DVG-1`

**Exit Criteria**

- the world evolves credibly without player input
- prophecies clearly act as one influence among many, not the only driver
- long campaigns produce distinct world histories

---

### `FR-DVG-3` Philosopher Threat Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Implement philosophers as named slow-burn adversaries or converts.
- Add philosopher archetypes:
  - Rationalist
  - Moralist
  - Rival Theologian
  - Political Agitator
- Add spread mechanics:
  - followers
  - council influence
  - local credibility erosion
  - philosophical drift among priests / youth / patrons
- Add counters:
  - public debate
  - scholarly rebuttals
  - acts of public integrity
  - private conversion attempts
- Add failure mode where a city-state can become permanently hostile to oracle authority.

**Dependencies**

- world character framework from `FR-INT-0`
- first Faction Web implementation

**Exit Criteria**

- philosopher pressure can reshape a run over decades
- philosophers are characters, not one-off events

---

## FR-3 — Deep Core Systems

### Goal

Turn the consultation loop and precinct layout into long-horizon mastery systems rather than only tactical loops.

### Core Features from the Ideal-State Doc

- prophecy interpretation phase
- multipart prophecies
- competing/contradicting oracles
- Sacred Record
- excavation
- sacred geography
- precinct soul

### `FR-DVG-4` Prophecy Full-Depth Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Add the **Interpretation Phase** after delivery:
  - internal factions within the consulted state
  - multiple interpretations of one prophecy
  - optional diplomatic nudge if favour is high enough
- Add **Multi-Part Prophecies** for top-tier patrons:
  - linked narrative commitments over multiple consultations
  - contradiction detection across years
  - major payoff / major collapse states
- Add **Competing Oracles** on shared questions:
  - rival prophecy intelligence
  - same-client same-question resolution logic
  - prestige swings from contradiction outcomes
- Add **Sacred Record** system:
  - codex UI
  - searchable past prophecies
  - historical reinterpretation events
  - retroactive credibility wins/losses

**Dependencies**

- prior consultation expansion from the earlier plan
- rival oracle scaffolding from politics / shadow systems

**Exit Criteria**

- major prophecies become multi-year commitments
- players can browse and meaningfully leverage their own history

---

### `FR-DVG-5` Precinct Full-Depth Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Add **Excavation System**:
  - diggable sub-tiles / chambers / relic finds
  - unique subterranean content
  - legitimacy complications from discovered history
  - natural feature reveals
- Add **Sacred Geography** rules:
  - terrain/site-specific bonuses
  - line-of-sight placement constraints
  - omen-quality effects based on geography
- Add **Precinct Soul** hidden/visible progression system:
  - long-run sacred integrity factors
  - reputation beyond faction credibility
  - legendary visit gating and late-game event hooks

**Secondary Scope**

- add antiquarian priest role if not already present
- expand site inspection UI and overlays for terrain bonuses

**Dependencies**

- geometry/content groundwork from earlier precinct plan

**Exit Criteria**

- layout optimization is no longer only logistical; it is also sacred/spatial
- the precinct acquires a long-memory identity beyond economy stats

---

## FR-4 — Character Society & Shadow Systems

### Goal

Make the oracle feel inhabited by people with agendas, secrets, loyalties, and schemes.

### Core Features from the Ideal-State Doc

- priest relationship web
- succession politics
- bribery/corruption pressure
- espionage web
- recurring named historical figures

### `FR-DVG-6` Priest Character Web Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Add priest personality/state model:
  - ambitions
  - loyalties
  - faith / philosophy leanings
  - rivalries / attractions / friendships
  - hidden secrets
- Add event generation from priest relationships.
- Add corruption and omen-tilt detection/investigation.
- Add head-priest succession contests with persistent internal factions.
- Add Elder Advisor return characters and long-tail internal politics.

**Dependencies**

- character framework from `FR-INT-0`

**Exit Criteria**

- internal politics can destabilize or strengthen a run over decades
- priests stop behaving like pure worker slots

---

### `FR-DVG-7` Espionage & Rival Oracle Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Build full spymaster / agent network model.
- Add espionage operations from the ideal-state doc:
  - Intercept Rival Prophecy
  - Plant False Omen Report
  - Recruit Envoy's Secretary
  - Sabotage the Eternal Flame
  - Seed a Philosopher
  - Protect Your Oracle
- Add rival oracles as active entities with:
  - precinct status
  - patrons
  - sabotage capability
  - visibility/intel uncertainty
- Add exposure and diplomatic blowback systems.

**Dependencies**

- Faction Web
- priest/character groundwork

**Exit Criteria**

- shadow play is a meaningful parallel game, not just a button press
- rivals feel like long-term competitors rather than event tokens

---

### `FR-DVG-8` Named Character Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Add recurring named-figure framework.
- Support:
  - multi-visit memory
  - character aging / role changes
  - cross-event callbacks
  - trust/fear/hostility accumulation
- Seed first roster of recurring archetypes before historical figure expansion.

**Why This Exists Separately**

Named characters cut across philosophers, generals, merchants, priests, and legendary visitors. They need shared scaffolding.

---

## FR-5 — Late Game, Endgame, and Historical Arc

### Goal

Deliver a full campaign lifecycle: rise, maturity, transformation, decline, and remembered legacy.

### Core Features from the Ideal-State Doc

- Age System
- Legendary Consultation Events
- Decline Arc
- Legacy Score and generated history

### `FR-DVG-9` Age System Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Implement five historical Ages:
  - Archaic Age
  - Classical Age
  - Hellenic Age
  - Hellenistic Age
  - Roman Shadow
- Add transition logic that transforms:
  - faction prominence
  - philosophical pressure
  - consultation types
  - economy/trade conditions
  - prestige expectations
- Add age-aware music/UI/theme hooks for later polish.

**Dependencies**

- world-gen + faction web
- scenario pacing system

**Exit Criteria**

- long sandbox runs feel historically transformed, not just numerically escalated

---

### `FR-DVG-10` Legendary Consultation Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Build authored multi-stage consultation framework.
- Implement at least the first legendary set:
  - Croesus of Lydia
  - Themistocles
  - Spartan King
  - Philip of Macedon
  - Alexander
  - Roman General
- Add history-aware reactions and map-level consequence branches.

**Dependencies**

- age system
- named character framework
- deeper prophecy system

**Exit Criteria**

- legendary consultations feel materially different from standard envoy loops

---

### `FR-DVG-11` Decline & Legacy Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Add **Decline Arc** game mode/state:
  - contraction decisions
  - mothballing buildings
  - triaging relationships
  - comeback gambles
  - lean-world succession decisions
- Add **Legacy Score** calculation and breakdown categories.
- Generate an end-of-run **History of the Oracle** artifact:
  - major prophecies
  - major patrons
  - disasters
  - peak precinct form
  - final years
- Add share/export path for legacy summaries.

**Dependencies**

- Sacred Record
- age system
- named character history

**Exit Criteria**

- late-game decline is a valid, emotionally coherent play state
- every run ends with a strong historical artifact

---

## FR-6 — Meta-Progression, Endless Mode, and Community Systems

### Goal

Make the game persist beyond a single run and support long-term player identity.

### Core Features from the Ideal-State Doc

- Oracle Lineage
- Challenge Oracle
- Endless Oracle Mode
- New Game Plus Burdens
- Oracle Council

### `FR-DVG-12` Lineage & Meta Profile Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Implement persistent **Oracle Lineage** profile data.
- Support inheritances such as:
  - legendary prophecy references
  - architectural memory / traditional designs
  - Pythia bloodlines
  - rival history
  - scholarly inheritance
- Add optionality so players can ignore meta if desired.

**Dependencies**

- profile persistence
- end-of-run legacy artifact generation

**Exit Criteria**

- multiple runs begin to form a player-specific mythology

---

### `FR-DVG-13` Endless Mode & Burdens Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Implement **Endless Oracle Mode** with:
  - long-century pacing
  - dynamic scaling
  - Oracle Council endgame layer
  - age transitions across long horizons
  - player-authored personal chronicle hooks
- Implement **New Game Plus / Burdens**:
  - Long Memory
  - Silenced God
  - Rival's Revenge
  - Debt
  - Iron Oracle
  - Sceptic Age
- Support burden stacking and separate prestige tracking.

**Dependencies**

- lineage/meta profile
- age system
- philosopher/rival systems

**Exit Criteria**

- advanced players have replay goals after finishing authored scenarios

---

### `FR-PLAY-2` Challenge Oracle / Seed Leaderboard Agent

**Owner:** `playwright` for validation, lead integrator for implementation plumbing

**Primary Scope**

- Add deterministic “same seed, same setup” challenge validation.
- Build smoke and state comparison for:
  - fixed monthly challenge seed
  - burden stack reproducibility
  - lineage-on / lineage-off new run states

**Note**

If online leaderboards are not in scope yet, ship challenge seeds and local leaderboard scaffolding first.

---

## FR-7 — Full Release Content, Scale, and Ship Layer

### Goal

Fill the content envelope, finalize long-tail systems, and make the game ship-ready.

### Core Features from the Ideal-State Doc

- full scenario roster
- 500+ events
- 60+ named historical characters
- scenario editor / workshop
- procedural text generation
- full consequence engine
- performance and scale targets

### `FR-DVG-14` Scenario & Event Content Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Expand scenario roster to include:
  - Croesus Gamble
  - Philosopher's Athens
  - Macedonian Horizon
  - Successor Wars
  - Roman Question
  - Legendary Age
  - Unknown Oracle
- Build toward the 500+ event target with:
  - common/uncommon/rare/unique taxonomy
  - chain events
  - callback events
- Populate 60+ named character entries using the shared framework.

**Dependencies**

- core systems from FR-1 through FR-6
- authoring tools

---

### `FR-DVG-15` Scenario Editor & Workshop Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Implement internal editor foundation first:
  - map editor
  - faction/personality editor
  - event graph editor
  - building override editor
  - origin editor
  - legendary consultation script editor
- Then expose player-facing scenario editor.
- Add import/export package format and validation.
- Add Workshop integration hooks after security review.

**Dependencies**

- content schema hardening from `FR-INT-0`
- security review

**Exit Criteria**

- dev team can build scenarios through tools rather than raw file editing
- workshop-safe content format exists

---

### `FR-DVG-16` Full Consequence & Textgen Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Upgrade to the full **Consequence Engine**:
  - lightweight agent simulation per city-state
  - decision pressure modified by prophecies
  - persistent history of choices and outcomes
- Implement **Procedural Text Generation** for:
  - consequence reports
  - Sacred Record commentary
  - age summaries
  - end-of-run legacy histories
  - certain event descriptions
- Ensure generated text feels authored through template / fragment systems rather than flat mad-libs.

**Dependencies**

- world simulation
- history artifact system
- named character system

**Exit Criteria**

- the game can generate rich, coherent histories from simulation output

---

### `FR-DVG-17` Performance, UI Polish, and Ship Systems Agent

**Owner:** `develop-web-game`

**Primary Scope**

- Hit full-release scale targets for:
  - 400 walkers
  - 12 city-states
  - 40 active event chains
  - 100+ named characters
  - save size and memory ceilings
- Add final ship systems that are still needed from the broader release scope:
  - final UI polish
  - accessibility options
  - controller support
  - localization readiness
  - full audio integration
  - Steam achievements/cloud/workshop wiring where appropriate

**Dependencies**

- most gameplay systems already stabilized

**Exit Criteria**

- build is feature-complete, scalable, and shippable

---

### `FR-ART-1` Meshy and Visual Asset Factory Agent

**Owner:** `develop-web-game`

**Primary Scope**

- turn the initial Meshy-assisted art pipeline into a full-release asset factory for precinct, world-map, character, fauna, and scenario assets
- scale concept generation, Meshy proxy generation, fixed-camera conversion, and in-engine QA into a repeatable production lane
- ensure visual throughput grows with the full-release content plan instead of becoming the schedule bottleneck

**Tasks**

- expand `docs/meshy-art-pipeline.md` into a full-production playbook
- lock and version the art bible
- build reusable asset-family rules for buildings, terrain, flora, characters, fauna, and consultation ornamentation
- add conversion tooling for fixed camera presets, transparent renders, export naming, and palette/readability checks
- add visual QA suites for readability, z-sort, minimap legibility, and late-game clutter

**Dependencies**

- the first implementation plan's `ART-1`
- stable building and character family schemas from precinct and social-system work

**Exit Criteria**

- the project has a repeatable full-release art-production pipeline
- visual production can run in parallel with content/system implementation without constant reinvention
- art readiness is no longer the main gating factor for full-release scope

---

## Parallelization Map by Phase

### After `FR-INT-0` lands

Run in parallel:

- `FR-DVG-1` Origins & World Generation
- `FR-DVG-AUTH` Content Tooling
- `FR-SHEET-1` Replayability balance model
- `FR-PLAY-1` seed reproducibility automation

### After `FR-DVG-1` reaches stable world generation

Run in parallel:

- `FR-DVG-2` Faction Web
- `FR-DVG-3` Philosopher Threat
- `FR-DVG-5` Precinct Full Depth
- `FR-DVG-8` Named Character foundation

### After `FR-DVG-2` / `FR-DVG-8` stabilize

Run in parallel:

- `FR-DVG-4` Prophecy Full Depth
- `FR-DVG-6` Priest Character Web
- `FR-DVG-7` Espionage & Rival Oracle
- `FR-PLAYI-1` interactive UI debug pass

### After the above systems are integrated

Run in parallel:

- `FR-DVG-9` Age System
- `FR-DVG-10` Legendary Consultations
- `FR-DVG-11` Decline & Legacy
- `FR-DVG-12` Lineage & Meta Profile

### Final long-tail parallel block

- `FR-DVG-13` Endless Mode & Burdens
- `FR-DVG-14` Scenario & Event Content
- `FR-DVG-15` Scenario Editor & Workshop
- `FR-DVG-16` Full Consequence & Textgen
- `FR-DVG-17` Performance/Ship systems
- `FR-ART-1` Meshy and visual asset factory
- `FR-PLAY-2` challenge reproducibility validation
- `FR-SEC-1` workshop/editor/profile review
- `FR-SHOT-1` visual artifact and UI audit

---

## Specialized QA / Support Tracks

### `FR-PLAYI-1` Interactive UI Debug Agent

**Owner:** `playwright-interactive`

**Primary Targets**

- origin/world setup flow
- world-map interactions
- prophecy interpretation UI
- Sacred Record browsing
- scenario editor interactions
- workshop content preview/import

### `FR-SEC-1` Workshop / Profile / Editor Security Agent

**Owner:** `security-best-practices`

**Primary Targets**

- editor import/export safety
- workshop package parsing
- profile/meta save boundaries
- Electron preload / file access surface
- untrusted content validation rules

### `FR-SHOT-1` Visual Audit Agent

**Owner:** `screenshot`

**Primary Targets**

- world-map readability
- scenario editor usability
- Sacred Record presentation
- late-game clutter at higher walker counts
- decline / legacy summary screens

---

## Recommended Milestones

### Milestone FR-A — Replayable Prototype

Must include:

- Origins
- procedural worlds
- world seed support
- first Faction Web pass
- first philosopher system pass

### Milestone FR-B — Living History Build

Must include:

- Sacred Record
- excavation / sacred geography / soul
- priest relationship web
- espionage operations
- rival oracle pressure

### Milestone FR-C — Long Campaign Build

Must include:

- Age System
- legendary consultations
- decline arc
- legacy score/history output

### Milestone FR-D — Infinite Replay Build

Must include:

- lineage
- endless mode
- burdens / NG+
- challenge seeds

### Milestone FR-E — Content & Ship Build

Must include:

- expanded scenario roster
- 500+ event library target trajectory
- named historical character breadth
- scenario editor / workshop foundation
- procedural text generation
- performance, accessibility, audio, and release polish

---

## Validation Requirements for Full Release Development

Every major new subsystem must provide:

- deterministic seed coverage
- targeted test or simulation harness
- `render_game_to_text` exposure for critical state
- at least one smoke or replay test if it has UI
- screenshot or headed validation if it changes the map/editor/overlay surface significantly

Add likely new smoke families over time:

- `smoke:worldgen`
- `smoke:origins`
- `smoke:faction-web`
- `smoke:legendary-consultation`
- `smoke:ages`
- `smoke:decline`
- `smoke:lineage`
- `smoke:editor`

---

## What Not to Start Too Early

Do **not** start these before their prerequisites are stable:

- Scenario Editor / Workshop before content schemas and import validation are hardened
- Challenge Oracle before seeds are fully deterministic
- Lineage before end-of-run legacy artifacts are stable
- NG+ Burdens before philosopher/rival/debt systems are mature enough to support them
- full text generation before history artifact structures are finalized
- large content-library writing before internal tools and validators exist

---

## Expected End State if This Plan Lands

If this follow-on plan is completed after the earlier implementation plan, The Oracle should end up with:

- strong run-to-run variance through origins, seeds, climates, and personalities
- a living geopolitical simulation that can surprise the player over centuries
- prophecy gameplay that remains strategically rich even after mastery
- priests, philosophers, rivals, and historical figures that create human drama
- a meaningful late game instead of a flat prestige plateau
- an end-of-run artifact players want to share
- meta systems that make repeated runs personally meaningful
- enough content tooling and workshop support to sustain the game beyond official authored content

That is the path from a highly promising systems-heavy prototype to the ideal full-release state described in `TheOracle_FullRelease.md`.

## First Move After the Earlier Plan Finishes

The first concrete move should be:

1. land `FR-INT-0`
2. immediately start `FR-DVG-1`, `FR-DVG-AUTH`, `FR-SHEET-1`, and `FR-PLAY-1` in parallel
3. start `FR-ART-1` as soon as the first post-slice art bible and preview surfaces exist
4. do not begin late-game, workshop, or meta-progression work until seeded replayability and world simulation are solid

That sequencing gives the rest of full release a stable spine instead of building a huge amount of content on top of prototype assumptions.
