# The Oracle — Comprehensive UI/UX Audit Report

**Date:** 2026-03-06
**Methodology:** 6 design expert frameworks (Tufte, Norman, Chen, Ueda, Koster, Wroblewski) applied across 10 screen/menu clusters. Research benchmarks drawn from Frostpunk, Anno 1800, Civ VI, RimWorld, Cities: Skylines, Banished, Factorio, Against the Storm, Foundation, and Old World.

---

## Executive Summary

| Screen / System | Score | Critical Issues |
|----------------|-------|-----------------|
| Top Bar | 5.5/10 | Save/Load/New Run don't belong; no trend grouping; too many items |
| Build Palette (Left Sidebar) | 5.8/10 | Hidden below 1200px; 190px too narrow; no affordability indicator |
| Precinct Map & Minimap | 5.0/10 | Minimap not dismissible; 3 stat naming schemes; not interactive |
| Bottom Strip | 6.0/10 | Low information hierarchy; duplicates top bar data; tiny font |
| Overlay Trigger Strip | 6.5/10 | Missing 3 overlays (Espionage/Legacy/Lineage); Unicode icons |
| Oracle Overlay | 4.5/10 | "Junk drawer" — 8 unrelated sections in one panel |
| World Overlay | 5.5/10 | Excavation buried at position 8/8; too much vertical scroll |
| Stores Overlay | 7.5/10 | Strongest data overlay; minor density issues |
| Priests Overlay | 6.5/10 | Good structure; decree buttons lack feedback |
| Consultation Overlay | 8.0/10 | Best panel — clear 3-column layout, good feedback loops |
| Run Setup Panel | 7.0/10 | Well-structured; origin cards slightly cramped |
| Espionage/Legacy/Lineage | 1.0/10 | **Completely unreachable via mouse** — keyboard-only ghost panels |
| Visual Design (Cross-cutting) | 6.4/10 | 27 font sizes; emoji icons; strong thematic cohesion |
| **Weighted Average** | **5.6/10** | |

---

## CRITICAL BUGS (Severity: Blocking)

### BUG-1: Three Ghost Overlays — Unreachable Features
- **Espionage** (E key), **Legacy** (L key), and **Lineage** (no trigger at all) have full panels implemented but NO buttons in `OverlayTriggerStrip.tsx`
- Mouse-only users cannot access these features at all
- Keyboard shortcuts are undiscoverable — no tooltip, no help screen
- **Impact:** Three entire game systems are invisible to most players
- **Fix:** Add to OverlayTriggerStrip or redesign navigation

### BUG-2: Build UI Hidden Below 1200px
- CSS `@media (max-width: 1200px)` sets `.left-sidebar { display: none }`
- Building is the core gameplay loop — this makes the game unplayable on smaller screens
- **Impact:** Game-breaking on laptops, tablets, and small monitors
- **Fix:** Responsive build menu (bottom drawer or radial menu at smaller sizes)

### BUG-3: Three Stat Naming Schemes for Same Data
- **Tile hover:** Approach / Sanctity / Shelter / Strain (4 stats)
- **Building info card:** Approach / Sanctity / Strain (3 stats, missing Shelter)
- **Walker info card:** Approach / Anchors / Strain (different name for Sanctity)
- **Precinct Soul (bottom strip):** Uses "Soul" + "Integrity" (different vocabulary entirely)
- **Impact:** Players cannot build mental models when the same data has different names
- **Fix:** Unify to one naming scheme everywhere

---

## Screen-by-Screen Analysis

---

### 1. Top Bar (header.top-bar)
**Score: 5.5/10**

**What's there:**
- Title "The Oracle"
- Clock (Y/M/D + Season icon + Season name)
- Age badge
- Resource pills (ritual/currency group + divider + food/trade group) with trend arrows
- Action buttons: New Run, Save, Load, Speed 0-3

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Tufte (Data-ink) | 6/10 | Resource pills show icon+label+value+trend — good density. But "The Oracle" title wastes ~80px for zero information. Season shown twice (icon in top bar + text in bottom strip). |
| Norman (Affordance) | 5/10 | Speed buttons look identical to Save/Load — no visual distinction between game actions and system actions. Active speed has no strong highlight. |
| Chen (Flow) | 4/10 | Save/Load/New Run are system actions competing with gameplay controls. Opening "New Run" mid-game is a dangerous action with no confirmation. |
| Koster (Chunking) | 6/10 | Resource pills are grouped with a divider — good. But the right side mixes system actions (Save/Load) with game actions (Speed) in one cluster. |
| Wroblewski (Disclosure) | 5/10 | No hover-for-detail on resources. No click-for-breakdown. Single layer only. |
| Ueda (Subtractive) | 5/10 | "The Oracle" title, "New Run", "Save", "Load" are ever-present but used <1% of play time. |

**What would make it 10/10:**
1. Remove Save/Load/New Run from top bar — put behind Escape menu (universal convention: RimWorld, Frostpunk, Civ VI)
2. Remove "The Oracle" title — the player knows what game they're playing
3. Add resource trend tooltips: hover shows breakdown (income sources, drains)
4. Group speed controls with clock (they're temporally related)
5. Add resource grouping labels or visual clusters (3-5 per group)
6. Show season once, in one place
7. Add critical alert indicator (flashing icon when something needs attention)

---

### 2. Build Palette (Left Sidebar)
**Score: 5.8/10**

**What's there:**
- Collapsible sidebar (190px, toggle with B key or arrow button)
- "Select" tool button at top
- 7 categories with collapsible headers (emoji icon + text label)
- Building items show: art thumbnail, color swatch, name, cost (gold)
- Locked buildings show lock icon + unlock tier
- Click shows BuildingDetail popup (cost, durability, upkeep, recipes, effects, storage, staffing)

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Norman (Affordance) | 6/10 | Building items look clickable. But no color distinction between "can afford" and "cannot afford" — player must mentally compare gold amount to cost. Locked items show lock icon — good. |
| Krug (Don't Make Me Think) | 5/10 | 7 categories all open by default = long scroll. Category headers use emoji icons that render differently per platform. A new player cannot find "build a temple" quickly. |
| Tufte (Data-ink) | 6/10 | Cost shown inline — good. But 190px width forces text truncation on longer building names. Art thumbnails are tiny and hard to distinguish. |
| Chen (Flow) | 4/10 | Building detail popup appears over the sidebar, not beside it. Player loses context of the category list when inspecting. |
| Rams (Less but better) | 5/10 | Upkeep shows "0.003 olive_oil/day" — too precise, underscore in name, unit not obvious. |
| Wroblewski (Disclosure) | 7/10 | Three tiers: list view > click for detail > recipes/effects within detail. Good structure. |

**What would make it 10/10:**
1. Move to bottom toolbar (Anno 1800 pattern) or make it wider (280px minimum)
2. Show affordability: green = can build, red/dim = insufficient gold
3. Replace emoji icons with consistent SVG/icon font
4. Building detail popup should appear to the RIGHT of the sidebar, not over it
5. Add search/filter when building count exceeds 20
6. Responsive: at <1200px, switch to bottom drawer instead of hiding entirely
7. Show resource cost as icon+number pairs, not text strings
8. Category collapse state should persist across sessions
9. Show ghost preview cost deduction in the resource bar during placement

---

### 3. Precinct Map & Minimap
**Score: 5.0/10**

**What's there:**
- Phaser canvas (game-host div) filling the viewport area
- PrecinctOverviewPanel rendered as `.minimap-overlay` in bottom-right (~310x400px)
- Site Inspection panel on tile hover (Approach/Sanctity/Shelter/Strain + tone badge)
- Building info card on building selection (condition bar, stats, recipes, assign priest)
- Walker info card on walker selection (role, state, carrying, fatigue/skill/radius)
- Advisor banner (severity-colored, positioned at top-center)
- Event log overlay (last 3 events, bottom-left area)
- Consultation banner when envoy approaching

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Ueda (Subtractive) | 3/10 | Minimap permanently occupies ~310x400px. Cannot be dismissed, resized, or repositioned. In a game about a single precinct, the minimap's value is questionable — the player can see the whole precinct. |
| Norman (Mapping) | 4/10 | Three different stat vocabularies for the same tile data (see BUG-3). Info cards show building IDs like "assignmentBuildingId" — raw data, not human-readable names. |
| Tufte (Data-ink) | 5/10 | Condition bar is good data visualization. But "Tile 12,5" is developer data, not player-meaningful location naming. Site Inspection stats lack units or context. |
| Chen (Flow) | 6/10 | Info cards appear on selection without mode-switching — good. But the card overlaps the viewport, potentially hiding the selected entity. |
| Koster (Chunking) | 6/10 | Building info card has reasonable grouping. But it shows too many stats at once (condition + priests + jobs + site reading + approach/sanctity/strain + recipes + stored resources = 8+ groups). |

**What would make it 10/10:**
1. Make minimap dismissible (or replace with a compass/orientation indicator)
2. Make minimap clickable to pan the main camera
3. Unify stat names across all contexts
4. Show building names instead of IDs in walker info
5. Add meaningful tile names or zone labels instead of "Tile 12,5"
6. Info card should not overlap the selected entity — position dynamically
7. Progressive disclosure: basic info visible, details on hover/expand
8. Event log should have urgency tiers (color-coded) instead of uniform styling

---

### 4. Bottom Strip
**Score: 6.0/10**

**What's there:**
- Season (icon + text) — duplicated from top bar
- Current tool mode + hint text
- Population counts (Pop, Priests, Pilgrims)
- Job count, Carrier count, Carrier strain
- Precinct Soul + tone badge

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Tufte (Data-ink) | 6/10 | Compact data presentation. But season is shown twice (also in top bar). Population is shown here AND in Oracle overlay. |
| Norman (Feedback) | 7/10 | Tool mode display is good feedback for current state. Hint text guides the player. |
| Koster (Chunking) | 5/10 | 11 data items in a single horizontal strip with identical styling. No visual grouping beyond dividers. |
| Wroblewski (Disclosure) | 4/10 | Single layer only — no hover for details, no click to expand. "Strain 45" means nothing without context (is 45 good? bad?). |
| Ueda (Subtractive) | 5/10 | "Soul" and "Integrity" labels are thematic but opaque. New players won't know what "Precinct Soul" means. |

**What would make it 10/10:**
1. Remove season (shown in top bar clock already)
2. Add visual grouping: Tool Zone | Population Zone | Economy Zone | Health Zone
3. Add hover tooltips with context (e.g., "Carrier Strain 45/100 — carriers are moderately fatigued")
4. Color-code values: green/yellow/red based on thresholds
5. Population counts should link to the Oracle overlay's Population section
6. Make the strip height configurable or auto-hide when not needed

---

### 5. Overlay Trigger Strip (Right Edge)
**Score: 6.5/10**

**What's there:**
- 40px vertical strip with 5 buttons: Oracle, World, Stores, Priests, Record
- Unicode symbol icons with text labels
- Active state highlighting
- Keyboard shortcut shown in tooltip
- ARIA labels and aria-pressed

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Norman (Affordance) | 7/10 | Buttons look clickable, active state is clear, aria attributes present. |
| Koster (Chunking) | 5/10 | Only 5 of 8 overlays are accessible. Espionage, Legacy, Lineage are missing. |
| Chen (Flow) | 6/10 | One-click access to overlays — good. But opening an overlay covers the game viewport. |
| Tufte (Data-ink) | 6/10 | Unicode icons are small and visually inconsistent across platforms. |

**What would make it 10/10:**
1. Add ALL overlays (Espionage, Legacy, Lineage) — currently ghost panels
2. Replace Unicode icons with SVG icons for consistency
3. Show keyboard shortcut badge ON the button face (not just tooltip)
4. Add notification dots when an overlay has pending/urgent items
5. Consider grouping: Gameplay (Oracle/Stores/Priests) | World (World/Record) | Systems (Espionage/Legacy/Lineage)

---

### 6. Oracle Overlay
**Score: 4.5/10**

**What's there:**
8 sections in one scrollable panel:
1. Reputation (tier + progress bar)
2. Treasury (progress bar + gold invested)
3. Population (4 role counts + priest roster)
4. Pythia (6 stat bars + Rest/Purify buttons)
5. Envoy (conditional — approaching envoy card)
6. Sacred Record (last 3 prophecies + 2 chronicle entries)
7. Consequence Tracker (sub-panel)
8. Legendary Consultations (sub-panel)
9. Trade Offers (list with Buy buttons)

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Krug (Don't Make Me Think) | 3/10 | This is a "junk drawer" — 8+ unrelated sections. A player looking for trade offers must scroll past reputation, treasury, population, pythia, envoy, chronicle, and consequences. |
| Koster (Chunking) | 3/10 | 8 sections far exceeds the 4+/-1 chunking guideline. No tabs, no grouping, no way to navigate within the panel. |
| Wroblewski (Disclosure) | 4/10 | Everything is shown at once. No progressive disclosure within the panel. |
| Tufte (Data-ink) | 6/10 | Individual components are well-designed (stat bars, credibility pips, condition badges). The problem is organizational, not visual. |
| Chen (Flow) | 4/10 | Player must scroll extensively to find what they need. No way to jump to a section. |
| Ueda (Subtractive) | 4/10 | Many sections could be their own overlays or tabs within this overlay. |

**What would make it 10/10:**
1. Split into tabs or sub-panels: "Status" (Reputation/Treasury/Population) | "Pythia" (stats/rest/purify) | "Prophecies" (Record/Consequences/Legendary) | "Trade"
2. Or: break into separate overlays — Pythia gets her own overlay, Trade gets its own
3. Add section navigation (anchor links or tab strip within the overlay)
4. Pythia card should be its own dedicated screen given her centrality to gameplay
5. Trade offers should be accessible from a dedicated button, not buried in Oracle

---

### 7. World Overlay
**Score: 5.5/10**

**What's there:**
- Atlas/Replay tab toggle at top
- Atlas view contains (in order):
  1. WorldMapPanel
  2. AgePanel (current age + history)
  3. Factions (sorted list with credibility pips, agenda, ally/rival, favour/debt/trade)
  4. PhilosopherThreatsPanel
  5. CrisisChainsPanel
  6. CharactersPanel
  7. RivalOraclesPanel
  8. ExcavationPanel (position 8/8 — deeply buried)
- Replay view: WorldTimelinePanel

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Krug (Don't Make Me Think) | 4/10 | Excavation — a major interactive system with commands — is buried at scroll position 8/8. A player who doesn't scroll to the very bottom will never find it. |
| Koster (Chunking) | 4/10 | 8 sub-panels in one scrollable column. Same junk-drawer problem as Oracle overlay. |
| Wroblewski (Disclosure) | 5/10 | Atlas/Replay toggle is good top-level disclosure. But within Atlas, everything is flat. |
| Norman (Mapping) | 6/10 | Faction cards are well-structured (name, credibility, agenda, relations). CredibilityPips visual encoding is intuitive. |
| Chen (Flow) | 5/10 | No way to jump directly to Excavation, Characters, or Rivals without scrolling. |

**What would make it 10/10:**
1. Add sub-tabs within Atlas: "Factions" | "Characters" | "Excavation" | "Threats"
2. Excavation should be its own overlay or at minimum its own tab — it has interactive commands
3. Section navigation or anchor links within the panel
4. Faction cards should be expandable (collapsed = name + credibility, expanded = full details)
5. Consider breaking WorldMapPanel out as a persistent mini-view rather than embedded in scroll

---

### 8. Stores Overlay
**Score: 7.5/10**

**What's there:**
- Resource inventory with seasonal badges
- Production chains visualization
- Carrier summary

**Expert Analysis:**
- Strongest data overlay after Consultation
- Good information density
- Seasonal badges add context without clutter
- Production chain visualization follows Anno 1800's pattern (input > process > output)
- Minor issues: carrier details could show more at-a-glance stats

**What would make it 10/10:**
1. Add resource trend graphs (sparkline charts showing last 30 days)
2. Color-code resources by sufficiency (green surplus, red deficit)
3. Add "bottleneck" indicator highlighting the weakest link in production chains
4. Carrier fatigue should have visual progress bars, not just numbers

---

### 9. Consultation Overlay
**Score: 8.0/10** (Gold Standard)

**What's there:**
- Full-screen 3-column layout
- Left: Supplicant info + Pythia stats
- Center: Tile builder (5 categories), depth readout, omen reports, assembled prophecy, scaffold
- Right: Reading quality meters (Clarity/Value/Risk/Depth), guidance text, risk warnings
- Footer: instruction hint + "Utter the Prophecy" button

**Expert Analysis:**

| Principle | Score | Assessment |
|-----------|-------|------------|
| Norman (Feedback) | 9/10 | Real-time score meters update as tiles are placed. Depth readout changes color by band. Guidance text adapts to current state. Risk warnings appear contextually. |
| Chen (Flow) | 8/10 | The game pauses during consultation — player has space to think. Tile selection is fast (click to toggle). |
| Tufte (Data-ink) | 8/10 | Score meters are compact and information-dense. Omen cards show reliability + semantics efficiently. |
| Koster (Chunking) | 8/10 | 5 tile categories chunk 20+ tiles into manageable groups. 3-column layout separates input (left), action (center), feedback (right). |
| Wroblewski (Disclosure) | 7/10 | Scaffold section reveals prophecy structure progressively. But omen reports are all shown at once — could benefit from expand/collapse. |

**What would make it 10/10:**
1. Add keyboard shortcuts for tile categories (1-5)
2. Drag-and-drop tile reordering in the assembled prophecy zone
3. Omen reports should be collapsible (show headline only, expand for detail)
4. Add a "preview consequence" tooltip showing likely faction reactions
5. Save/Load buttons don't belong here — should be in Escape menu

---

### 10. Run Setup Panel
**Score: 7.0/10**

**What's there:**
- Two-column layout: Origin selection (left) | World Preview (right)
- Seed input with Randomize button
- Origin cards with climate, divine mood, oracle density, faction mix, tags
- World preview with metric cards, faction mix chips, pressures, history, optional map
- "Begin the Omen Cycle" start button

**Expert Analysis:**
- Well-structured for a complex setup screen
- Origin cards provide meaningful differentiation
- World preview gives strategic information before committing
- Metric cards with progress bars are good data visualization

**What would make it 10/10:**
1. Origin cards are slightly cramped — show only name + summary at first, expand on click
2. Add difficulty rating to each origin
3. Seed field should show a "copy seed" button for sharing
4. Add a "Quick Start" button that skips to gameplay with default settings
5. Should be accessible from Escape menu, not a top bar button

---

### 11. Espionage / Legacy / Lineage Panels
**Score: 1.0/10** (Discoverability)

**What's there:**
- EspionagePanel: Agent roster, operation launcher, recruit form. Fully implemented.
- LegacyPanel: Score preview, decline warning, legacy artifact display. Fully implemented.
- LineagePanel: Run history, burden selection, challenge seeds. Fully implemented.

**The Problem:**
- Espionage has keyboard shortcut E but NO button in OverlayTriggerStrip
- Legacy has keyboard shortcut L but NO button in OverlayTriggerStrip
- Lineage has NEITHER keyboard shortcut NOR button — completely unreachable
- No help screen or shortcut reference exists to discover these

**What would make it 10/10:**
1. Add all three to OverlayTriggerStrip
2. Or: restructure navigation entirely (see recommendations below)

---

## Cross-Cutting Issues

### Typography: 27 Font Sizes (Should be 7-8)
Identified distinct font sizes in use: ~7px, ~9px, 0.7rem, 0.72rem, 0.75rem, 0.78rem, 0.8rem, 0.82rem, 0.85rem, 0.88rem, 0.9rem, 0.95rem, 1rem, 1.05rem, 1.1rem, 1.15rem, 1.2rem, 1.25rem, 1.3rem, 1.5rem, 1.8rem, 2rem, 2.5rem, and various pixel sizes.

**Impact:** No visual rhythm or hierarchy. Player cannot distinguish "section title" from "label" from "detail text" because sizes are separated by 0.02-0.05rem increments.

**Fix:** Implement a modular scale (Major Third 1.25 ratio):
- `--text-xs: 0.64rem` (10px — minimum, labels only)
- `--text-sm: 0.8rem` (13px — secondary info)
- `--text-base: 1rem` (16px — body text)
- `--text-lg: 1.25rem` (20px — section headers)
- `--text-xl: 1.563rem` (25px — panel titles)
- `--text-2xl: 1.953rem` (31px — page headers)
- `--text-3xl: 2.441rem` (39px — hero text, consultation title)

### Iconography: Emoji-Only (Platform Dependent)
All icons use Unicode emoji: buildings, resources, seasons, categories, navigation.

**Impact:** Emoji render differently on macOS, Windows, Linux, and mobile. Size, color, and alignment vary. Some emoji don't render at all on older systems.

**Fix:** Replace with a consistent SVG icon set (Lucide, Phosphor, or custom). Keep thematic cohesion with Greek-inspired line icons.

### Thematic Cohesion: 8/10 (Strongest Dimension)
The ancient Greek oracle aesthetic is deeply consistent: warm earth tones, gold accents, papyrus-like textures, thematic terminology (Pythia, Precinct Soul, Sacred Record, Omen).

**This is the game's strongest design asset and must be preserved through all redesign work.**

---

## Prioritized Recommendations

### P0 — Fix Before Anything Else
1. **Add Espionage/Legacy/Lineage to OverlayTriggerStrip** — 3 feature systems are invisible
2. **Make build menu responsive** — game is unplayable below 1200px
3. **Unify stat naming** — 3 naming schemes for the same data

### P1 — High Impact Redesign
4. **Move Save/Load/New Run to Escape menu** — frees top bar, follows universal convention
5. **Split Oracle overlay into tabs** — Status | Pythia | Prophecies | Trade
6. **Give Excavation its own overlay or prominent tab** — buried at scroll position 8/8
7. **Implement type scale** — replace 27 sizes with 7 tokens
8. **Replace emoji with SVG icons** — consistent rendering across platforms

### P2 — Medium Impact Improvements
9. **Add notification dots to trigger strip** — show when overlays have pending items
10. **Make minimap dismissible** — or replace with a toggle-able compass widget
11. **Add resource hover tooltips** — breakdown of income/drain sources
12. **Color-code resource sufficiency** — green/yellow/red in top bar and stores
13. **Add section navigation within large overlays** — tab strips or anchor links
14. **Show keyboard shortcuts on button faces** — not just in tooltips
15. **Building affordability indicator** — green/red in build palette based on current gold

### P3 — Polish
16. **Add trend sparklines to resource pills** — show 30-day history on hover
17. **Building detail popup positioning** — render to right of sidebar, not over it
18. **Event log urgency tiers** — color-code by severity, not uniform styling
19. **Add help/keyboard reference screen** — accessible via ? key
20. **Animate overlay open/close** — <300ms slide or fade transitions

---

## Proposed Navigation Architecture (Redesign Direction)

### Current State
```
Top Bar: Title | Clock | Age | Resources | New Run | Save | Load | Speed
Left Sidebar: Build Palette (collapsible)
Right Strip: Oracle | World | Stores | Priests | Record (5 of 8 overlays)
Bottom Strip: Season | Tool | Pop | Jobs | Soul
Keyboard-only: Espionage (E) | Legacy (L)
Unreachable: Lineage
```

### Proposed State
```
Top Bar: Clock+Speed | Resources (grouped, with trends) | Alert Indicator
Escape Menu: Save | Load | New Run | Settings | Help
Bottom Bar: Build Categories (8 tabs, Anno-style) | Tool Mode
Right Strip: Oracle | World | Stores | Priests | Record | Espionage | Legacy | Lineage
Bottom Strip: (removed — data moved to top bar hover tooltips)
```

**Key Changes:**
- Build menu moves from 190px left sidebar to bottom toolbar (scales to all resolutions)
- System actions (Save/Load/New Run) move to Escape menu
- All 8 overlays accessible from trigger strip
- Bottom strip data promoted to top bar or moved to hover tooltips
- Minimap becomes toggle-able (M key)

---

## Scoring Summary by Expert Framework

| Framework | Current Score | Key Violation |
|-----------|-------------|---------------|
| Tufte (Data-ink) | 6/10 | Title wastes space; season shown twice; no resource breakdowns |
| Norman (Affordance) | 5/10 | 3 ghost overlays; no affordability indicators; ID-based labels |
| Chen (Flow) | 5/10 | Excavation buried; overlays cover viewport; Save/Load compete with gameplay |
| Ueda (Subtractive) | 5/10 | Minimap permanent; bottom strip low-value; too many always-visible items |
| Koster (Chunking) | 5/10 | Oracle overlay has 8 sections; 27 font sizes; bottom strip 11 items |
| Wroblewski (Disclosure) | 5/10 | Most panels are single-layer (no hover > click > detail progression) |
| **Average** | **5.2/10** | |

---

---

## Gameplay Screen Deep-Dive (Agent 12 Results)

### Consultation Overlay — Refined Score: 7.5/10
**Strengths:** Tile grammar teaching compositional prophecy structure. Real-time ScoreMeter feedback. Scaffold display showing structural skeleton with missing/charged states. "Utter the Prophecy" gold button is thematically perfect.

**New Issues Found:**
- No animation on tile selection — tiles appear/disappear without movement, killing the ritual feeling
- No cancel/close button — player cannot back out without delivering
- Omen reliability shown as dim 0.72rem text — should be a visual indicator (bar/color) since it's decision-critical
- Omen semantics (domain/target/polarity) are raw data with no explanation for new players
- Pythia stats don't explain their effect on the prophecy being crafted

**10/10 reference:** Cultist Simulator's card-verb interaction with animated sliding, glow on valid combinations, ticking timers

### Save/Load System — Score: 5.0/10
**Strengths:** Autosave fires silently on day changes. Error messages are genuinely helpful.

**Critical Issues:**
- No pause/Escape menu exists — Save/Load appear in TWO locations (top bar + consultation overlay) but neither is standard
- Only one manual save slot ("slot-1") — no slot selection, naming, or preview
- No confirmation dialog before loading (immediately overwrites current state)
- No autosave visual indicator — player never knows when game was last saved
- SeedReplayInspector is always visible as a top-right button — developer tool at gameplay prominence level
- No keyboard shortcuts for save/load despite having shortcuts for everything else

**10/10 reference:** RimWorld's Escape menu (Resume, Save with slots, Load with previews, Settings, Quit)

### Legacy / End-of-Run — Score: 6.5/10
**Strengths:** Rich artifact with epitaph, turning points, named figures, major prophecy details. Declining phase shows actionable guidance.

**Issues:**
- No ceremony between gameplay and legacy screen — no dramatic pause, no tallying animation
- Thriving phase (most common state) shows almost nothing — just a bare ScoreBar
- No "Start New Run" button on terminal legacy screen — player sees epitaph but has no clear next action
- No comparison to previous runs

**10/10 reference:** Hades' animated score tallying, each category revealing one at a time with sound

### Lineage / Meta-Progression — Score: 7.0/10
**Strengths:** Carryover bonuses show concrete next-run benefits. Challenge seeds with completion tracking add structure. Run history with epitaphs creates a "hall of records."

**Issues:**
- No visual unlock tree — origins and burdens are flat lists with no progression visualization
- Burden selection has no toggle UI in this panel (dispatch accepts burdens but no compose mechanism)
- No "Start New Run" integration — purely informational, forcing separate Run Setup navigation
- No milestone indicators on the lineage score bar (what unlocks at 1000? 2500?)

**10/10 reference:** Rogue Legacy's family tree, Hades' mirror showing greyed-out future upgrades

### Sacred Record — Score: 8.0/10
**Strengths:** Filter row (All/Awaiting/Resolved). Rich detail view with depth card, meta grid, quoted prophecy, scaffold breakdown, interpretation section. Status chips color-coded by outcome.

**Issues:**
- No prophecy arc visualization connecting related prophecies
- Flat column layout — should be master-detail side-by-side
- No search or text filtering for 30+ prophecies
- No contradiction detection UI despite data model support

**10/10 reference:** Obra Dinn's deduction journal with set-completion animations

### Legendary Consultations — Score: 6.0/10
**Strengths:** Clear three-state lifecycle. Available consultations show full reward breakdown.

**Issues:**
- "Complete Stage" button has no visible gating logic — appears always clickable
- No visual progress indicator (should be stage dots or progress bar, not just "Stage 2/4" text)
- No atmosphere — legendary figures rendered identically to ordinary envoys
- Reward display is raw text, not styled chips

**10/10 reference:** Hades' boss encounters with portrait art, dramatic framing, narrative interludes

---

## Design Standards Documents Produced

The following reference documents were created by research agents:

1. **`tasks/hud-design-research-report.md`** — HUD layout, top bar, bottom strip, navigation standards (691 lines)
2. **`research/build-menu-ui-design-standards.md`** — Build menu organization, placement UX, info panels (538 lines)
3. **`docs/ui-ux-spatial-display-standards.md`** — Minimap design, tile inspection, data overlays, site panels (~600 lines)
4. **`docs/ui-ux-design-standards.md`** — Color system, typography scale, iconography, component language, atmospheric design (~582 lines)

### Key Standards from Design Documents

**Typography (from docs/ui-ux-design-standards.md):**
- Major Third (1.250) ratio type scale: 7 levels from 28px display to 10px caption
- Georgia serif for thematic text, system-ui sans-serif for numeric data (Civ VI principle: "theme the words, clarify the math")
- 10px absolute minimum floor — current `0.58rem` labels are below this

**Color (from docs/ui-ux-design-standards.md):**
- Missing semantic tokens: `--amber` for warnings, `--purple` for sacred/divine
- WCAG contrast audit of every existing token pair with minimum ratios
- Palette ceiling: 5 semantic + 3 accent hues (matching Old World/CK3 discipline)

**Iconography (from docs/ui-ux-design-standards.md):**
- Filled-silhouette pictographic style inspired by Greek pottery black-figure technique
- Four sizes: 12px badges, 16px inline, 24px buttons, 32px map overlays
- Rule: labels for first encounters, icon-only after learning, tooltips always

**Atmospheric Design (from docs/ui-ux-design-standards.md):**
- Core principle: UI should feel like it was *made by* an ancient Greek craftsman, not like it is *depicting* ancient Greece
- Strict "one ornament per screen" budget
- Motion timing: "stone settling into place" easing
- Material reference table: bronze/parchment/stone/terracotta mapped to CSS

**Minimap (from docs/ui-ux-spatial-display-standards.md):**
- 10% viewport size ceiling
- Decision framework: bottom-right for builders, bottom-left for RTS
- When minimaps become wasted space: too small a world, not interactive, vertically-oriented
- Three layers: base terrain, toggle-able symbols, context-dependent overlays

**Tile Inspection (from docs/ui-ux-spatial-display-standards.md):**
- Two-tier hover/click model (Civ VI and Factorio)
- Stat display: labeled bars for 3-6 variables, small multiples grid for 4-8
- Concrete wireframes for approach/sanctity/shelter/strain stats

---

## Revised Final Scores (All 12 Agents Complete)

| Screen / System | Initial | Refined | Delta | Notes |
|----------------|---------|---------|-------|-------|
| Top Bar | 5.5 | 5.5 | -- | Unchanged |
| Build Palette | 5.8 | 5.8 | -- | Unchanged |
| Precinct Map & Minimap | 5.0 | 5.0 | -- | Spatial standards doc confirms minimap issues |
| Bottom Strip | 6.0 | 6.0 | -- | Unchanged |
| Overlay Trigger Strip | 6.5 | 6.5 | -- | Unchanged |
| Oracle Overlay | 4.5 | 4.5 | -- | Unchanged |
| World Overlay | 5.5 | 5.5 | -- | Unchanged |
| Stores Overlay | 7.5 | 7.5 | -- | Unchanged |
| Priests Overlay | 6.5 | 6.5 | -- | Unchanged |
| Consultation Overlay | 8.0 | 7.5 | -0.5 | No cancel button, no tile animation, Save/Load misplaced |
| Run Setup Panel | 7.0 | 7.0 | -- | Start button in header confirmed as issue |
| Sacred Record | -- | 8.0 | new | Strongest information architecture |
| Save/Load System | -- | 5.0 | new | No Escape menu, single slot, no confirmation |
| Legacy / End-of-Run | -- | 6.5 | new | Rich data, no ceremony |
| Lineage / Meta | -- | 7.0 | new | Good systems, no visual progression |
| Legendary Consultations | -- | 6.0 | new | No atmosphere, no stage gating UI |
| Espionage/Legacy/Lineage (discoverability) | 1.0 | 1.0 | -- | Still completely unreachable |
| Visual Design (cross-cutting) | 6.4 | 6.4 | -- | Type scale + icon standards now documented |
| **Weighted Average** | **5.6** | **5.8** | +0.2 | New screens pull average up slightly |

### Composite by Expert Framework (Revised)

| Framework | Score | Strongest Screen | Weakest Screen |
|-----------|-------|-----------------|----------------|
| Tufte (Data-ink) | 6.2/10 | Sacred Record (8) | Oracle Overlay (4.5) |
| Norman (Affordance) | 5.3/10 | Consultation (8) | Ghost Overlays (1) |
| Chen (Flow) | 5.5/10 | Consultation (8) | Save/Load (4) |
| Ueda (Subtractive) | 5.0/10 | Stores (7.5) | Oracle Overlay (4) |
| Koster (Chunking) | 5.4/10 | Consultation (8) | Oracle Overlay (3) |
| Wroblewski (Disclosure) | 5.8/10 | Sacred Record (8) | Bottom Strip (4) |
| **Average** | **5.5/10** | | |

---

*Full report compiled from 12 parallel agents (6 research + 6 analysis). Source files analyzed: OracleHud.tsx, OverlayTriggerStrip.tsx, OracleOverlayPanel.tsx, WorldOverlayPanel.tsx, StoresOverlayPanel.tsx, PriestsOverlayPanel.tsx, ConsultationOverlay.tsx, BuildPalette.tsx, RunSetupPanel.tsx, EspionagePanel.tsx, LegacyPanel.tsx, LineagePanel.tsx, RecordOverlayPanel.tsx, PrecinctOverviewPanel.tsx, styles.css, app.tsx. Research benchmarks in tasks/hud-design-research-report.md, research/build-menu-ui-design-standards.md, docs/ui-ux-spatial-display-standards.md, and docs/ui-ux-design-standards.md.*
