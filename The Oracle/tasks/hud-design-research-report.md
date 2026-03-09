# Game HUD & Navigation Design Research Report
## Gold Standards, Design Principles, and Scoring Rubric

*Research compiled for: Simulation/Management/Building game UI analysis*
*Date: 2026-03-06*

---

## Table of Contents
1. [Design Expert Frameworks](#1-design-expert-frameworks)
2. [Overall Screen Layout & Viewport Allocation](#2-overall-screen-layout--viewport-allocation)
3. [Top Bar Design](#3-top-bar-design)
4. [Bottom Strip / Event Feed](#4-bottom-strip--event-feed)
5. [Navigation & Menu Systems](#5-navigation--menu-systems)
6. [Consolidated Scoring Rubric](#6-consolidated-scoring-rubric)

---

## 1. Design Expert Frameworks

These six frameworks form the theoretical foundation for all scoring criteria in this document. Every design decision should be evaluable through at least one of these lenses.

### 1.1 Edward Tufte --- Data-Ink Ratio & Information Density
**Core Principle:** Maximize the ratio of data-carrying pixels to total pixels. Every UI element should earn its screen space. Remove non-data-ink (decorative borders, redundant labels, background textures that compete with data). Prefer high data density when the data reveals patterns.

**Applied to Game UI:**
- Resource bars should show numbers, trends, and capacity without ornamental chrome
- Tooltips should layer additional data, not repeat what is already visible
- Background colors on HUD elements should be functional (dark to recede, bright to alert), never decorative
- Anno 1800 exemplifies this: dark HUD colors recede, bright pop-ups demand attention

**Measurable Test:** Can you remove any pixel from the HUD without losing information? If not, the data-ink ratio is optimal.

### 1.2 Don Norman --- Affordances, Mapping, Feedback
**Core Principle:** Every interactive element must visually communicate what it does (affordance), map logically to its effect (mapping), and confirm the result of interaction (feedback). Users should never wonder "did that work?" or "what does this do?"

**Applied to Game UI:**
- Buttons must look clickable; sliders must look draggable
- Spatial layout of controls should mirror spatial layout of their effects (e.g., camera controls matching screen direction)
- Every click produces immediate visual + audio feedback
- Banished's icon similarity problem (herbalist vs gatherer) is a textbook affordance failure

**Measurable Test:** Can a new player correctly guess the function of each HUD element within 3 seconds of seeing it?

### 1.3 Jenova Chen --- Flow State & Friction Reduction
**Core Principle:** UI should never break the player's flow state. The interface must be invisible during peak engagement. Friction (unnecessary clicks, modal dialogs, confirmation prompts for routine actions) pulls players out of flow. Dynamic difficulty of UI complexity should match player skill.

**Applied to Game UI:**
- Routine actions require minimal clicks (1-2 max)
- UI elements fade or minimize during moments of high engagement
- No forced interruptions for low-priority information
- Progressive revelation of advanced UI as player mastery increases
- Frostpunk's radial menus pausing time is a flow-preserving decision --- complex choices happen outside the pressure loop

**Measurable Test:** Count clicks-to-action for the 10 most common player tasks. Gold standard is <=2 for each.

### 1.4 Fumito Ueda --- Subtractive Design
**Core Principle:** Start with everything, then remove until only the essential core remains. If a UI element doesn't directly serve the core experience, it should be eliminated. Communicate through absence --- what you don't show is as important as what you show.

**Applied to Game UI:**
- Default HUD state should be minimal; information appears on demand
- Decorative elements are removed unless they reinforce thematic identity
- If data can be conveyed by the game world itself (diegetic UI), prefer that over HUD overlays
- Banished's toggleable UI elements exemplify this: everything off by default, added as needed

**Measurable Test:** What percentage of HUD elements can be hidden without the player being unable to play? Higher percentage = better subtractive design.

### 1.5 Raph Koster --- Cognitive Load & Chunking
**Core Principle:** The brain processes information in chunks of 4 +/- 1 items. Fun comes from pattern recognition and mastery. UI that exceeds cognitive load capacity (too many simultaneous elements) destroys both fun and comprehension. Group related information into digestible chunks.

**Applied to Game UI:**
- Top bar resource groups should contain 3-5 items per visual cluster
- Menu categories should have 5-9 top-level items maximum
- Sub-menus should contain no more than 7-9 items before requiring further subdivision
- Information hierarchy must be clear: primary > secondary > tertiary
- Old World's "tooltips within tooltips within tooltips" is a chunking failure

**Measurable Test:** Count the number of distinct information items visible simultaneously. If >12 on a single screen region, cognitive overload is likely.

### 1.6 Luke Wroblewski --- Progressive Disclosure & Focus
**Core Principle:** Show only the most important information and actions first. Reveal secondary and tertiary options as the user demonstrates need. Constraints force better design --- designing for the smallest useful surface area first ensures nothing unnecessary survives.

**Applied to Game UI:**
- New players see simplified HUD; complexity unlocks with game progression
- Hover/click reveals detail layers rather than showing everything at once
- Context-sensitive UI that shows relevant tools based on what the player is doing
- RimWorld's architect menu becomes overwhelming with mods precisely because it lacks progressive disclosure

**Measurable Test:** Does the game have at least 3 distinct information density levels the player can access? (Glance > Inspect > Deep Dive)

---

## 2. Overall Screen Layout & Viewport Allocation

### 2.1 Gold Standard: Frostpunk (1 & 2)

**What they do right:**
- Viewport dominance: The game world occupies approximately 85-90% of the screen at default zoom
- HUD elements are translucent and hug the edges, creating a natural "frame" that reinforces the crater/circle aesthetic
- Persistent elements (resource bar, temperature gauge) are compact and positioned at screen top
- Contextual elements (build menus, council decisions) appear on demand and can be dismissed
- The radial menu system overlays the center of the screen temporarily, then disappears entirely
- Time automatically pauses when complex menus open, separating "observe" mode from "decide" mode

**Design Principles Applied:**
- **Ueda (Subtractive):** Default state shows almost nothing. Information reveals on interaction.
- **Chen (Flow):** Time-pause on menu open means UI never competes with real-time gameplay pressure.
- **Tufte (Data-ink):** The temperature gauge is a single visual element encoding temperature, trend, AND danger threshold.

**Runner-up: Banished**
- Even more minimal: the base HUD is a single compact toolbar
- All information panels are opt-in and repositionable
- Proves that a city builder can function with <5% screen dedicated to persistent UI

### 2.2 Screen Real Estate Guidelines

| Category | Recommended % | Rationale |
|----------|--------------|-----------|
| Persistent HUD (always visible) | 5-10% | Top bar + minimap + speed controls only |
| Contextual panels (on demand) | 15-25% when open | Build menus, detail panels, overlays |
| Game viewport (minimum) | 75-85% | Must remain dominant at all times |
| Viewport during full UI | 60% minimum | Even with all panels open, world must be visible |

### 2.3 Persistent vs Contextual UI Decision Framework

**Make it persistent if:**
- The player checks it more than once per minute (resources, population, time)
- Missing the data causes immediate failure (temperature in Frostpunk, food in Banished)
- It requires <50 pixels of width/height to display

**Make it contextual if:**
- It's only relevant during specific activities (build mode, diplomacy, trade)
- It contains more than 3 data points requiring interpretation
- It would overlap with other contextual panels

**Make it an overlay if:**
- It modifies the viewport itself (heat maps, traffic, happiness)
- It's a temporary analytical lens the player applies and removes
- Cities: Skylines' info views are the gold standard for overlays

### 2.4 Fixed Sidebars vs Floating Panels vs Overlays

| Pattern | Best For | Example | Risk |
|---------|----------|---------|------|
| Fixed sidebar | Persistent tool palettes with <10 items | Original Pharaoh's left-side panel | Steals viewport permanently |
| Floating panels | Detail views, multi-panel workflows | Anno 1800's production chains | Can occlude critical game area |
| Overlays | Data visualization on the game world | Cities: Skylines info views | Can obscure building details |
| Radial menus | Quick selection from categorized options | Frostpunk build menu | Limited to ~8-12 items per ring |
| Drawer/tray | Large item lists, inventories | RimWorld's architect tabs | Can feel disconnected from world |

### 2.5 Anti-Patterns to Avoid

1. **Permanent sidebar syndrome:** Fixed panels that steal 20%+ of screen width at all times (classic Pharaoh's weakness)
2. **Modal everything:** Forcing full-screen menus for simple decisions that could be inline
3. **Viewport tug-of-war:** Multiple panels competing for the same screen region simultaneously
4. **Information desert:** Going so minimal that players must memorize menu locations (early Dwarf Fortress)
5. **Resolution amnesia:** UI that doesn't scale or reposition for different screen sizes

### 2.6 Scoring Criteria (10/10 Implementation)

| Criterion | Weight | 10/10 Standard |
|-----------|--------|----------------|
| Viewport dominance | 25% | Game world visible >=80% at rest, >=60% with panels open |
| Persistent UI economy | 20% | <=8% of screen used by always-visible elements |
| Contextual reveal | 20% | Panels appear/dismiss in <300ms with clear trigger |
| Spatial consistency | 15% | HUD elements never shift position unexpectedly |
| Resolution scaling | 10% | Identical usability at 1080p, 1440p, and 4K |
| Customizability | 10% | Player can toggle/reposition at least core HUD groups |

---

## 3. Top Bar Design

### 3.1 Gold Standard: Civilization VI

**What they do right:**
- Horizontal resource strip across the full top of the screen
- Resources grouped by type: yields (science, culture, gold, faith) | strategic | luxury
- Each resource shows: icon + current stockpile + per-turn change ("+12")
- Trend indicators via the +/- numbers tell the player if they're growing or shrinking
- Expand/collapse per resource group --- players can lock/unlock sections
- Hover reveals breakdown tooltip (where income comes from, what's consuming it)
- Compact: approximately 30-40px tall, spanning the screen width

**Design Principles Applied:**
- **Tufte (Data-ink):** Icon + number + trend in ~60px per resource. No wasted space.
- **Koster (Chunking):** Resources grouped into 3-4 visual clusters of 3-5 items each.
- **Wroblewski (Progressive Disclosure):** Glance = icon + number. Hover = full breakdown. Click = management screen.

**Runner-up: Frostpunk**
- Temperature gauge is a masterclass in single-element information density
- Encodes: current temperature, trend direction, danger thresholds, and time-to-crisis in one visual arc
- Resources displayed as icon + count, compact and scannable

### 3.2 Managing 10+ Resources Without Overwhelming

**Strategy 1: Visual Grouping (Civ VI approach)**
- Cluster resources into 3-4 semantic groups separated by dividers or spacing
- Each group has 2-5 items
- Player's eye scans left-to-right across groups, not individual items

**Strategy 2: Priority Tiering (Anno 1800 approach)**
- Show the 4-6 most critical resources persistently
- Secondary resources appear on hover over a "more" indicator or in a expandable tray
- Population tier resources (different citizen classes) appear contextually when managing that tier

**Strategy 3: Contextual Relevance (Frostpunk approach)**
- Show only resources relevant to the current game phase
- Early game: food, wood, coal. Late game: add steam cores, automaton parts
- Resources appear in the top bar as they become relevant

**Strategy 4: Dashboard Drill-Down (Pharaoh: A New Era approach)**
- Top bar shows aggregate health indicators (green/yellow/red)
- Overseer menus provide detailed breakdowns per category
- The top bar acts as an early warning system, not an exhaustive display

### 3.3 Clock/Speed Controls Placement

**Convention (nearly universal):**
- Speed controls in the top-right corner (RimWorld, Banished, Cities: Skylines)
- OR integrated with the minimap cluster in bottom-right (Civilization VI)
- Display: current speed indicator (1x/2x/3x) + pause button + date/time
- Keyboard shortcuts (1/2/3 for speed, Space for pause) are expected and must be available

**Gold standard: RimWorld**
- Three speed buttons + pause in top-right
- Current date displayed adjacent
- Keyboard shortcuts 1-4 mapped directly
- Small footprint, always accessible, never in the way

### 3.4 What Belongs in the Persistent Top Bar

**MUST be persistent:**
- Primary currency/resource counts (3-6 items)
- Population count or primary unit metric
- Current game speed / pause indicator
- Turn counter or date/time
- Critical alerts indicator (flashing icon when something needs attention)

**SHOULD be contextual (hover or expand):**
- Resource breakdowns and trends
- Secondary/tertiary resource counts beyond the top 6
- Per-turn calculations
- Trade balance information

**MUST NOT be in the top bar:**
- Navigation/build tools (belong on bottom or side)
- Settings/save/load (belong in pause menu or corner)
- Event log or notifications (belong on bottom or side)
- Tutorial or help text (belongs in tooltips or dedicated panel)

### 3.5 Anti-Patterns to Avoid

1. **Resource bar overflow:** Showing 15+ resources in a single row with no grouping (Old World tendency)
2. **Icon-only resources:** Icons without numbers force players to hover every icon to understand state
3. **Numbers without context:** "Gold: 1,453" means nothing without knowing if that is good, bad, or trending
4. **Static displays:** Resources that show current value but no rate-of-change
5. **Inconsistent positioning:** Resource order that changes based on game state, breaking muscle memory

### 3.6 Scoring Criteria (10/10 Implementation)

| Criterion | Weight | 10/10 Standard |
|-----------|--------|----------------|
| Information density per pixel | 20% | Each resource shows icon + value + trend in <=80px width |
| Visual grouping | 20% | Resources clustered into 3-5 groups with clear separators |
| Progressive disclosure | 20% | 3 layers: glance (bar), inspect (hover), deep-dive (click) |
| Scannability | 15% | Player can assess all critical resources in <2 seconds |
| Trend communication | 15% | Every resource shows direction of change (up/down/stable) |
| Height economy | 10% | Top bar height <=40px at 1080p (3.7% of vertical space) |

---

## 4. Bottom Strip / Event Feed

### 4.1 Gold Standard: Frostpunk (Event + Narrative Integration)

**What they do right:**
- Events appear as contextual notifications that are tied to the game world
- Critical events (deaths, generator failure) use distinct visual language (red, larger, persistent)
- Routine events (building complete, research done) use subtle toasts that auto-dismiss
- Events link to their source --- clicking navigates the camera to the relevant location
- The "Book of Laws" and council events use dedicated modal panels, not the event feed
- Environmental storytelling: the UI itself (frost on edges, darkening) communicates crisis without text

**Design Principles Applied:**
- **Norman (Feedback):** Every game action has a proportional notification response.
- **Chen (Flow):** Routine events don't interrupt; critical events demand attention proportional to urgency.
- **Ueda (Subtractive):** Environmental storytelling through UI chrome replaces explicit notifications.

**Runner-up: Civilization VI**
- Right-side notification queue with stacking
- Each notification is an icon-button that jumps to the relevant game element
- Queue auto-advances on interaction, creating a natural "to-do list" flow
- Notifications persist until acknowledged, preventing missed events

### 4.2 Event Log Patterns

| Pattern | Best For | Example | Strengths | Weaknesses |
|---------|----------|---------|-----------|------------|
| **Scrolling feed** | Games with continuous small events | Dwarf Fortress, RimWorld | Complete history, grep-able | Overwhelming, hard to prioritize |
| **Notification queue** | Turn-based or low-frequency events | Civilization VI | Forces acknowledgment | Can create backlogs |
| **Toast system** | Routine confirmations | Cities: Skylines | Non-intrusive, auto-dismiss | Easy to miss important info |
| **Priority-tiered hybrid** | Complex sims with mixed urgency | Frostpunk | Best of all worlds | Harder to implement well |

**Recommended: Priority-tiered hybrid**
- **Critical (red):** Persistent banner, requires acknowledgment, pauses game optionally. Example: "Colony starving", "Attack incoming"
- **Important (amber):** Toast that persists for 8-10 seconds, click to navigate. Example: "Building complete", "Research finished"
- **Routine (neutral):** Brief toast, auto-dismiss in 3-4 seconds, logged to history. Example: "Citizen born", "Trade route active"
- **Ambient (subtle):** No toast, logged only. Example: "Seasonal change", "Minor mood shift"

### 4.3 Information Density at Screen Bottom

**Bottom of screen should contain:**
- Event feed / notification area (left or center-bottom)
- Build toolbar or action bar (center-bottom) --- if not using radial/side placement
- Minimap (bottom-right or bottom-left corner)
- Speed controls (if not in top-right)

**Bottom strip total height:** 60-100px at 1080p (5.5-9.3% of vertical space)

**Key principle:** The bottom strip is the "doing" zone. Top bar is "knowing." Bottom is "acting." This spatial metaphor maps to natural reading/interaction patterns --- status information above (you glance up), tools and actions below (you reach down).

### 4.4 Save/Load/Settings Placement

**Universal convention:** Accessible via Escape key opening a pause/system menu.

**Secondary access patterns:**
- Small gear icon in top-right corner (beside speed controls)
- Never in the bottom action bar --- it's not a gameplay action
- Never competing visually with game controls

**Gold standard: RimWorld**
- Escape opens a clean menu overlay
- Options, Save, Load, Quit clearly listed
- Auto-save indicator is a small icon, not a button

### 4.5 Surfacing Urgent vs Routine Information

**The "Siren Principle" (from Norman's feedback theory):**
- Urgency must be communicated through MULTIPLE channels simultaneously:
  - Visual: color shift (neutral > amber > red), size increase, animation (pulse/flash)
  - Audio: escalating sound design (chime > alert > alarm)
  - Spatial: UI element moves toward center of screen as urgency increases
  - Temporal: urgent items persist; routine items auto-dismiss

**Frostpunk's Environmental Urgency Model:**
- As temperature drops, the UI itself frosts over at the edges
- The screen narrows visually, creating subconscious pressure
- No notification needed --- the player FEELS the crisis through the interface chrome
- This is diegetic UI at its finest and represents the gold standard for environmental urgency

### 4.6 Anti-Patterns to Avoid

1. **Notification spam:** More than 3 toasts visible simultaneously
2. **Uniform urgency:** All events styled the same regardless of importance (everything is red = nothing is red)
3. **Log dependency:** Requiring players to scroll through a text log to find actionable items
4. **Blocking modals for routine events:** Pausing the game to tell you a building finished
5. **Missing event history:** No way to review past notifications (especially missed ones)
6. **Sound-only alerts:** Important events with audio but no persistent visual indicator

### 4.7 Scoring Criteria (10/10 Implementation)

| Criterion | Weight | 10/10 Standard |
|-----------|--------|----------------|
| Urgency differentiation | 25% | >=3 visual urgency tiers with distinct color/size/behavior |
| Non-intrusiveness | 20% | Routine events never block gameplay or obscure viewport |
| Actionability | 20% | Clicking a notification navigates to its source in-game |
| History access | 15% | Full event log accessible within 1 click from HUD |
| Multi-channel feedback | 10% | Critical events use visual + audio + spatial cues |
| Auto-management | 10% | Old notifications auto-dismiss or stack; no manual cleanup |

---

## 5. Navigation & Menu Systems

### 5.1 Gold Standard: Frostpunk (Radial Menus) + RimWorld (Tab-Based Architect)

These two games represent opposite-but-excellent approaches:

**Frostpunk's Radial Menus:**
- The city is built in concentric circles around the Generator; the radial UI mirrors this spatial metaphor
- Top-level ring shows categories (housing, resources, industry, etc.)
- Inner rings reveal specific buildings within each category
- Pauses time when opened, giving players space to think
- Works exceptionally well for 6-12 categories with 4-8 items each
- Keyboard shortcut: single key opens the menu, then directional selection

**RimWorld's Architect Tabs:**
- Bottom-of-screen horizontal tab strip with ~12 categories
- Clicking a tab reveals a horizontal sub-bar of items within that category
- Each tab is labeled with text + icon for dual-encoding
- Keyboard shortcuts mapped to each tab (B for buildings, etc.)
- Strengths: Scales to many items, clear hierarchy, scannable
- Weakness: Breaks down with 20+ modded categories (needs subcategory support)

**Design Principles Applied:**
- **Norman (Mapping):** Frostpunk's radial menu spatially maps to the circular city. RimWorld's horizontal tabs map to "types of things you can build."
- **Koster (Chunking):** Both limit top-level categories to 8-12 items, each containing 4-10 sub-items.
- **Wroblewski (Progressive Disclosure):** Category > Item > Configuration. Three levels maximum.

### 5.2 Tab Bars vs Icon Strips vs Radial Menus vs Hotkey-Driven

| System | Capacity | Speed (Expert) | Discoverability | Best For |
|--------|----------|----------------|-----------------|----------|
| **Tab bar** | High (12+ categories) | Medium | High (labels visible) | Complex games with many build options |
| **Icon strip** | Medium (8-12 items) | Medium-High | Medium (icons only) | Games with recognizable visual language |
| **Radial menu** | Medium (8 per ring) | High | Medium (spatial memory) | Games with strong spatial metaphor |
| **Hotkey-driven** | Low (depends on memorization) | Highest | Low | Expert/competitive play |
| **Hybrid** | Highest | Highest | Highest | The actual gold standard |

**The Gold Standard is Hybrid:**
- Visual menu (tabs or radial) for discoverability and new players
- Keyboard shortcuts for every action, displayed on tooltips
- Search function for finding specific items by name (Anno 1800 does this)
- Favorites/pinning for player-customized quick access

### 5.3 Organizing 8+ Overlay/Panel Categories

**Strategy: The "Overseer" Pattern (from Pharaoh / Cities: Skylines)**

1. **Group by domain:** Physical (roads, buildings, zones) | Social (health, education, happiness) | Economic (trade, production, budget) | Information (overlays, statistics, reports)
2. **Maximum 4-6 top-level groups**
3. **Each group contains 3-7 items**
4. **Only one panel open at a time** (unless explicitly pinned by the player)
5. **Last-used panel remembered** per category for quick re-access

**Cities: Skylines Overlay Organization:**
- Info view button reveals a categorized grid of overlay options
- Each overlay replaces the normal world rendering with a data visualization
- Toggle on/off pattern --- not a modal, player can still interact with the world
- Categories: Electricity, Water, Pollution, Land Value, Crime, Health, etc.
- Color-coded overlays with consistent visual language (green = good, red = bad)

**Anno 1800's Layered Approach:**
- Contextual panels appear when selecting buildings/production chains
- Global statistics accessible via a dedicated statistics panel
- Trade routes have their own management screen
- Population needs are tier-specific, revealed when clicking population groups
- This per-tier progressive disclosure handles the game's extreme complexity

### 5.4 Keyboard Shortcut Discoverability

**The Three-Stage Discoverability Model:**

1. **Stage 1 - Tooltips (Passive):** Every button displays its keyboard shortcut in the tooltip. Format: "Build Road (R)" --- shortcut in parentheses, same visual weight as the label.

2. **Stage 2 - Corner Badges (Active):** Keyboard shortcut letter displayed as a small badge in the corner of each button/icon. Visible without hovering. Civilization VI and RimWorld do this well.

3. **Stage 3 - Shortcut Reference (On Demand):** A full keyboard shortcut reference accessible via a "?" or help button. Organized by category, matching the menu organization. Ideally overlaid on the actual game HUD so players can see spatial correspondence.

**Anti-pattern:** Hiding shortcuts only in the settings/keybinding menu where players never look during gameplay.

### 5.5 Context-Sensitive Navigation

**What should change based on game state:**

| Game State | Navigation Adaptation |
|------------|----------------------|
| Normal play | Full build toolbar visible, all categories accessible |
| Crisis/event | Relevant response options highlighted, irrelevant options dimmed |
| Selected building | Building-specific actions replace generic toolbar (upgrade, demolish, inspect) |
| Overlay active | Overlay-specific legend appears, other UI dims |
| Tutorial/early game | Locked categories greyed out, active ones highlighted |
| Pause menu open | Game UI fully dimmed, system menu centered |

**Gold Standard: Foundation (Polymorph Games)**
- Build menu transforms based on whether you're zoning, placing buildings, or painting terrain
- Different tool modes surface different options
- The UI is never showing you options that aren't relevant to your current action

### 5.6 Anti-Patterns to Avoid

1. **Flat menu hell:** All 30+ options in a single list with no hierarchy (pre-Steam Dwarf Fortress)
2. **Hidden categories:** Important functions buried 3+ levels deep with no breadcrumb trail
3. **Inconsistent shortcut schemes:** Some shortcuts are letters, some are F-keys, some are Ctrl+combos, with no pattern
4. **Menu amnesia:** Menus that reset to their top-level state every time they close (losing the player's place)
5. **Non-rebindable shortcuts:** Especially for non-QWERTY keyboard layouts
6. **Tooltip-only shortcuts:** Shortcuts that only appear on hover, never shown on the button face

### 5.7 Scoring Criteria (10/10 Implementation)

| Criterion | Weight | 10/10 Standard |
|-----------|--------|----------------|
| Information hierarchy | 25% | 3 clear levels: category > item > detail. <=7 items per level |
| Dual input support | 20% | Every action achievable via both mouse and keyboard |
| Shortcut discoverability | 15% | Shortcuts visible on button faces + tooltips + reference sheet |
| Context sensitivity | 15% | UI adapts to current game state, hiding/showing relevant options |
| Scalability | 15% | System remains organized with 50+ total items (future-proof) |
| Memory/consistency | 10% | Menus remember last state; layouts never shift unexpectedly |

---

## 6. Consolidated Scoring Rubric

### How to Use This Rubric

Score each area independently on a 1-10 scale using the weighted criteria above. The final score is an average of the four area scores, weighted as follows:

| Area | Weight in Final Score | Rationale |
|------|----------------------|-----------|
| Screen Layout & Viewport | 30% | Foundation of the entire experience |
| Top Bar Design | 25% | Player's primary information source |
| Bottom Strip / Event Feed | 20% | Player's action and awareness zone |
| Navigation & Menu Systems | 25% | Player's primary interaction mechanism |

### Quick Reference: Score Anchors

**10/10 - Exemplary:**
- Frostpunk-level viewport dominance with environmental UI storytelling
- Civ VI-level resource bar with 3-layer progressive disclosure
- Hybrid navigation with radial + tabs + shortcuts + search
- Priority-tiered notification system with multi-channel urgency
- All expert principles (Tufte, Norman, Chen, Ueda, Koster, Wroblewski) demonstrably applied

**7-8/10 - Strong:**
- Good viewport allocation (>75%) with clear persistent/contextual separation
- Resource bar with grouping and trend indicators
- Tab-based or icon-based navigation with keyboard shortcuts
- Differentiated notification urgency (at least 2 tiers)
- Most expert principles applied, minor violations

**5-6/10 - Adequate:**
- Acceptable viewport (>65%) but some panel competition
- Resource bar shows values but lacks trends or grouping
- Functional navigation but no shortcut discoverability
- Notifications present but poorly prioritized
- Some expert principles applied, some clear violations

**3-4/10 - Below Standard:**
- Viewport squeezed below 60% by permanent panels
- Resource bar is overwhelming or insufficient
- Navigation requires excessive clicks or memorization
- Notifications are uniform urgency or frequently missed
- Few expert principles applied

**1-2/10 - Failing:**
- UI obscures the game world
- Critical information is hidden or unreadable
- Navigation is a barrier to gameplay
- Events are missed or cause constant interruption
- Expert principles violated throughout

### Cross-Cutting Quality Gates

These must ALL pass for a score above 7 in any area:

| Gate | Requirement |
|------|-------------|
| **Tufte Test** | Data-ink ratio > 0.7 (estimated). No purely decorative HUD elements that don't reinforce theme. |
| **Norman Test** | Every interactive element has visible affordance + immediate feedback. |
| **Chen Test** | The 10 most common actions each require <=2 clicks. No forced interruptions for routine events. |
| **Ueda Test** | Default HUD state (nothing selected, no panels open) uses <=10% of screen. |
| **Koster Test** | No screen region presents >12 simultaneous distinct information items without grouping. |
| **Wroblewski Test** | At least 3 information depth levels exist (glance > inspect > deep-dive). |

---

## Appendix A: Game-Specific Reference Patterns

### Frostpunk 1 & 2 (11 bit studios)
- **Key Pattern:** Radial UI mirroring circular city layout
- **Strength:** Environmental storytelling through UI (frost, darkening)
- **Strength:** Time-pause on menu open preserves flow
- **Weakness (FP2):** Initial release had readability issues, patched post-launch
- **Use for:** Radial menu benchmarking, environmental urgency, viewport allocation

### Anno 1800 (Ubisoft Blue Byte)
- **Key Pattern:** Layered information density with dark HUD / bright alerts
- **Strength:** Complex production chains managed through contextual panels
- **Strength:** Per-tier population needs disclosure
- **Weakness:** Trade route UI complexity remains challenging
- **Use for:** Information density benchmarking, color strategy, production chain UIs

### Civilization VI (Firaxis)
- **Key Pattern:** Horizontal top bar with grouped, expandable resources
- **Strength:** Right-side notification queue with camera-jump
- **Strength:** Turn structure naturally organizes information flow
- **Weakness:** Late-game information density can overwhelm with many civs
- **Use for:** Top bar benchmarking, notification queue pattern, resource display

### Dwarf Fortress (Bay 12 / Kitfox)
- **Key Pattern:** Maximum information density, redesigned for accessibility in Steam version
- **Strength (Steam):** Mouse-driven navigation, text filters, tabs
- **Weakness:** Tension between power-user density and newcomer readability
- **Weakness:** New UI reduced some power-user tools
- **Use for:** Information density vs readability tension, accessibility transformation case study

### RimWorld (Ludeon)
- **Key Pattern:** Bottom architect tab bar with horizontal sub-items
- **Strength:** Speed controls (1-4 + space) are best-in-class for simplicity
- **Strength:** Event feed (left side) with prioritized alerts
- **Weakness:** Architect menu does not scale to modded content without community patches
- **Use for:** Bottom toolbar benchmarking, speed control standard, extensibility anti-pattern

### Pharaoh: A New Era (Dotemu / Triskell Interactive)
- **Key Pattern:** Overseer system for categorized city management
- **Strength:** Dramatically improved UX over 1999 original while preserving structure
- **Strength:** Workers Tooltip and Fixed Worker Ratio surface critical data inline
- **Weakness:** Original fixed sidebar approach consumed significant screen space
- **Use for:** Overseer/category panel pattern, legacy UI modernization reference

### Cities: Skylines (I & II) (Colossal Order / Paradox)
- **Key Pattern:** Info view overlay system for data visualization on game world
- **Strength:** Color-coded overlays with consistent visual language
- **Strength:** Demand bars (green/blue/orange) are a masterclass in aggregate indicators
- **Weakness (CS2):** Launch UI had usability regressions from CS1
- **Use for:** Overlay system benchmarking, aggregate indicator design, toolbar organization

### Banished (Shining Rock)
- **Key Pattern:** Minimalist, fully toggleable, repositionable HUD
- **Strength:** Proves extreme minimalism works for city builders
- **Strength:** Everything is opt-in; player builds their own HUD
- **Weakness:** Similar icons cause confusion (herbalist/gatherer/forester)
- **Use for:** Minimalism benchmark, toggleable UI reference, icon distinctiveness anti-pattern

### Foundation (Polymorph Games)
- **Key Pattern:** Context-sensitive build tools that transform based on current action
- **Strength:** Zone-painting and building placement have distinct UI modes
- **Strength:** Organic layout tools reflect the game's gridless design
- **Weakness:** Early access UI underwent significant iteration
- **Use for:** Context-sensitive navigation, mode-based UI adaptation

### Old World (Mohawk Games)
- **Key Pattern:** Nested tooltip system for deep information layering
- **Strength:** Undo button for misclicks is exceptional accessibility
- **Strength:** Card-based tech tree adds visual variety to information display
- **Weakness:** Tooltip-in-tooltip-in-tooltip creates cognitive overload
- **Weakness:** Small icons and text density hostile to new players
- **Use for:** Tooltip depth anti-pattern, undo-as-feature reference, onboarding challenges

---

## Appendix B: Key Takeaways for Implementation

1. **Start subtractive.** Design with everything, then remove until only essentials remain. Add back complexity through progressive disclosure.

2. **The top bar is a dashboard, not a database.** Show 5-8 critical metrics with trends. Everything else lives one hover/click deeper.

3. **The bottom strip is for doing, not knowing.** Tools, actions, and notifications belong here. Status and state belong up top.

4. **Notifications must earn their interruption level.** Four tiers minimum: critical (banner), important (persistent toast), routine (auto-dismiss toast), ambient (log only).

5. **Every button needs a keyboard shortcut, and the shortcut needs to be visible on the button.**

6. **Radial menus are powerful but limited.** Best for 6-12 categories of 4-8 items. Beyond that, use tabs with search.

7. **Test with the 2-second scan.** If a player cannot assess the critical state of their game within 2 seconds of glancing at the HUD, the information architecture has failed.

8. **Color is meaning, not decoration.** Dark HUD = recede. Bright element = attention. Red = danger. Green = healthy. Never use color without semantic purpose.

9. **The game world is the primary display.** Every HUD pixel must justify its existence against an alternative: showing more of the game world.

10. **Build for three players simultaneously:** The newcomer (needs discoverability), the intermediate (needs efficiency), and the expert (needs shortcuts and density).

---

*Sources used in compilation of this report are listed below.*

## Sources

- [Frostpunk Radial-Driven Design (Xbox Wire)](https://news.xbox.com/en-us/2019/09/20/frostpunk-console-edition-radial-driven-design-of-gameplay-and-controls/)
- [Frostpunk Game UI Database](https://www.gameuidatabase.com/gameData.php?id=38)
- [Frostpunk 2 Game UI Database](https://www.gameuidatabase.com/gameData.php?id=1965)
- [Frostpunk 2 UI Improvements (PCGamesN)](https://www.pcgamesn.com/frostpunk-2/ui-improvements)
- [Anno 1800 DevBlog: User Interface (Anno Union)](https://www.anno-union.com/devblog-user-interface-2/)
- [Anno 1800 Game UI Database](https://www.gameuidatabase.com/gameData.php?id=1118)
- [Civilization VI Game UI Database](https://www.gameuidatabase.com/gameData.php?id=639)
- [Civilization VI Interface In Game](https://interfaceingame.com/games/sid-meiers-civilization-vi/)
- [RimWorld User Interface Wiki](https://rimworldwiki.com/wiki/User_interface)
- [Cities: Skylines Game UI Database](https://www.gameuidatabase.com/gameData.php?id=526)
- [Banished Indie Review (Medium)](https://medium.com/@KlaraMelinaca/banished-indie-review-1abcd5f7bf91)
- [Foundation UI Overview (Polymorph Wiki)](https://wiki.polymorph.games/foundation/UI_Overview)
- [Foundation 1.7 UI Update](https://www.polymorph.games/foundation/news/2021/06/22/1-7-new-ui-update-live/)
- [Dwarf Fortress New UI (PC Gamer)](https://www.pcgamer.com/the-new-dwarf-fortress-ui-looks-so-much-better/)
- [Pharaoh: A New Era (Steam)](https://store.steampowered.com/app/1351080/Pharaoh_A_New_Era/)
- [Old World (Mohawk Games)](https://mohawkgames.com/oldworld/)
- [Tufte's Principles of Data-Ink (Columbia)](https://jtr13.github.io/cc19/tuftes-principles-of-data-ink.html)
- [Mastering Tufte's Data Visualization Principles (GeeksforGeeks)](https://www.geeksforgeeks.org/data-visualization/mastering-tuftes-data-visualization-principles/)
- [Don Norman's Principles of Interaction Design (Sachin Rekhi)](https://medium.com/@sachinrekhi/don-normans-principles-of-interaction-design-51025a2c0f33)
- [Don Norman's Principles (UX Magazine)](https://uxmag.com/articles/understanding-don-normans-principles-of-interaction)
- [Jenova Chen - Flow in Games (MFA Thesis)](https://www.jenovachen.com/flowingames/Flow_in_games_final.pdf)
- [Jenova Chen - Design Flow](http://jenovachen.info/design-flow)
- [Fumito Ueda - Design by Subtraction (SUPERJUMP)](https://medium.com/super-jump/the-definition-of-design-by-subtraction-a051e127f171)
- [Raph Koster - A Theory of Fun](https://www.theoryoffun.com/)
- [Luke Wroblewski - Mobile First](https://www.lukew.com/resources/mobile_first.asp)
- [HUD Size Analysis (Medium)](https://medium.com/@joshmoskowitz/size-does-matter-an-analysis-of-videogame-huds-24321750b665)
- [HUD Design Best Practices (Polydin)](https://polydin.com/game-hud-design/)
- [AAA HUD Design Best Practices (iABDI)](https://www.iabdi.com/designblog/2022/6/1/nier-automata-chips)
- [Game UI Database - Notifications](https://www.gameuidatabase.com/index.php?scrn=145)
- [UI Strategy Game Design Dos and Don'ts (Game Developer)](https://www.gamedeveloper.com/design/ui-strategy-game-design-dos-and-don-ts)
