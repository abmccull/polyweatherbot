# Build Menu, Construction System & Building Detail Panel: Design Standards Report

## Comprehensive UI/UX Research for City-Builder & Management Games

---

## Table of Contents

1. [Build Menu Organization](#1-build-menu-organization)
2. [Building Selection & Placement](#2-building-selection--placement)
3. [Building Detail / Info Panel](#3-building-detail--info-panel)
4. [Build Menu Positioning](#4-build-menu-positioning)
5. [Cross-Cutting Design Principles](#5-cross-cutting-design-principles)
6. [10/10 Criteria Checklist](#6-1010-criteria-checklist)

---

## 1. Build Menu Organization

### 1.1 Menu Topology: Three Archetypes

#### Horizontal Bottom Toolbar (Anno 1800, Cities: Skylines II, Timberborn)

**How it works:** A persistent toolbar anchored to the bottom-center of the screen. Top-level category icons expand upward into sub-category grids or lists.

**Pros:**
- Follows the "ground-up" spatial metaphor -- you build from the bottom of the screen, where the terrain is
- Maximum horizontal real estate for categories (8-12 icons visible simultaneously)
- Does not occlude the map edges where most construction happens
- Anno 1800's devs chose this explicitly so players could "build complex production lines with the smallest amount of clicks possible"

**Cons:**
- Vertical sub-menus can grow tall with 20+ items, pushing content off-screen
- Competes with other bottom-anchored HUD elements (resource bars, timeline controls)

**Gold Standard:** Anno 1800 -- dark HUD for the persistent toolbar (reducing eye fatigue), bright popups for transient info panels. Categories organized by population tier AND function, with customizable construction menu ordering.

---

#### Vertical Left Sidebar (RimWorld, Timberborn alternate, Banished)

**How it works:** A column of category tabs along the left edge. Selecting a tab reveals a sub-panel of building options.

**Pros:**
- Natural left-to-right reading flow: categories -> items -> placement on map
- Leaves the bottom of the screen completely free for timeline/speed controls
- Vertically scrollable lists handle large building counts gracefully

**Cons:**
- Occludes the left side of the map during placement
- Can feel narrow; icon+label combinations get cramped

**Gold Standard:** RimWorld's Architect tab -- 12 clearly-named categories (Orders, Zone, Structure, Production, Furniture, Power, Security, Misc, Floors, Recreation, Ship, Temperature). Each opens a horizontal row of building icons at the bottom. The combination of vertical tabs + horizontal item rows creates an L-shaped menu that maximizes both category count and item visibility.

---

#### Radial / Pie Menu (Frostpunk 1 & 2, Foundation)

**How it works:** Triggered contextually (right-click or dedicated button). Options fan out in a circle around the cursor.

**Pros:**
- Follows Fitts' Law: all options are equidistant from the cursor starting point
- Fastest selection time for up to 8 items per ring
- Natural for gamepad input (analog stick direction maps to selection)
- Frostpunk uses this as thematic reinforcement -- circular menus echo the circular generator and ring-based city layout

**Cons:**
- Hard cap of ~8 items per ring before sectors become too small to target
- Nested radials (radial within radial) quickly become disorienting
- Poor for games with 20+ building categories

**Gold Standard:** Frostpunk 2's Quick Radial -- context-sensitive (shows different options based on what you're hovering over: Frostbreaking on frozen terrain, District menu on buildable areas, Special Buildings on existing districts). Color-codes districts for orientation. Auto-pauses time when open, eliminating time pressure during menu navigation.

---

### 1.2 Category Organization Patterns

| Pattern | Example Games | When to Use |
|---------|--------------|-------------|
| **By Function** (housing, production, infrastructure) | Anno 1800, Cities: Skylines II, Banished | Games where understanding what buildings DO matters most |
| **By Resource/Chain** (food chain, industry chain) | Against the Storm, Factorio | Games with complex production dependencies |
| **By Unlock Tier** (population level, research tier) | Anno 1800, Pharaoh: A New Era | Games with strong progression gating |
| **Hybrid: Function + Tier** | Anno 1800 | Best-in-class for complex games; primary sort by function, secondary filter by tier |

**Design Principle (Don Norman - Natural Mapping):** The category system should map to how players THINK about their city, not how the data is structured internally. Players think "I need food production" not "I need a tier-2 building." Function-first categories respect this mental model.

**Design Principle (Steve Krug - Don't Make Me Think):** Each category should be a "mindless, unambiguous choice." If a player hesitates about whether a Greenhouse belongs in "Food" or "Advanced Buildings," the taxonomy has failed.

---

### 1.3 Item Density: How Many Is Too Many?

**The 7+/-2 Rule (Miller's Law) applied to build menus:**

- **Per category visible at once:** 6-12 items without scrolling (sweet spot: 8)
- **Total categories in top-level menu:** 6-10 (sweet spot: 8)
- **Maximum before requiring search/filter:** 40+ total buildings

**How the best games handle 20+ buildings:**

| Game | Total Buildings | Strategy |
|------|----------------|----------|
| Anno 1800 | 200+ | Tiered categories + customizable toolbar + search |
| Factorio | 300+ | 10 customizable quickbars (each with 10 slots) + blueprint library + text search |
| RimWorld | 100+ | 12 architect tabs, each with 5-15 items in a horizontal row |
| Against the Storm | 40+ | Card-based system with smaller icons in construction panel; overflow handled by scrolling |
| Cities: Skylines II | 100+ | Multi-level category tree with expressive icons |

**Anti-pattern:** Pharaoh: A New Era's original UI -- "made for visual reasons to sell the look and feel of Egypt" rather than for function. Beautiful but functionally opaque.

**Design Principle (Raph Koster - Chunking):** The brain processes information by "chunking" it into groups. A build menu with 50 flat items is 50 chunks. The same 50 items in 6 categories of ~8 each is 6 chunks (categories) + 8 chunks (items within the selected category) = 14 total cognitive units. That is a 70% reduction in cognitive load.

---

### 1.4 Collapsible vs Always-Visible Categories

**Always-visible** (Anno 1800, Cities: Skylines II): Better for games where players frequently switch between categories during a single construction session. The top-level categories serve as persistent orientation -- "you are here" wayfinding.

**Collapsible** (RimWorld): Better for games where players enter a "construction mode" and stay within one category for extended periods. Collapsing frees screen real estate.

**Best Practice:** Always-visible top-level categories with collapsible sub-categories. This provides persistent wayfinding while managing depth.

---

## 2. Building Selection & Placement

### 2.1 Preview / Ghost Building

**Gold Standard:** Factorio's ghost placement system.

- Semi-transparent "ghost" building follows the cursor at all times during placement mode
- Green tint = valid placement; Red tint = invalid (with specific reason shown)
- Ghost persists after placement as a "blueprint" that workers will eventually construct
- Shift+click places ghost without consuming resources (planning mode)

**Key patterns across games:**

| Feature | Factorio | Cities: Skylines II | Anno 1800 | RimWorld |
|---------|----------|-------------------|-----------|----------|
| Ghost preview | Yes | Yes | Yes | Yes (blueprint) |
| Color-coded validity | Green/Red | Green/Red/Yellow | Green/Red | Blue (plan) / Green (valid) |
| Snap to grid | Yes | Road-snap | Yes | Yes |
| Rotation preview | R key, shows output direction | R key | R key | R key |
| Multi-place (drag) | Yes (belts, walls) | Yes (roads, zones) | Yes (roads) | Yes (walls, floors) |
| Range/radius overlay | Yes (turrets, inserters) | Yes (services) | Yes (influence) | No (mod adds it) |

**Design Principle (Don Norman - Feedback):** The ghost building IS the feedback. It answers three questions simultaneously: "What will I build?" (shape), "Where will it go?" (position), and "Can I build it here?" (color). No separate feedback UI is needed.

**Anti-pattern:** Placing a building with no preview and receiving a text error "Cannot build here" after the click. This violates the principle of feedforward -- showing consequences before commitment.

---

### 2.2 Cost Display During Placement

**Gold Standard:** Against the Storm's building card system.

- When you select a building to place, the card shows all costs inline
- Resources you HAVE are displayed in normal color; resources you LACK are in red
- Cost deduction preview: your resource bar shows the projected post-build amounts in parentheses

**Best practices:**
- Show cost at point of selection, not just after placement
- Use color-coding: sufficient resources (white/green), insufficient (red/dimmed)
- Show maintenance cost alongside construction cost (Anno 1800 does this with an upkeep indicator on hover)
- For multi-resource costs, stack resources left-to-right with icons (not text labels)

**Design Principle (Edward Tufte - Data-Ink Ratio):** Every pixel in the cost display should convey information. An icon of wood + "50" is higher data-ink ratio than "Requires: 50 units of Wood" -- same information, 60% less visual space.

---

### 2.3 Requirements / Unlock Visibility

**Gold Standard:** Anno 1800's tiered unlock system.

- Locked buildings appear greyed-out in the construction menu with a lock icon
- Hovering shows the specific requirement: "Requires 1,000 Artisans" or "Requires Research: Steel Beams"
- Buildings approaching unlock are subtly highlighted (glow effect)

**Anti-pattern:** Hiding locked buildings entirely. Players cannot plan ahead or understand what they are working toward if they cannot see what exists. The menu should teach the game's possibility space.

**Design Principle (Dieter Rams - "Good design makes a product understandable"):** Showing locked buildings with clear requirements turns the build menu into a progression roadmap. The player understands the system before they have full access to it.

---

### 2.4 Cancel / Undo Patterns

**Best practices from surveyed games:**

- **ESC or Right-click** to cancel current placement (universal convention -- do not deviate)
- **Ctrl+Z** for undo of placed-but-not-yet-built structures (Factorio, Cities: Skylines II)
- **Delete/bulldoze mode** as a separate tool, never mixed into the build tool
- **Confirmation for expensive operations** only: demolishing a fully upgraded building should prompt; canceling a blueprint should not

**Anti-pattern:** Requiring players to switch to a demolish tool, find the accidentally placed building, and click it to remove. Ctrl+Z is the minimum viable undo.

---

## 3. Building Detail / Info Panel

### 3.1 Information Density

**The hierarchy of information needs when clicking a building:**

| Priority | Information | Display Method |
|----------|------------|----------------|
| 1 (Critical) | Building name + status (working/idle/disabled) | Header with status icon |
| 2 (High) | Current production/output | Primary content area with progress indicators |
| 3 (High) | Worker assignment (current/max) | Compact counter near header |
| 4 (Medium) | Input/output resources with rates | Icon grid with numbers |
| 5 (Medium) | Efficiency % and factors affecting it | Progress bar or percentage |
| 6 (Low) | Upgrade options | Button/section at bottom |
| 7 (Low) | Maintenance cost | Small text or icon |
| 8 (Contextual) | Range/influence radius overlay on map | Toggle, not always shown |

**Design Principle (Edward Tufte - "Above all else show the data"):** Priority 1-3 information should be visible WITHOUT scrolling. Priority 4-5 may require a single scroll or tab switch. Priority 6-8 should be behind an explicit action (click "more" or switch tab).

**Design Principle (Dieter Rams - "As little design as possible"):** Every decorative border, ornamental flourish, or background texture in a building info panel is non-data-ink that competes with actual information. Anno 1800 chose "clean design with minimal ornamentation" for exactly this reason.

---

### 3.2 Panel Presentation Patterns

#### Tooltip (hover)
**Used by:** Timberborn, Banished (initial hover), Factorio
**Best for:** Quick reference during placement
**Capacity:** 3-5 lines of information maximum
**Rule:** Disappears when cursor moves. Never contains interactive elements.

#### Sidebar Panel (persistent, anchored to edge)
**Used by:** Cities: Skylines II, Anno 1800
**Best for:** Detailed building management during gameplay
**Capacity:** Full building info including tabs for production/storage/upgrades
**Rule:** Must not overlap the selected building on the map. Right-side placement is conventional.

#### Popup / Modal (centered, requires dismissal)
**Used by:** Pharaoh: A New Era, some RimWorld interactions
**Best for:** Rare, high-stakes decisions (upgrade confirmation, demolish warning)
**Capacity:** Any amount of information
**Rule:** Use sparingly. Every popup interrupts flow.

#### Floating Panel (near the building, not anchored to edge)
**Used by:** Against the Storm, Timberborn (info windows)
**Best for:** Quick building info without losing map context
**Capacity:** Medium (3-8 data points)
**Rule:** Must not obscure the building it describes. Auto-reposition if near screen edge.

**Gold Standard:** Anno 1800's layered approach:
1. **Hover** = tooltip with name, output, and status (2 seconds)
2. **Click** = right-side sidebar panel with full details, tabbed (Production / Workers / Statistics)
3. **Upgrade/action** = inline buttons within the sidebar, no additional popup

This three-tier system respects the player's intent: hovering = "What is this?", clicking = "I want to manage this", action button = "I want to change this."

---

### 3.3 Production Chain Visualization

**Gold Standard:** Anno 1800's production chain display.

- Visual tree showing: Raw Input -> Processing -> Output
- Each node shows the building icon, production rate, and current stock
- Color-coded: green (surplus), yellow (balanced), red (deficit)
- Clicking a node in the chain selects that building on the map

**Against the Storm's recipe system:**
- Building panel divided into tabs: Production, Storage, Effects
- Recipe list with checkboxes to enable/disable specific recipes
- Ingredients section shows goods stored for use in recipes
- Efficiency stars indicate how well-suited a building is for a recipe

**Design Principle (Don Norman - Visibility of System Status):** Production chains must make invisible relationships visible. A Bakery's info panel should show not just "makes bread" but "needs flour from the Mill, which needs wheat from the Farm." The player must see the SYSTEM, not just the node.

---

### 3.4 Workforce / Assignment Display

**Gold Standard:** Pharaoh: A New Era's workforce overseer system (functionality, not aesthetics).

- Global workforce bar: green = employed, red = vacant positions
- Per-building worker count: current / maximum
- Priority system: drag to reorder which buildings get workers first
- Sector grouping: workers categorized by industry type

**Best practices:**
- Show worker count as `[current]/[max]` with color: green (full), yellow (partial), red (empty)
- Allow priority adjustment from the building info panel itself, not just a global screen
- Show commute status: "Workers assigned" vs "Workers en route" vs "No workers available"

**Anti-pattern:** Showing worker count without showing WHY workers are missing. Is it a housing shortage? A pathing issue? Competing priority? The info panel should diagnose, not just report.

---

### 3.5 Upgrade / Repair Actions Placement

**Convention:** Bottom of the info panel, below all informational content.

**Rationale (Don Norman - Mapping):** Actions change the building's state. Information describes the building's state. Placing actions below information follows the natural flow: understand first, then act. This maps to the decision process: "What is happening?" -> "What should I do about it?"

**Best practice layout:**

```
+----------------------------------+
| [Icon] Building Name     [X Close]|
| Status: Active                    |
|-----------------------------------|
| Production: 4.5/min              |
| Workers: 8/10                    |
| Efficiency: 85%                  |
|-----------------------------------|
| Inputs:  [wood] 12  [stone] 5    |
| Outputs: [brick] 4               |
|-----------------------------------|
| [ Upgrade to Level 2 ] [ Repair ]|
| [ Disable ] [ Relocate ] [ Demo ]|
+----------------------------------+
```

**Anti-pattern:** Placing the "Demolish" button next to the "Upgrade" button with the same visual weight. Destructive actions should be visually distinct (red, smaller, or behind a confirmation).

---

## 4. Build Menu Positioning

### 4.1 Left Sidebar Convention

**Who uses it:** RimWorld, some Timberborn configurations, many mobile city builders.

**Why it works:**
- Western reading order (left-to-right): menu is the first thing the eye hits, then the map
- Natural "table of contents" position -- the sidebar IS the index of what you can build
- Does not compete with right-side info panels (selected building details)

**Why it doesn't work:**
- Occludes the left edge of the map, which is just as valuable as any other edge
- On ultrawide monitors, the sidebar is very far from center-screen action
- If the game also has a resource bar at the top-left, the corner becomes congested

---

### 4.2 Full-Width Bottom Bar

**Who uses it:** Anno 1800, Cities: Skylines II, Timberborn.

**Why it works:**
- The bottom edge is the least valuable map real estate (sky/horizon in most camera angles)
- Maximum width for category icons -- can show 8-12 categories without scrolling
- Sub-menus expand upward, which feels like "building up from the foundation"
- Does not compete with left or right sidebars for info panels

**Why it doesn't work:**
- Vertical space is at a premium on 16:9 monitors (bottom bar eats into it)
- On lower resolutions, a bottom bar + top resource bar squeezes the playable viewport

**Gold Standard:** Anno 1800 -- the bottom toolbar uses dark, subdued colors for the persistent HUD. Only the actively selected category "lights up." Sub-menus are brighter to draw attention to the current action context.

---

### 4.3 Floating Panel

**Who uses it:** Foundation (contextual), some modes in Against the Storm.

**Why it works:**
- Appears exactly where the player's attention already is (near the cursor)
- Minimal screen footprint when not in use (zero persistent UI)
- Ideal for contextual/radial menus

**Why it doesn't work:**
- No persistent category visibility means players must remember what categories exist
- Inconsistent position makes muscle memory impossible
- Can obscure the exact area the player wants to build in

---

### 4.4 Handling Build Menu + Info Panel Simultaneously

**The core conflict:** When a player is building, they need the build menu open. If they click an existing building to check its status, the info panel opens. Both need screen space.

**Solutions ranked by quality:**

1. **Best: Spatial separation** (Anno 1800) -- Build menu at bottom, info panel on right side. Both can be open simultaneously without overlap. The player can check a factory's output while selecting a new building to place next to it.

2. **Good: Contextual swap** (Frostpunk 2) -- Build menu replaces info panel contextually. The radial menu auto-pauses time, so there is no penalty for the swap. But you cannot reference building info while in the build menu.

3. **Acceptable: Tabbed coexistence** (RimWorld) -- Architect tab at bottom-left, info panel in bottom-left (same area, tabbed). Works because RimWorld's building management is turn-based-ish (you can pause freely).

4. **Poor: Modal exclusion** -- Opening the build menu closes any open info panel, and vice versa. Forces the player to memorize information before acting on it.

**Design Principle (Steve Krug - "Don't Make Me Think"):** If a player needs to remember what a building produces in order to decide what to build next to it, the UI has failed. Both pieces of information must be visible simultaneously, or the info must be embedded in the build menu itself (e.g., showing production chain context when hovering over a building in the construction toolbar).

---

## 5. Cross-Cutting Design Principles

### 5.1 Don Norman: Affordances & Natural Mapping

| Principle | Application to Build Menus |
|-----------|--------------------------|
| **Affordance** | Building icons should look "selectable" -- slight 3D raise, hover highlight, cursor change to hand |
| **Natural Mapping** | Category icons should visually represent their contents: a house for housing, a gear for industry, a leaf for food |
| **Feedback** | Every click should produce immediate visual + audio response. Ghost building appears instantly upon selection |
| **Constraints** | Greyed-out / locked items constrain the player's choices to valid ones, preventing errors before they happen |

### 5.2 Steve Krug: Self-Evident Navigation

| Principle | Application |
|-----------|------------|
| **Don't make me think** | A new player should be able to find and place a house within 10 seconds of opening the build menu |
| **Mindless clicks** | Each click in the build menu chain should be obvious: Category -> Sub-category -> Building. No decision should require reading documentation |
| **Half the words** | Use icons over text labels wherever possible. "Sawmill" with a saw icon, not "Lumber Processing Facility" |

### 5.3 Dieter Rams: Less But Better

| Principle | Application |
|-----------|------------|
| **Innovative** | Against the Storm's card-based system -- uses a familiar metaphor (cards) in a novel context (construction) |
| **Useful** | Every element in the build menu must serve either navigation or information. Decorative borders are waste |
| **Unobtrusive** | The build menu should never feel like it's fighting for attention against the game world |
| **Honest** | If a building costs 500 gold, show 500 gold. Don't hide costs behind clicks |
| **As little design as possible** | Anno 1800's "clean design with minimal ornamentation" -- the gold standard |

### 5.4 Edward Tufte: Data-Ink Maximization

| Principle | Application |
|-----------|------------|
| **Above all else show the data** | Production rates, worker counts, and resource costs are the data. Everything else is chrome |
| **Maximize data-ink ratio** | An icon + number (`[wood]12`) beats a labeled field (`Wood: 12 units`) |
| **Erase non-data-ink** | Remove separating lines between items if spacing alone creates grouping. Remove background gradients if flat color suffices |
| **Erase redundant data-ink** | Don't show "Workers: 8/10 (80%)" -- the fraction alone implies the percentage |
| **Small multiples** | Production chain visualization should use consistent mini-icons at consistent scale, not varying sizes |

### 5.5 Raph Koster: Cognitive Load Budgets

| Principle | Application |
|-----------|------------|
| **Chunking** | Group buildings into 6-10 categories of 6-12 items each. The brain can hold ~7 chunks in working memory |
| **Mastery through patterns** | Consistent icon language (color = resource type, shape = building function) lets players develop pattern recognition |
| **Fun = learning** | A well-organized build menu TEACHES the game's systems through its structure. If Food is a category containing Farm -> Mill -> Bakery, the menu itself teaches the production chain |

---

## 6. 10/10 Criteria Checklist

Use this checklist to evaluate any build menu / construction system / building info panel design. A 10/10 system meets ALL of these criteria.

### Build Menu Organization (25 points)

- [ ] **5pts** - Categories organized by function with clear, unambiguous names
- [ ] **3pts** - No category contains more than 12 items without sub-grouping
- [ ] **3pts** - Top-level shows 6-10 categories simultaneously (no scrolling required)
- [ ] **3pts** - Locked/unavailable buildings are visible but clearly distinguished
- [ ] **3pts** - Category icons are instantly recognizable without reading labels
- [ ] **3pts** - A new player can find "build a house" within 10 seconds
- [ ] **3pts** - Search/filter available when total building count exceeds 40
- [ ] **2pts** - Menu remembers last-used category between sessions

### Building Selection & Placement (25 points)

- [ ] **5pts** - Ghost preview visible at all times during placement mode
- [ ] **4pts** - Color-coded validity (green = valid, red = invalid + reason)
- [ ] **4pts** - Cost displayed at point of selection with have/need color coding
- [ ] **3pts** - Grid/snap system with clear visual guides
- [ ] **3pts** - Rotation preview with output/direction indicators
- [ ] **2pts** - Multi-place support for linear buildings (walls, roads, pipes)
- [ ] **2pts** - Range/influence radius overlay during placement
- [ ] **2pts** - Undo (Ctrl+Z) for last placed building

### Building Info Panel (25 points)

- [ ] **5pts** - Three-tier information: hover tooltip -> click for panel -> action buttons within panel
- [ ] **4pts** - Status (working/idle/disabled) visible at a glance without reading
- [ ] **4pts** - Production input/output shown with icons and rates, not just text
- [ ] **3pts** - Worker count displayed as current/max with color-coded status
- [ ] **3pts** - Panel does not obscure the selected building on the map
- [ ] **3pts** - Upgrade/action buttons at bottom, destructive actions visually distinct
- [ ] **3pts** - Production chain context visible (what feeds this building, what it feeds)

### Menu Positioning & Coexistence (25 points)

- [ ] **5pts** - Build menu and info panel can be open simultaneously without overlap
- [ ] **5pts** - Build menu does not occlude primary gameplay area during placement
- [ ] **4pts** - Consistent position across all game states (no jumping/relocating)
- [ ] **4pts** - Menu opens/closes in under 200ms with clear animation
- [ ] **3pts** - ESC/right-click universally closes or cancels current action
- [ ] **2pts** - Works at all supported resolutions without content overflow
- [ ] **2pts** - Keyboard shortcuts for every category (1-0 keys or similar)

### Total: /100

**Rating Scale:**
- 90-100: Best-in-class (Anno 1800, Factorio tier)
- 75-89: Professional quality (Cities: Skylines II, RimWorld tier)
- 60-74: Functional but has notable friction points
- Below 60: Significant UX debt that will generate player complaints

---

## Summary: The Five Commandments

1. **Organize by mental model, not data model.** Players think in functions (housing, food, defense), not in database tables.

2. **Show everything, constrain gracefully.** Locked buildings should be visible but greyed-out. The build menu is both a tool and a teacher.

3. **Three tiers of information: hover, click, act.** Never force a player to click to get information they could get from hovering. Never put action buttons where information should be.

4. **The ghost building IS the interface.** A well-designed placement preview eliminates the need for separate validity checks, cost confirmations, and orientation indicators.

5. **Build menu and info panel are complementary, never competing.** If opening one closes the other, the player is forced to memorize instead of reference. Spatial separation (bottom + side) is the proven solution.

---

## Sources

- [Frostpunk: Console Edition -- Radial-Driven Design of Gameplay and Controls](https://www.screenhacker.com/frostpunk-console-edition-radial-driven-design-of-gameplay-and-controls/)
- [Frostpunk 2 Gamepad Adaptation -- Xbox Wire](https://news.xbox.com/en-us/2025/09/18/adapting-frostpunk-2s-depth-to-a-gamepad/)
- [Anno 1800 DevBlog: User Interface -- Anno Union](https://www.anno-union.com/devblog-user-interface-2/)
- [Organising the Construction Menu in Anno 1800 -- Ubisoft](https://www.ubisoft.com/en-us/help/anno-1800/gameplay/article/organising-the-construction-menu-in-anno-1800/000063621)
- [Cities: Skylines 2 UI Extreme Makeover -- eTeknix](https://www.eteknix.com/iceflake-giving-cities-skylines-2-ui-an-extreme-makeover-with-their-first-big-update/)
- [Factorio Friday Facts #278: The New Quickbar](https://www.factorio.com/blog/post/fff-278)
- [Factorio Friday Facts #170: Blueprint Library GUI Design](https://factorio.com/blog/post/fff-170)
- [Against the Storm Interface Update -- Eremite Games](https://eremitegames.com/interface-update/)
- [Against the Storm Building Guide -- TechRaptor](https://techraptor.net/gaming/guides/against-storm-building-guide)
- [RimWorld Architect Tab -- RimWorld Wiki](https://rimworldwiki.com/wiki/Architect)
- [How Timberborn's Complex Runtime UI Was Built -- Unity Case Study](https://unity.com/case-study/timberborn)
- [Timberborn User Interface -- Wiki](https://timberborn.wiki.gg/wiki/User_Interface)
- [Banished's 'Build What You Want' Game Design -- Game Developer](https://www.gamedeveloper.com/business/-i-banished-i-s-build-what-you-want-when-you-want-it-game-design)
- [Game UI/UX Design Best Practices -- Wayline](https://www.wayline.io/blog/game-ui-ux-design-best-practices-and-examples)
- [Don Norman's Principles of Interaction Design -- Sachin Rekhi](https://www.sachinrekhi.com/don-norman-principles-of-interaction-design)
- [Don't Make Me Think -- Steve Krug](https://sensible.com/dont-make-me-think/)
- [Dieter Rams: 10 Principles in a Digital World -- empathy.co](https://empathy.co/blog/dieter-rams-10-principles-of-good-design-in-a-digital-world/)
- [Tufte's Principles of Data-Ink](https://jtr13.github.io/cc19/tuftes-principles-of-data-ink.html)
- [Theory of Fun for Game Design -- Raph Koster](https://www.theoryoffun.com/)
- [Radial Menus in Video Games -- The Picky Champy](https://champicky.com/2022/01/21/radial-menus-in-video-games/)
- [Game UI Database](https://gameuidatabase.com/)
- [Frostpunk UI -- Game UI Database](https://www.gameuidatabase.com/gameData.php?id=38)
- [Frostpunk 2 UI -- Game UI Database](https://www.gameuidatabase.com/gameData.php?id=1965)
- [Anno 1800 UI -- Game UI Database](https://www.gameuidatabase.com/gameData.php?id=1118)
- [Reducing Cognitive Load in UI Design -- IJRASET](https://www.ijraset.com/best-journal/reducing-cognitive-load-in-ui-design)
