# Spatial Information Display Standards
## Best-in-Class UI/UX for City-Builder & Management Games

Reference document synthesized from analysis of Anno 1800, Cities: Skylines II, Civilization VI, Pharaoh: A New Era, RimWorld, Banished, Factorio, Oxygen Not Included, Foundation, and Dwarf Fortress -- filtered through the design principles of Edward Tufte, Don Norman, Dieter Rams, and Christopher Alexander.

---

## Table of Contents

1. [Design Philosophy Framework](#1-design-philosophy-framework)
2. [Minimap Design](#2-minimap-design)
3. [Tile/Hex Inspection](#3-tilehex-inspection)
4. [Data Overlays](#4-data-overlays)
5. [Site Inspection Panels](#5-site-inspection-panels)
6. [Cross-Cutting Concerns](#6-cross-cutting-concerns)
7. [Application to The Oracle](#7-application-to-the-oracle)

---

## 1. Design Philosophy Framework

Before diving into specific UI components, these four lenses should be applied to every spatial display decision:

### Tufte: Data-Ink Ratio
> "Above all else, show data. Maximize the data-ink ratio. Erase non-data-ink. Erase redundant data-ink."

- Every pixel on a minimap, tooltip, or overlay must carry information
- Decorative borders, ornamental textures, and chrome are enemies of comprehension
- Sparkline principle: a data display with data-ink ratio of 1.0 -- nothing that isn't data
- "Chartjunk" in game UI = unnecessary frames, gradient fills, animated borders that don't encode data

**Applied to game UI:** Anno 1800 exemplifies this -- dark, neutral HUD colors that don't compete with gameplay, with brighter colors reserved for notifications that demand attention. The HUD is always-visible but low-contrast; pop-ups are high-contrast.

### Norman: Visibility of System Status & Natural Mapping
> "The more visible an element is, the more likely users will know about them and how to use them."

- The system must always show the player what state the world is in
- Controls should spatially correspond to the things they affect (natural mapping)
- Feedback must be immediate -- hover response, click response, overlay toggle
- Bridge the "gulf of execution" -- the gap between what the player wants to do and how they figure out how to do it

**Applied to game UI:** Civ VI's lens system puts the toggle button directly above the minimap -- spatially near the map it transforms. RimWorld's room stats appear as overlays directly on the rooms they describe.

### Rams: As Little Design as Possible
> "Less, but better -- because it concentrates on the essential aspects."

Key principles for game UI:
- **Unobtrusive:** The UI is a tool, not a decorative object. Leave room for the game world
- **Honest:** Don't make a stat bar look bigger than it is. Don't use visual tricks that misrepresent data
- **Thorough to the last detail:** Every pixel matters at small scales (minimap, sparklines, tiny stat bars)
- **As little design as possible:** If a number communicates it, don't add an icon. If an icon communicates it, don't add a label. If color communicates it, don't add an icon

**Applied to game UI:** Banished's initial UI is a single compact toolbar -- panels appear only when summoned. Factorio's 4px module grid ensures every UI element scales proportionally without waste.

### Alexander: Pattern Language & Spatial Relationships
> "There are no things, really, but relationships."

- UI elements exist in spatial relationship to each other and to the game world
- A minimap is not an isolated widget -- it's a portal that bridges macro and micro views
- Inspection panels relate to the object being inspected through spatial proximity or visual linking
- Overlay systems create layers of meaning on the same spatial canvas

**Applied to game UI:** Foundation's zone-painting system lets you paint directly on the world, making the UI and game space the same thing. Dwarf Fortress's multi-level view maintains spatial relationships across vertical layers.

---

## 2. Minimap Design

### Gold Standard: Civilization VI + Factorio Hybrid

Civ VI places the minimap bottom-right with lens toggles directly above it -- creating a self-contained "spatial awareness station." Factorio adds seamless zoom-to-map functionality where the minimap and full map are the same system at different zoom levels.

### Position Conventions

| Position | Usage Pattern | Best For |
|----------|--------------|----------|
| **Bottom-left** | Traditional RTS (StarCraft, Age of Empires). Near command cards | Games with frequent unit commands alongside map navigation |
| **Bottom-right** | Strategy/builder (Civ VI, Anno 1800). Near info panels | Games where minimap is a reference, not an action center |
| **Top-right** | Some builders (Cities: Skylines info views). Near menus | Games where bottom screen is reserved for toolbar/action bar |
| **Collapsible/toggleable** | Banished, Factorio. Space-efficient | Games where screen real estate is at a premium |

**Decision framework:**
- Place the minimap **near other spatial controls** it relates to (zoom, camera, overlays)
- Keep it **away from frequently-clicked action bars** to prevent misclicks
- Bottom placement is generally safer: the bottom of the screen is less likely to occlude important game content (most important action happens in center/upper screen)
- Research finding: minimap should not exceed **10% of viewport area**. Smaller than 5% becomes decorative/unusable

### Information Density

**Layer 0 -- Always visible (the base map):**
- Terrain type via color (water, land, elevation)
- Owned/unowned territory boundaries
- Camera viewport rectangle (the "where am I" indicator)
- This is Tufte's "base ink" -- it should be recognizable at a glance

**Layer 1 -- Toggle-able symbolic overlay:**
- Building footprints (simplified to colored dots/blocks)
- Resource locations
- Alert indicators (flashing regions for fires, shortages, threats)
- Road/path network (simplified)

**Layer 2 -- Context-dependent (shown when relevant):**
- Selected unit/building highlighted on minimap
- Active overlay data (pollution heatmap projected onto minimap)
- Trade routes, influence ranges

**Anti-pattern: Showing everything at once.** A minimap crammed with icons, labels, and overlays simultaneously becomes a noise field. Follow ONI's approach: the minimap reflects the currently-active overlay mode.

### Interaction Patterns

| Interaction | Standard | Notes |
|-------------|----------|-------|
| **Click to navigate** | Left-click jumps camera to that location | Universal. Must feel instant |
| **Click-drag to pan** | Hold-click and drag to scroll the view | Lets players "scrub" across the map |
| **Scroll to zoom** | Mouse wheel over minimap changes map zoom | Factorio does this brilliantly -- seamless zoom from minimap to full map |
| **Right-click context** | Right-click on minimap for context actions | Optional. Used in RTS for move commands |
| **Hover preview** | Hovering minimap shows coordinates or region name | Light-touch. Don't obscure the minimap |

### When Minimaps Become Wasted Space

A minimap is wasted space when:
1. **The map is too small to need one** -- if the player can see 80%+ of the world, skip the minimap
2. **It doesn't update** -- a static image with no camera rectangle is wallpaper
3. **It's not interactive** -- if clicking does nothing, it's just decoration
4. **It duplicates the main view** -- if it shows the same info at the same fidelity, it adds nothing
5. **The game is vertically-oriented** (like Dwarf Fortress z-levels or ONI) -- a 2D top-down minimap misrepresents a 3D world. Consider a cross-section indicator or level selector instead

**Alternative to a minimap for your domain:** If your game world is tile-based with a fixed viewport, consider a **world map toggle** (full-screen overlay) rather than a persistent minimap. Factorio's approach of seamless zoom is the best of both worlds.

### 10/10 Criteria for Minimaps

- [ ] Camera viewport rectangle is always visible and updates in real-time
- [ ] Click-to-navigate with < 100ms response
- [ ] No more than 10% of screen real estate
- [ ] Reflects active overlay mode (not a separate data silo)
- [ ] Toggle-able with a single keypress
- [ ] Shows alerts/events as pulsing indicators
- [ ] Supports click-drag panning
- [ ] Color-coded by the most important spatial variable (territory, terrain, or selected overlay)
- [ ] Positioned near related spatial controls (overlay toggles, zoom)
- [ ] Remains useful at all zoom levels of the main view

---

## 3. Tile/Hex Inspection

### Gold Standard: Civilization VI Tooltip + Factorio Entity Info

Civ VI shows a concise tooltip on hover with tile yields (food, production, gold, etc.) as icon+number pairs. Factorio shows entity details in a persistent right-side panel on hover -- no click required for basic info, click for detailed configuration.

### Hover vs. Click: The Two-Tier Model

This is the most critical design decision for tile inspection. The industry consensus:

**Hover (instant, lightweight):**
- Tile name/type
- 1-3 key stats as icon+number pairs
- Owner/zone status
- Current alerts or status effects
- Disappears when cursor moves away

**Click (persistent, detailed):**
- Full stat breakdown
- Historical data / trends
- Actionable controls (build, upgrade, demolish)
- Relationship to adjacent tiles
- Stays open until dismissed or another tile clicked

**Why this split matters:** Players hover constantly while scanning. If hover tooltips are too heavy (large panels, slow to render, obscure the world), scanning becomes painful. If click panels are too light, players can't drill into detail without opening submenus.

### Tooltip Design Patterns

**Pattern A: Floating Tooltip (Civ VI, Anno 1800)**
```
+---------------------------+
| [icon] Grassland Hills    |
| Food: 2  Prod: 1  Gold: 0|
| Appeal: Breathtaking (+6) |
| Improvement: Farm (+1F)   |
+---------------------------+
```
- Appears near cursor (above-right by default)
- Never covers the tile being inspected
- Max 4-5 lines of content
- Uses icons over text where possible

**Pattern B: Fixed Info Panel (Factorio, RimWorld)**
```
RIGHT PANEL (always same position):
+---------------------------+
| Grassland Hills           |
|---------------------------|
| Food:       2  [====--]   |
| Production: 1  [==----]   |
| Gold:       0  [------]   |
| Appeal:    +6  [======]   |
|---------------------------|
| Improvement: Farm         |
| Built by: Player          |
+---------------------------+
```
- Always in the same screen position (typically right edge)
- Updates as cursor moves -- no pop-in/pop-out
- Can be taller and more detailed than floating tooltips
- Factorio uses this for ~120 different entity types

**Pattern C: Hybrid (recommended for complex tile systems)**
- Hover = Pattern A (floating, 3-4 stats)
- Click = Pattern B (fixed panel, full detail)
- This is what Cities: Skylines II effectively does

### Multi-Variable Tile Stats Display

For tiles with multiple stats (approach, sanctity, shelter, strain, or similar multi-axis systems):

**Option 1: Labeled Bars (best for 3-6 variables)**
```
Approach:  [======----]  62
Sanctity:  [========--]  84
Shelter:   [====------]  41
Strain:    [==--------]  18
```
- Horizontal bars with numeric value
- Color-code the fill: green > yellow > red based on thresholds
- Bars allow instant comparison between variables
- Follows Tufte: the bar IS the data, the number IS the data, nothing else needed

**Option 2: Small Multiples Grid (best for 4-8 variables)**
```
+--------+--------+
| APP 62 | SAN 84 |
| [====] | [====] |
+--------+--------+
| SHL 41 | STR 18 |
| [==--] | [=---] |
+--------+--------+
```
- Tufte's "small multiples" -- same structure repeated, only data changes
- Compact, scannable, comparable
- Each cell is a sparkline-style micro-visualization

**Option 3: Radar/Spider Chart (use sparingly)**
- Only appropriate for 5+ roughly-equal variables
- Harder to read precise values
- Good for "profile shape" comparison (is this tile balanced? spiky?)
- Anti-pattern: using radar charts for 2-3 variables (a bar chart is always clearer)

**Anti-patterns:**
- Showing raw numbers without visual encoding -- forces mental math
- Using different scales for different bars without indicating it
- Tooltip text walls -- more than 6 lines of text forces reading rather than scanning
- Placing the tooltip where it covers adjacent tiles the player is trying to compare

### Visual Encoding Hierarchy

Use these in order of perceptual strength (Cleveland & McGill ranking):

1. **Position on a common scale** (bar charts, aligned axes) -- most accurate
2. **Length** (bar length) -- very accurate
3. **Color saturation/hue** (heat maps) -- good for spatial, bad for precision
4. **Size/area** (icon sizing) -- imprecise but attention-grabbing
5. **Numerical labels** -- always accurate, but slow to parse visually

For tile inspection: combine **bars** (position/length) with **numbers** (precision) and **color** (urgency/quality signal).

### 10/10 Criteria for Tile Inspection

- [ ] Hover tooltip appears within 150ms, disappears within 100ms of leaving
- [ ] Tooltip never covers the tile being inspected
- [ ] Shows 3-5 key stats with both visual encoding (bars/icons) AND numeric values
- [ ] Click opens persistent detailed panel without closing tooltip flow
- [ ] Multi-variable stats use aligned bars or small multiples, not just text
- [ ] Color encodes quality/urgency (green/yellow/red thresholds)
- [ ] Stats are ordered by importance, not alphabetically
- [ ] Comparison is possible (hover two tiles in sequence, or pin one panel)
- [ ] Icons are consistent across all UI (same food icon in tooltip as in resource bar)
- [ ] Performance: tooltip rendering never causes frame drops, even on large maps

---

## 4. Data Overlays

### Gold Standard: Civilization VI Lens System + Oxygen Not Included Overlays

Civ VI provides 11+ discrete lenses (Appeal, Religion, Settler, Government, Political, Continent, District Placement, Tourism, Loyalty, Empire, Power) each accessible from a single button above the minimap. ONI pushes this further with overlays for temperature, gas pressure, decor value, light level, liquid flow, electrical networks, and more -- each completely transforming the visual representation of the world.

### The Lens/Overlay System Architecture

**Core principle:** One overlay active at a time (modal), toggled from a unified control. Not stacked.

```
OVERLAY CONTROL BAR
[Default] [Overlay A] [Overlay B] [Overlay C] [Overlay D] ...
   ^active

When Overlay B is selected:
- World tiles re-colored by Overlay B's data
- Minimap reflects Overlay B's coloring
- Tooltip shows Overlay B's relevant stat prominently
- Non-relevant visual details fade or simplify
- Buildings/entities dim unless relevant to Overlay B
```

**Why modal, not stackable:**
- Stacking overlays creates visual noise -- two heatmaps on top of each other are unreadable
- Modal overlays let each lens own the full color spectrum
- Players think in one question at a time: "where is pollution worst?" not "where is pollution worst AND land value highest?"
- Exception: one subtle background layer (like territory borders) can persist beneath any overlay

### Heat Map Patterns for Spatial Data

**Color Scale Selection:**

| Data Type | Recommended Scale | Rationale |
|-----------|------------------|-----------|
| Good/bad spectrum (desirability, appeal) | Red-Yellow-Green diverging | Intuitive polarity. Green = good |
| Intensity/magnitude (pollution, traffic) | Sequential single-hue (white to dark red, or white to dark blue) | No polarity confusion. Darker = more |
| Categorical (territory, zone type) | Distinct hues (qualitative palette) | Must be distinguishable, not ranked |
| Temperature/deviation from norm | Blue-White-Red diverging | Blue = cold/low, Red = hot/high. Universal |

**Critical rules for heat maps:**
1. **Perceptual uniformity:** Equal data differences should produce equal perceived color differences. Avoid rainbow scales -- they create false boundaries
2. **Transparency:** Overlay at 40-60% opacity so terrain/buildings remain visible beneath
3. **Legend:** Always show the color scale and its range. ONI does this with a sidebar gradient
4. **Grid alignment:** Each tile gets ONE color. Don't interpolate between tiles unless data is truly continuous
5. **Colorblind safety:** Provide alternative palettes. Avoid red-green only -- add blue, or use saturation instead of hue

### Building Influence Ranges

How to show "this building affects tiles within radius X":

**Pattern A: Radial Highlight (Pharaoh, Cities: Skylines)**
- Colored circle/hexagonal area around the building
- Green = covered, fading to transparent at edge
- Shows on selection/hover of the building
- Anno 1800: production buildings show their supply radius when selected

**Pattern B: Coverage Overlay (Cities: Skylines II Info Views)**
- Full-map overlay showing which tiles are covered by ANY building of that type
- Green = covered, Red = uncovered
- Accessed via the overlay/lens system, not per-building
- Better for "where do I need more coverage?" questions

**Pattern C: Ripple/Gradient (Foundation patrol ranges)**
- Single clear circle per coverage group, not per-unit
- Reduces visual clutter when many units share similar coverage

**Recommendation:** Use Pattern A for individual building inspection (hover/select a building, see its range) and Pattern B as a dedicated overlay lens for service coverage planning.

### Resource Flow Visualization

For showing how resources move through your settlement:

**Pattern: Animated Flow Lines (Factorio belt system)**
- Thin lines connecting source to destination
- Arrow direction shows flow
- Line thickness or color encodes throughput
- Only shown when relevant overlay is active or when a building in the chain is selected

**Anti-pattern:** Showing all resource flows simultaneously. It becomes spaghetti. Show flows per-resource or per-production-chain.

### Overlay Toggle UX

**Button bar approach (Civ VI):**
- Row of icon buttons, one per overlay
- Active overlay button is highlighted/depressed
- Clicking active overlay deactivates it (returns to default view)
- Keyboard shortcuts: number keys 1-9 for quick access

**Dropdown approach (avoid for frequent use):**
- Hides available overlays behind a click
- Adds interaction cost
- Only acceptable if there are 15+ overlays and screen space is extremely limited

**Hotkey approach (RimWorld, ONI):**
- F1-F12 or custom keybinds for each overlay
- Best for power users
- Must also have mouse-accessible buttons for discoverability

### 10/10 Criteria for Data Overlays

- [ ] One overlay active at a time (modal, not stacked)
- [ ] Toggle bar visible at all times with clear active indicator
- [ ] Keyboard shortcuts for every overlay
- [ ] Active overlay is reflected in: (a) world tiles, (b) minimap, (c) tooltip priority
- [ ] Heat maps use perceptually uniform color scales, never raw rainbow
- [ ] Color legend with range is always visible when overlay is active
- [ ] Building influence ranges shown on hover/select with clear radius
- [ ] Non-relevant game elements dim/simplify when overlay is active
- [ ] Overlay renders at 40-60% opacity, terrain remains visible
- [ ] Return-to-default is one click or keypress (Escape or re-click active overlay)
- [ ] Colorblind-safe palette option available
- [ ] Overlay switching is instant (< 200ms transition)

---

## 5. Site Inspection Panels

### Gold Standard: Factorio Entity Windows + Anno 1800 Building Menus

Factorio has ~120 entity windows, each following a consistent template: entity name, status, relevant stats, configuration controls. Anno 1800's building menus show production chain status, worker allocation, and upgrade paths in a clean pop-up panel.

### Panel Architecture Patterns

**Pattern A: Inline Tooltip (for simple entities)**
```
Use when: Entity has < 5 stats, no configurable actions
Trigger: Hover
Position: Floating near entity
Content: Name, status, 2-3 key stats
Example: Factorio inserter showing items/sec
```

**Pattern B: Side Panel (for complex entities)**
```
Use when: Entity has 5+ stats, configuration options, or history
Trigger: Click to open, click elsewhere or Escape to close
Position: Fixed right-side panel (Factorio) or left-side panel
Content: Full stat breakdown, action buttons, tabs for sub-views
Example: Anno 1800 production building showing input/output chain
```

**Pattern C: Modal Detail View (for deep inspection)**
```
Use when: Entity has sub-entities, historical data, or complex config
Trigger: Double-click, or "Details" button in side panel
Position: Center screen overlay or dedicated screen
Content: Graphs, detailed breakdowns, management controls
Example: Cities: Skylines building showing visitor stats, upgrade options
```

**The progression:** Hover (instant preview) -> Click (working detail) -> Deep inspect (full management). Each level reveals more, costs more attention.

### Multi-Stat Display Patterns

**For 2-4 stats: Vertical List with Bars**
```
+----------------------------------+
| Lumber Mill          [Active]    |
|----------------------------------|
| Workers:   3/5       [===---]    |
| Output:    12/hr     [=====-]    |
| Storage:   45/100    [==----]    |
|----------------------------------|
| [Upgrade] [Pause] [Demolish]     |
+----------------------------------+
```

**For 5-8 stats: Grouped Sections**
```
+----------------------------------+
| Temple of Dawn       [Active]    |
|================================--|
| PRODUCTION                       |
|   Sanctity:  +4/turn  [====]    |
|   Influence: 12 tiles  [===]    |
|----------------------------------|
| REQUIREMENTS                     |
|   Workers:   2/3       [==-]    |
|   Resources: 5 stone   [===]    |
|----------------------------------|
| EFFECTS                          |
|   Approach:  +2 nearby           |
|   Strain:    -1 nearby           |
|================================--|
| [Upgrade] [Reassign] [Demolish]  |
+----------------------------------+
```
- Group stats by category (production, requirements, effects)
- Dividers between groups
- Actions always at the bottom

**For comparison views (this tile vs. that tile):**

**Option A: Side-by-Side Panel**
```
+------------------+------------------+
| Tile A (3,7)     | Tile B (5,9)     |
|------------------|------------------|
| Approach:  62    | Approach:  41    |
| Sanctity:  84    | Sanctity:  22    |
| Shelter:   41    | Shelter:   78    |
| Strain:    18    | Strain:    55    |
+------------------+------------------+
```
- Pin one tile, then hover another to compare
- Highlight which tile "wins" each stat with subtle color

**Option B: Delta Display**
```
+----------------------------------+
| Comparing: (3,7) vs (5,9)       |
|----------------------------------|
| Approach:  62 vs 41   (+21)     |
| Sanctity:  84 vs 22   (+62)     |
| Shelter:   41 vs 78   (-37)     |
| Strain:    18 vs 55   (-37)     |
+----------------------------------+
```
- Show the difference explicitly
- Green for favorable delta, red for unfavorable

### Context-Sensitive Actions

The inspection panel should show only actions relevant to the current entity AND the current game state:

```
IF building is under construction:
  Show: [Priority Up] [Priority Down] [Cancel]

IF building is active:
  Show: [Pause] [Upgrade] [Demolish]

IF building is paused:
  Show: [Resume] [Demolish]

IF building is damaged:
  Show: [Repair] [Demolish]
```

**Anti-pattern:** Showing all possible actions with some grayed out. This creates visual clutter and forces the player to parse which are available. Show only what's actionable.

**Exception:** If there are exactly 2-3 persistent actions (like Upgrade and Demolish), always showing them is fine -- the consistency helps muscle memory.

### RimWorld Room Stats: A Case Study

RimWorld's room stats system (via Room Sense mod becoming official) demonstrates excellent contextual inspection:
- Stats shown depend on room type: bedrooms show beauty, space, cleanliness; kitchens show cleanliness; storerooms show nothing irrelevant
- Color-coded meters: red/yellow/green for bad/okay/good
- Overlay toggleable with a single button or hotkey
- Stats appear directly ON the room, not in a separate panel

**Lesson:** The best inspection UI is spatially co-located with the thing being inspected, shows only relevant stats, and uses color to communicate quality without requiring number-reading.

### 10/10 Criteria for Site Inspection Panels

- [ ] Three-tier depth: hover preview -> click detail -> deep inspect
- [ ] Consistent template across all entity types (name, status, stats, actions)
- [ ] Stats grouped by category with visual dividers
- [ ] Bars + numbers for every quantitative stat (never just numbers alone)
- [ ] Actions are context-sensitive -- only show what's currently possible
- [ ] Panel position is consistent (always same screen region)
- [ ] Comparison mode available (pin one, hover another)
- [ ] Panel opens/closes without camera jump or viewport disruption
- [ ] Keyboard shortcut to close (Escape) and to cycle through nearby entities (Tab)
- [ ] Panel remembers its state (if you had a tab selected, it stays selected on the next entity)

---

## 6. Cross-Cutting Concerns

### Typography

- **Stat labels:** Short, abbreviated, consistent. "PROD" not "Production Rate Per Turn"
- **Numbers:** Monospace or tabular figures so columns align
- **Units:** Always show units on first occurrence, optional after ("12 wood/turn" then "8" for the next line if units are obvious)
- **Size:** Minimum 12px at standard resolution. Factorio uses a 4px module grid, making 12px = 3 modules

### Color System

Define a project-wide semantic color palette:

```
STATUS:
  Active/Good:    #4CAF50 (green)
  Warning/Medium: #FFC107 (amber)
  Critical/Bad:   #F44336 (red)
  Inactive/Neutral: #9E9E9E (gray)

OVERLAYS:
  High value:     Saturated end of scale
  Low value:      Desaturated/white end of scale
  No data:        Crosshatch or dotted pattern (not gray -- gray is a valid data value)

UI CHROME:
  Background:     Dark, low-contrast (#1A1A2E or similar)
  Text:           High contrast on dark (#E0E0E0)
  Borders:        Subtle (#333344)
  Active element: Accent color (#BB86FC or project-specific)
```

### Responsiveness & Performance

- Tooltips must render in < 150ms. If you need to compute stats, cache them per-tile and invalidate on change
- Overlay re-render on toggle should be < 200ms. Pre-compute overlay data when tile state changes, not when overlay is activated
- Minimap updates should be batched (update every 500ms or on significant change, not every frame)
- Panel animations should be < 200ms or instant. Slow slide-ins waste player time

### Accessibility

- All color-encoded information must also be available as text/numbers
- Provide colorblind-safe palettes (deuteranopia affects ~8% of males)
- Support UI scaling (Factorio's 100%, 125%, 150%, 200% scaling system)
- Keyboard navigation for all panel interactions
- Screen reader support for stat values (even if partial)

---

## 7. Application to The Oracle

Based on the multi-variable tile system described (approach, sanctity, shelter, strain), here are specific recommendations:

### Tile Tooltip (Hover)
```
+---------------------------+
| Sacred Grove  (4, 12)     |
|   APP 62  SAN 84          |
|   SHL 41  STR 18          |
+---------------------------+
```
- 2x2 grid of abbreviated stats with values
- Color-code each number by its quality threshold
- 3 lines, appears in ~100ms, no bars needed at hover level

### Tile Detail Panel (Click)
```
RIGHT PANEL:
+----------------------------------+
| Sacred Grove           (4, 12)   |
| Zone: Sanctified Woodland        |
|==================================|
| Approach:  [=======---]  62/100  |
| Sanctity:  [==========-]  84/100 |
| Shelter:   [=====-----]  41/100  |
| Strain:    [==--------]  18/100  |
|==================================|
| Active Effects:                  |
|   Temple of Dawn: SAN +4, APP +2 |
|   Ancient Tree:   SHL +3         |
|   Pilgrim Trail:  STR +6         |
|----------------------------------|
| [Build] [Inspect Surroundings]   |
+----------------------------------+
```

### Overlay Set
Define overlays mapping to your core stats plus gameplay systems:

| Key | Overlay | Color Scale |
|-----|---------|-------------|
| 1 | Approach | White-to-Blue (sequential) |
| 2 | Sanctity | White-to-Gold (sequential) |
| 3 | Shelter | White-to-Green (sequential) |
| 4 | Strain | White-to-Red (sequential, red = bad) |
| 5 | Building Influence | Per-building radial highlight |
| 6 | Zone Boundaries | Categorical distinct hues |
| 0 | Default (no overlay) | Standard terrain colors |

### Minimap
- Bottom-right, below or beside the overlay toggle bar
- ~8% of viewport
- Shows terrain + zone boundaries by default
- Reflects active overlay coloring
- Click to navigate, scroll to zoom
- Camera rectangle always visible

---

## Summary: The Four Tests

Before shipping any spatial display UI element, ask:

1. **Tufte Test:** "Is there any ink/pixel here that doesn't represent data?" Remove it.
2. **Norman Test:** "Can a new player understand what this shows and how to interact with it within 5 seconds?" If not, improve visibility and mapping.
3. **Rams Test:** "Can I remove anything without losing function?" If yes, remove it.
4. **Alexander Test:** "Does this element relate spatially to the thing it describes? Does it connect to the broader system?" If it's isolated, integrate it.

---

## Sources

### Game References
- [Anno 1800 DevBlog: User Interface](https://www.anno-union.com/devblog-user-interface-2/)
- [Anno 1800 UI on Interface In Game](https://interfaceingame.com/games/anno-1800/)
- [Anno 1800 UI on Game UI Database](https://www.gameuidatabase.com/gameData.php?id=1118)
- [Cities: Skylines II Info Views](https://cs2.paradoxwikis.com/Info_views)
- [Civilization VI Lens System](https://civilization.fandom.com/wiki/Lens_(Civ6))
- [Factorio Friday Facts #212 - The GUI Update](https://www.factorio.com/blog/post/fff-212)
- [Factorio Friday Facts #277 - GUI Progress Update](https://factorio.com/blog/post/fff-277)
- [Factorio Friday Facts #348 - The Final GUI Update](https://factorio.com/blog/post/fff-348)
- [Pharaoh: A New Era - How to Use Overlays](https://gamerdigest.com/pharaoh-a-new-era-how-to-use-overlays/)
- [RimWorld Room Sense Mod](https://steamcommunity.com/sharedfiles/filedetails/?id=1355637255)
- [RimWorld Rooms Wiki](https://rimworldwiki.com/wiki/Rooms)
- [Oxygen Not Included Overlays](https://oxygennotincluded.fandom.com/wiki/Overlays)
- [Foundation Zones](https://wiki.polymorph.games/foundation/Zones)
- [Dwarf Fortress Tile Attributes](https://dwarffortresswiki.org/index.php/DF2014:Tile_attributes)
- [Banished UI Mods](https://www.nexusmods.com/banished/mods/40)
- [Game UI Database - Minimap](https://www.gameuidatabase.com/index.php?scrn=135)

### Design Research
- [Where Should We Place the Mini Map? (Gamedeveloper.com)](https://www.gamedeveloper.com/business/-where-should-we-place-the-mini-map-)
- [Minimaps Research](https://alejandro61299.github.io/Minimaps_Personal_Research/)
- [Mini-Map Design Features as Navigation Aid (MDPI)](https://www.mdpi.com/2220-9964/12/2/58)
- [Strategy Game Tooltips - Patterns Game Prog](https://www.patternsgameprog.com/strategy-game-20-tooltips)
- [Tooltip UI Design Best Practices (Mockplus)](https://www.mockplus.com/blog/post/tooltip-ui-design)

### Design Theory
- [Tufte's Principles of Data-Ink](https://jtr13.github.io/cc19/tuftes-principles-of-data-ink.html)
- [Tufte's Data Design Principles](https://guypursey.com/blog/202001041530-tufte-principles-visual-display-quantitative-information)
- [Data-Ink Ratio Explained (Holistics)](https://www.holistics.io/blog/data-ink-ratio/)
- [Sparklines History by Tufte](https://www.edwardtufte.com/notebook/sparklines-history-by-tufte-1324-to-now/)
- [Don Norman's Principles of Interaction Design](https://medium.com/@sachinrekhi/don-normans-principles-of-interaction-design-51025a2c0f33)
- [Natural Mappings and Stimulus-Response Compatibility (NN/g)](https://www.nngroup.com/articles/natural-mappings/)
- [Dieter Rams: Ten Principles for Good Design](https://www.vitsoe.com/us/about/good-design)
- [Dieter Rams 10 Timeless Commandments (IxDF)](https://www.interaction-design.org/literature/article/dieter-rams-10-timeless-commandments-for-good-design)
- [Christopher Alexander: Pattern Language for Game Design](https://patternlanguageforgamedesign.com/)
- [A Pattern Language for Games (Medium)](https://perspectivesingamedesign.com/a-pattern-language-for-games-3d1c6849a3cd)
