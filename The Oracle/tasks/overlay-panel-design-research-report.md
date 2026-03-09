# Overlay Panels, Modal Screens & Deep-Dive Management Interfaces
## Best-in-Class Standards for Strategy/Management Games

*Companion to: hud-design-research-report.md (HUD, Top Bar, Bottom Strip, Navigation)*
*Research compiled for: Simulation/Management/Building game UI analysis*
*Date: 2026-03-06*

---

## Table of Contents
1. [Design Expert Frameworks (Overlay-Specific Applications)](#1-design-expert-frameworks-overlay-specific-applications)
2. [Overlay Architecture](#2-overlay-architecture)
3. [Information Organization Within Panels](#3-information-organization-within-panels)
4. [Deep Management Screens](#4-deep-management-screens)
5. [Progressive Disclosure in Overlays](#5-progressive-disclosure-in-overlays)
6. [Contextual Actions in Overlays](#6-contextual-actions-in-overlays)
7. [Consolidated Scoring Rubric](#7-consolidated-scoring-rubric)

---

## 1. Design Expert Frameworks (Overlay-Specific Applications)

The foundational frameworks from the HUD report apply here with overlay-specific nuances. This section extends each framework into the overlay/panel domain.

### 1.1 Edward Tufte --- Layering, Small Multiples, Micro/Macro Readings

**Core Overlay Principle:** Overlays exist to add an information layer on top of the base experience. Tufte's concept of *layering and separation* is the foundational technique --- use color, transparency, and spatial grouping to build distinct information layers that the eye can parse independently.

**Small Multiples in Panels:** When displaying multiple entities of the same type (characters, provinces, trade routes, production chains), use repeating visual structures where only the data changes. The viewer's brain locks onto the repeating pattern and focuses on variation. CK3's vassal list, Stellaris's species panel, and Anno 1800's production statistics all use this technique.

**Micro/Macro Readings:** A well-designed overlay panel conveys one message at the macro level (glancing at the panel as a whole) and detailed data at the micro level (reading individual entries). Victoria 3's market screen accomplishes this: the macro reading is "my economy is healthy/struggling" via aggregate color coding, while the micro reading is individual goods prices and supply/demand ratios.

**Applied to Overlays:**
- Use consistent card/row templates across all list-type panels
- Color-code at the aggregate level (green = healthy, red = crisis) so the panel itself is readable before any text is parsed
- Avoid decorative borders, shadows, or textures that compete with data
- Tooltips should add a data layer, never repeat what the panel already shows

**Measurable Test:** Can the player extract the panel's primary message (good/bad/neutral state) within 1 second, before reading any text?

### 1.2 Don Norman --- Conceptual Models & the Gulf of Evaluation

**Core Overlay Principle:** The "gulf of evaluation" is the distance between the system's actual state and the player's understanding of it. Overlays exist to close this gulf. Every overlay panel must answer the question "what is happening and why?" --- not just "what are the numbers?"

**System Image:** Norman's concept of "system image" --- the external representation that helps users build a mental model --- is critical for deep management screens. The overlay IS the system image for complex subsystems like espionage networks, religion, or trade routes. If the overlay's structure doesn't match the system's actual structure, players will build wrong mental models and make wrong decisions.

**Applied to Overlays:**
- Panel structure should mirror the conceptual structure of the system it represents
- If espionage has a hierarchy (network > agents > operations), the panel should have that same hierarchy
- Cause-and-effect relationships must be visually traceable --- if modifier A affects outcome B, they should be visually connected or sequentially ordered
- Status indicators must show both *current state* and *why* (Victoria 3's market screen shows price AND the supply/demand forces driving it)

**Measurable Test:** Can the player trace from any displayed outcome back to its cause within 2 clicks or visual scans?

### 1.3 Steve Krug --- "Don't Make Me Think" in Complex Panels

**Core Overlay Principle:** The most complex management screens must still be "obvious" in their operation. Krug's billboard design principle applies directly --- panels should be scannable, not readable. Headers, groupings, and visual weight should guide the eye before any text is processed.

**Mindless Choices in Overlays:** When an overlay presents action options, the correct choice should be the obvious one in most situations. CK3's character interaction menu sorts options by relevance and dims unavailable ones. The player doesn't think about what's possible --- they see what's available.

**Applied to Overlays:**
- Every panel needs a clear visual hierarchy: title > section headers > data > secondary info
- Available actions must be visually distinct from unavailable ones (not just greyed out text --- distinct spatial treatment)
- Navigation within panels must follow conventions (tabs at top, back button top-left, close button top-right)
- If the player hesitates about what to click next, the panel has failed

**Measurable Test:** Can a player who has never seen the panel identify its purpose and primary action within 5 seconds?

### 1.4 Jenova Chen --- Flow Preservation Across Panel Transitions

**Core Overlay Principle:** Opening an overlay panel is a context switch. Every context switch risks breaking flow. The transition into and out of overlays must be instantaneous and reversible. The player should feel like they're "leaning in" to look closer, not "leaving" the game.

**Applied to Overlays:**
- Panel open/close animations should be < 200ms
- The game world should remain partially visible behind non-fullscreen overlays
- Escape key must always close the current panel (no exceptions)
- Panel state should persist when closed and reopened (don't reset scroll position, tab selection, or filter state)
- Audio should continue (perhaps muted) during overlay display --- silence signals "you left the game"

**Measurable Test:** Does the player feel they are still "in the game" while using the overlay? Can they return to gameplay in < 500ms total (animation + mental re-orientation)?

### 1.5 Jakob Nielsen --- 10 Usability Heuristics Applied to Overlays

**The most relevant heuristics for overlay panels:**

1. **Visibility of system status:** Panels must show loading states, progress indicators, and confirmation of changes immediately.
2. **Match between system and real world:** Use language and concepts familiar to the genre. "Levy troops" not "instantiate military units."
3. **User control and freedom:** Every panel must have an obvious exit. Undo must be available for destructive actions.
4. **Consistency and standards:** All panels should use the same tab style, button placement, scrollbar behavior, and color coding.
5. **Recognition rather than recall:** Show options in context rather than requiring players to remember codes, names, or sequences. CK3's character finder lets you browse and filter rather than type names from memory.
6. **Error prevention:** Dangerous actions (declare war, execute prisoner, demolish building) need friction. Safe actions (view information, switch tabs) need zero friction.

**Measurable Test:** Run each panel through all 10 heuristics. Score each 1-5. Average >= 4 for 10/10 quality.

### 1.6 Luke Wroblewski --- Progressive Disclosure as Panel Architecture

**Core Overlay Principle:** Wroblewski's mobile-first constraint thinking translates directly to overlay design: treat the overlay as a constrained viewport that must prioritize ruthlessly. The smallest useful view comes first. Detail is earned through interaction, not dumped on arrival.

**Three-Tier Overlay Model:**
- **Tier 1 (Summary):** What the player sees on the HUD or in a tooltip. 2-3 data points. "Economy: Strong (+12 gold/turn)."
- **Tier 2 (Overview Panel):** What appears when the player opens the management screen. 10-20 data points organized into sections. Categories, summaries, aggregate indicators.
- **Tier 3 (Detail View):** What appears when the player drills into a specific item. Full data dump, historical charts, every modifier and contributing factor.

**Applied to Overlays:**
- Never open directly to Tier 3. Always start at Tier 2.
- Each tier must be accessible from the previous tier in exactly 1 click.
- Breadcrumbs or back buttons must allow return to higher tiers without data loss.
- The transition from Tier 2 to Tier 3 should be contextual (clicking a specific item), not global (a "show details" toggle that changes everything).

**Measurable Test:** Count the tiers of information disclosure. Must be >= 3. Each tier must add genuine value, not just more numbers.

---

## 2. Overlay Architecture

### 2.1 Modal vs Non-Modal Overlays

**When to use modal overlays (blocks interaction with the game world):**
- Decisions that are irreversible or high-consequence (declaring war, founding a religion, passing a law)
- Multi-step workflows that require sequential attention (creating a trade route, designing a custom faith)
- Events that demand narrative attention (story events in Old World, crisis decisions in Frostpunk)
- Confirmations for destructive actions

**When to use non-modal overlays (game world remains interactive):**
- Information-only panels (character sheets, production statistics, market data)
- Monitoring panels the player wants to keep open while playing (outliner in Stellaris, pinned resources in Victoria 3)
- Overlays that modify the viewport itself (heat maps in ONI, info views in Cities: Skylines)
- Quick-reference panels (tech tree, policy overview)

**Gold Standard: Crusader Kings III**
CK3 uses a hybrid approach masterfully:
- Character sheets, realm panels, and information screens are non-modal --- the player can still interact with the map, click other characters, and receive notifications
- Event popups are semi-modal --- they dim the background and demand a choice, but the game continues (time can still be controlled)
- Certain critical decisions (founding a faith, declaring a holy war) are fully modal with backdrop dimming

**Gold Standard: Victoria 3**
Victoria 3 pins situational panels to the right side of the screen as non-modal widgets. Ongoing diplomatic plays, law passages, and revolutions appear as persistent, non-blocking panels that the player can monitor without interrupting gameplay. This is an excellent pattern for long-duration processes.

**Design Principle:** Default to non-modal. Only go modal when the decision is consequential enough that background gameplay would be a distraction, not an asset.

**Anti-Patterns:**
- Using modal overlays for information-only screens (forces unnecessary dismiss actions)
- Using non-modal overlays for irreversible decisions (player might click past them accidentally)
- Inconsistent modality --- some info screens are modal, others aren't, with no discernible logic
- Modal overlays that don't pause or slow the game clock in real-time strategy games

**10/10 Criteria:**

| Criterion | Weight | Standard |
|-----------|--------|----------|
| Modality matches consequence | 30% | Information = non-modal; irreversible actions = modal; always consistent |
| Non-modal panels allow world interaction | 25% | Map clicks, camera movement, and notifications all work while panels are open |
| Modal panels justify their interruption | 20% | Every modal contains a decision or acknowledgment, never just information |
| Clear visual distinction between modal/non-modal | 15% | Modal = backdrop dim + centered panel; non-modal = side panel + no dim |
| Game clock behavior is appropriate | 10% | Modal panels pause or offer pause; non-modal panels don't affect time |

### 2.2 Simultaneous Panel Management

**How many panels should be simultaneously possible?**

The answer depends on the game's information density requirements:

- **Minimum:** 1 primary panel + the HUD (suitable for focused games like Frostpunk)
- **Recommended:** 1 primary panel + 1-2 pinnable side panels + the HUD (CK3 model)
- **Maximum:** 1 primary panel + 3-4 pinnable widgets + the HUD (Stellaris/Victoria 3 model)
- **Never:** Unlimited stacking panels with no management (leads to Dwarf Fortress-style clutter)

**Gold Standard: Stellaris**
Stellaris allows the player to pin certain elements to the outliner (right side of screen): fleets, planets, leaders, situations. These persist as compact entries while the player opens and closes full panels. The outliner acts as a "dashboard" layer between the HUD and full overlays.

**Gold Standard: Victoria 3**
Victoria 3's right-side pinning of active situations (diplomatic plays, law passages, revolutions) creates a natural "things I'm monitoring" column that doesn't compete with full management panels opened from the left or center.

**Panel Collision Rules:**
1. Opening a new full-size panel should close the previous full-size panel (unless explicitly pinned)
2. Pinned widgets should survive panel changes
3. If two panels would overlap, the new one takes priority and the old one auto-closes or minimizes
4. The player must always be able to see the game world --- if panels would cover 100% of the viewport, prevent the last panel from opening

**Anti-Patterns:**
- Unlimited panel stacking with no collision management
- Panels that open on top of each other with no z-order management
- No way to close all panels at once (Escape should cascade-close: innermost first, then outward)
- Pinned panels that can't be unpinned or repositioned

**10/10 Criteria:**

| Criterion | Weight | Standard |
|-----------|--------|----------|
| Clear primary/secondary panel hierarchy | 30% | One primary panel at a time; secondary panels are compact and non-competing |
| Pin/unpin functionality | 25% | Key monitoring widgets can persist across panel changes |
| Panel collision handling | 20% | New panels auto-close or minimize conflicting panels gracefully |
| Cascade close (Escape) | 15% | Escape closes innermost panel first, then next, then returns to gameplay |
| Viewport preservation | 10% | Game world always occupies >= 40% of screen even with maximum panels open |

### 2.3 Backdrop and Dimming Patterns

**Why dimming matters:** Backdrop dimming is a visual signal that says "this content is temporarily subordinate." It reduces visual competition between the overlay and the game world, focuses attention on the panel, and communicates modality.

**Dimming Levels:**

| Level | Opacity | Use Case | Example |
|-------|---------|----------|---------|
| None (0%) | Fully transparent | Non-modal info panels, side panels | CK3 character sheet |
| Light (20-30%) | Slight tint | Semi-modal event popups, optional decisions | Old World event cards |
| Medium (40-60%) | Clear dim | Important decisions, multi-step workflows | CK3 faith creation |
| Heavy (70-85%) | Near-opaque | Critical/irreversible decisions, narrative moments | Frostpunk law signing |
| Full (90-100%) | Opaque | Cutscenes, loading screens, full-screen menus | Settings/save/load |

**Gold Standard: Old World**
Old World's event panels appear with a medium dim backdrop and feature richly illustrated character cards. The dimming focuses attention on the narrative moment while the illustration quality justifies the interruption. Players don't resent the modal because the content is worth the attention.

**Design Principle:** Dimming intensity should be proportional to the decision's weight. Light decisions get light dimming; world-altering decisions get heavy dimming. The player subconsciously learns to calibrate their attention level based on how dark the backdrop gets.

**Anti-Patterns:**
- Binary dimming (either full dim or no dim) with no intermediate states
- Dimming non-modal panels (sends conflicting signals about interactivity)
- No dim on genuinely modal panels (player thinks they can interact with the world and gets confused)
- Dim animation that takes > 200ms (feels sluggish)

### 2.4 Overlay Size Conventions

**Panel size should match information density and decision weight:**

| Panel Type | Width | Height | Screen Coverage | Example |
|------------|-------|--------|-----------------|---------|
| Quick tooltip | 200-350px | Auto-fit | 5-10% | CK3 character tooltip |
| Side panel | 300-450px | Full height | 20-30% | Stellaris empire interface |
| Standard overlay | 500-800px | 400-700px | 25-45% | CK3 character sheet |
| Wide management screen | 900-1200px | 600-800px | 45-65% | Victoria 3 market screen |
| Full management dashboard | 80-90% width | 80-90% height | 65-80% | Anno 1800 statistics |
| Full-screen menu | 100% | 100% | 100% | Civ VI tech tree |

**Gold Standard: Crusader Kings III**
CK3's character sheet occupies roughly 40-50% of the screen width, anchored to the right side. This leaves the map visible on the left --- critical because the player often wants to reference geographic information while reading character data. The panel height fills most of the vertical space, using internal scrolling for overflow content.

**Design Principle (Tufte):** The panel should be exactly large enough to display its Tier 2 information without scrolling. If scrolling is required at Tier 2, either the panel is too small or it contains too much Tier 2 information (some should be demoted to Tier 3).

**Anti-Patterns:**
- Small panels with excessive scrolling (information buried below the fold)
- Oversized panels for simple information (panel size should match content density)
- Panels that change size dynamically based on content (unstable visual reference)
- Centered panels when side-anchored would preserve game world visibility

### 2.5 Escape/Close Patterns

**Universal Standard (violate at your peril):**
1. **Escape key** closes the topmost/innermost panel. Always. No exceptions.
2. **Clicking outside** the panel closes it (for non-critical panels). Modal panels with backdrop should close on backdrop click ONLY if the content is information-only, never if there are unsaved changes.
3. **X button** in the top-right corner of the panel. Always visible, always in the same position relative to the panel.
4. **Right-click** on the panel background closes it (CK3 convention, appreciated by power users).
5. **Same-key toggle:** If a panel was opened with a hotkey (e.g., 'C' for character), pressing that key again should close it.

**Cascade Close Order:**
Escape closes panels from innermost to outermost:
1. First press: Close sub-panel/tooltip/dropdown within the overlay
2. Second press: Close the overlay panel itself
3. Third press: Close the HUD category/menu that spawned the overlay
4. Final press: Open the pause/system menu

**Gold Standard: Crusader Kings III**
CK3 nails this. Every panel has an X button, responds to Escape, closes on right-click, and follows the cascade pattern. Nested panels (e.g., viewing a character's liege's realm from within a character sheet) close innermost-first. The player never feels "trapped" in a panel hierarchy.

**Anti-Patterns:**
- Escape doing nothing (player feels trapped)
- Escape closing ALL panels at once instead of cascading (disorienting, loses context)
- No X button or X button in inconsistent positions
- Panels that can't be closed without making a choice (unless genuinely modal for a required decision)
- Close buttons that are small, visually subtle, or positioned inconsistently

**10/10 Criteria (Escape/Close):**

| Criterion | Weight | Standard |
|-----------|--------|----------|
| Escape always works | 30% | Escape closes topmost panel in every context, no exceptions |
| Cascade close order | 25% | Nested panels close innermost-first, system menu is always last |
| X button consistency | 20% | Every panel has X button in top-right corner, same size and style |
| Same-key toggle | 15% | Hotkey that opens a panel also closes it |
| Click-outside behavior | 10% | Non-critical panels close on outside click; modal panels don't |

---

## 3. Information Organization Within Panels

### 3.1 Tab Systems Within Overlays

**When to use tabs:**
- The panel serves multiple related-but-distinct purposes (CK3 character sheet: Biography, Family, Relations, Claims)
- Information can be cleanly categorized into 3-7 groups
- The player doesn't need to see all categories simultaneously
- Each tab represents a different "question" the player might ask about the same subject

**Gold Standard: Crusader Kings III Character Sheet**
CK3's character sheet has tabs across the top of the panel: Diplomacy, Martial, Stewardship, Intrigue, Learning (skill-based), plus Lifestyle, Relations, and other contextual tabs. Each tab contains a focused view of that domain:
- Skill tab shows the stat, contributing modifiers, and relevant actions
- Relations tab shows family tree, vassals, court
- Lifestyle tab shows focus, perks, and progress

The tabs use icon + text labels, have clear active/inactive states, and maintain scroll position independently (switching tabs doesn't reset the other tab's state).

**Gold Standard: Stellaris Empire Interface**
Stellaris allows tab reordering within the empire interface. Players can drag tabs to customize their information hierarchy. Each tab (Government, Leaders, Species, Factions, Relics, Traditions) is a self-contained view with its own internal organization.

**Tab Design Rules:**

| Rule | Rationale |
|------|-----------|
| 3-7 tabs maximum per panel | Cognitive chunking limit; beyond 7, use sub-tabs or restructure |
| Icon + text labels | Dual encoding for scannability; icon-only fails for unfamiliar panels |
| Active tab visually distinct | Color fill, underline, or tab "elevation" --- must pass the 1-second test |
| Tab order is meaningful | Most-used or most-important tabs first (left-to-right in LTR languages) |
| Tabs remember state | Switching away and back preserves scroll position, selections, filters |
| Keyboard navigation | Left/right arrow keys or numbered shortcuts cycle tabs |
| No nested tab bars | If you need sub-tabs within tabs, use a different pattern (sections, accordion, or drill-down) |

**Anti-Patterns:**
- More than 8 tabs in a single row (forces scrolling or tiny labels)
- Tabs within tabs within tabs (CK3 avoids this; Victoria 3 occasionally falls into it)
- Tab labels that are abbreviations or icons without tooltips
- Tabs that look like buttons (ambiguous affordance --- "will this navigate or act?")
- Tab state resetting when the panel is closed and reopened

### 3.2 Section Headers and Collapsible Groups

**When to use collapsible sections instead of tabs:**
- The information is hierarchical rather than categorical (sections of a single document vs. separate views)
- The player benefits from seeing multiple sections simultaneously
- Sections vary dramatically in size (some have 2 lines, some have 20)
- The panel represents a single entity's full profile (all information relates to one thing)

**Gold Standard: Anno 1800 Production Statistics**
Anno 1800's statistics panel uses collapsible sections for each island and each production category. The player can expand "Schnapps Production" to see individual distillery output, or collapse it to see only the aggregate. This allows the player to focus on problem areas while keeping the full picture visible.

**Section Design Rules:**

| Rule | Rationale |
|------|-----------|
| Section headers must be visually heavy | Bold text, background color, or horizontal rule --- must be instantly scannable |
| Collapse/expand state persists | Closing and reopening the panel remembers which sections were expanded |
| Aggregate data shown in collapsed state | The header itself shows a summary (e.g., "Army: 4 units, 2400 troops") |
| Expand/collapse indicator | Chevron or +/- icon, always in the same position relative to the header |
| Sections can be individually collapsed | Not just "expand all" / "collapse all" (though those should also exist) |
| Default state is thoughtful | Most important sections expanded, secondary sections collapsed |

**Anti-Patterns:**
- Sections that are always expanded with no collapse option (information overload)
- Collapsed sections that show no summary data (player must expand to learn anything)
- Inconsistent collapse behavior (some sections collapse, others don't)
- No "collapse all" / "expand all" shortcut for panels with many sections

### 3.3 Stat Bars, Meters, and Data Visualization

**The hierarchy of data visualization in game panels:**

| Visualization | Best For | Example |
|--------------|----------|---------|
| **Single number** | Exact values that need precision | Gold: 1,453 |
| **Number + trend arrow** | Values where direction matters | Gold: 1,453 (+12/turn) |
| **Progress bar** | Values with a known maximum | XP: 340/500 (68%) |
| **Segmented bar** | Values with thresholds/tiers | Happiness: [green][green][yellow][red] |
| **Pie/donut chart** | Composition of a whole | Population by religion: 40% / 35% / 25% |
| **Sparkline** | Historical trend in minimal space | Gold income, last 20 turns: [tiny line chart] |
| **Small multiples** | Comparing same metric across entities | All provinces: income bar per province |
| **Heat map** | Spatial distribution of a value | ONI temperature overlay, CK3 realm development |

**Gold Standard: Victoria 3 Market Screen**
Victoria 3's market screen combines multiple visualization types for each good:
- Icon + name (identification)
- Price number + color coding (current state: green = balanced, red = shortage)
- Supply/demand bars facing each other (visual representation of market forces)
- Buy/sell orders list (detailed breakdown)
- Historical price graph accessible via drill-down (trend over time)

This layered approach gives the player macro-level understanding at a glance and micro-level detail on demand.

**Gold Standard: Frostpunk 2 Heat Map Overlay**
Frostpunk 2's heat map overlay color-codes each district from blue (freezing) to red (warm), overlaid directly on the game world. The player sees spatial patterns instantly --- "the north districts are all cold, the south is warm" --- without reading any numbers. Numbers appear on hover for precision.

**Data Visualization Rules:**

| Rule | Rationale |
|------|-----------|
| Color carries semantic meaning | Green = good, Red = bad, Yellow = warning. Never decorative color. |
| Numbers need context | "Gold: 1,453" is meaningless. "Gold: 1,453 (+12/turn, 24 turns to goal)" is actionable. |
| Bars need scales | A bar at 60% fill means nothing if the player doesn't know the maximum. |
| Trends beat snapshots | Show direction of change, not just current value. Sparklines are powerful. |
| Density matches importance | Critical metrics get more visual space; secondary metrics get compact representations. |

**Anti-Patterns:**
- Using color purely for decoration (red borders that don't mean "danger")
- Progress bars without labels or scale markers
- Pie charts for more than 5 categories (becomes unreadable)
- Historical data available only in a separate screen (should be accessible via hover/expand)
- Inconsistent color coding across different panels (red means "bad" in one panel, "hostile faction" in another)

### 3.4 Text Density Limits

**When is too much text?**

The text density threshold depends on the panel's purpose:

| Panel Type | Max Text Per View | Rationale |
|------------|-------------------|-----------|
| Tooltip | 3-5 lines (50-80 words) | Player is hovering, not reading. Must parse in 2-3 seconds. |
| Event/narrative panel | 80-150 words visible | Player is in "reading mode" but shouldn't feel like a novel. |
| Info section within a panel | 20-40 words per section | Scanning, not reading. Headers do the work. |
| Detailed breakdown | 100-300 words with structure | Player has deliberately drilled down. Structure with headers/lists. |
| Full documentation/lore | Unlimited (with scroll) | Player is in reference mode. Good typography and spacing essential. |

**Gold Standard: Old World Event Panels**
Old World's events present 60-120 words of narrative text alongside character portraits, relationship indicators, and 2-4 choice buttons. The text is large enough to read comfortably, formatted with line breaks, and never exceeds what fits in the panel without scrolling. Each event reads like a vignette, not an essay.

**Gold Standard: Crusader Kings III Tooltips**
CK3's tooltips are information-dense but structured: a header with the key stat, 2-3 lines of explanation, then a list of modifiers contributing to the value. The modifier list uses icons + text + numbers in aligned columns. The tooltip itself is a well-organized micro-panel.

**Text Design Rules:**

| Rule | Rationale |
|------|-----------|
| Scannable structure | Headers, bullet points, and bold keywords. Never a wall of paragraph text. |
| Visual hierarchy in text | Title (18-20px) > Section header (14-16px) > Body (12-14px) > Caption (10-12px) |
| Line length: 45-75 characters | Optimal reading speed per typographic research |
| Line spacing: 1.4-1.6x font size | Readability in low-attention scanning contexts |
| Contrast ratio >= 4.5:1 | WCAG AA standard; critical for readable text on stylized backgrounds |

**Anti-Patterns:**
- Paragraph text in tooltips (tooltips are for structured data, not prose)
- Text-heavy panels with no visual anchors (headers, icons, dividers)
- Font sizes below 11px at 1080p (unreadable for many players)
- Important information buried in the middle of a text block
- Lore text and mechanical text using the same visual treatment (should be clearly distinct)

### 3.5 Action Button Placement Within Info Panels

*This is detailed further in Section 6, but the organizational principle is addressed here.*

**The "Read-Then-Act" Principle:** Information panels follow a top-to-bottom reading flow. The player reads/scans information first, then wants to act. Therefore:
- **Information flows top-to-bottom**
- **Actions live at the bottom of the panel or bottom of each section they relate to**
- **The primary action is the rightmost or most prominent button in the action row**
- **Destructive actions are visually separated (different color, extra spacing, or positioned after a divider)**

**Gold Standard: CK3 Character Interaction**
CK3 places a row of interaction buttons at the bottom of the character panel: common actions (Send Gift, Arrange Marriage, Declare War) in a horizontal strip. Right-clicking the character portrait opens a full interaction menu. The most common actions are always 1-click accessible; the full menu is 2 clicks.

---

## 4. Deep Management Screens

### 4.1 Espionage & Spy Networks

**How CK3 handles espionage:**
CK3's Intrigue system uses the character's Intrigue skill page as the management hub. The Spymaster council position is assigned from the council panel. Schemes (murder, abduction, seduction) are initiated from the target character's interaction menu and tracked in a dedicated "Schemes" panel showing progress percentage, agents recruited, and success chance.

Key UI patterns:
- Scheme progress shown as a percentage bar with tooltips explaining modifiers
- Agent recruitment displayed as character portraits with their contribution value
- Discovery risk shown as a secondary bar (what happens if you fail)
- Active schemes listed in the character's intrigue panel as a "things I'm doing" section

**How Stellaris handles espionage:**
Stellaris's espionage system assigns envoys to build spy networks in target empires. The UI shows:
- Infiltration level as a progress bar (determines what operations become available)
- Available operations unlocked at specific infiltration thresholds
- Each operation shows: cost, difficulty, required infiltration level, potential outcomes
- The Espionage tab in the empire diplomatic view centralizes all active networks

Stellaris's own developers acknowledged the system "isn't satisfying" due to being hard to track, low-impact, and lacking counterplay. The UI was part of the problem: spy networks were scattered across multiple target empire screens rather than centralized.

**Lessons and Gold Standard Pattern for Espionage UI:**
- **Centralized dashboard:** One screen showing ALL active espionage operations across all targets, not per-target
- **Network-as-entity:** Each spy network should have its own "character card" showing health, size, risk level
- **Operation pipeline:** Pending > Active > Completed operations, like a project management board
- **Risk visualization:** Both success chance and discovery chance as dual bars --- the player should see the trade-off visually
- **Notification integration:** Espionage events feed into the main notification system with appropriate urgency tiering

**Anti-Patterns:**
- Scattering espionage data across multiple unrelated panels
- No centralized view of all active operations
- Operations with unclear or hidden success/failure factors
- No feedback loop showing the player what their spy network actually accomplished

### 4.2 Religion & Faith Management

**How CK3 handles religion:**
CK3's religion panel is accessible from the bottom-left icon row. It shows:
- The player's current faith with its icon and name
- Core tenets displayed as icon cards with tooltip descriptions
- Fervor meter (a single percentage that indicates the faith's internal strength)
- List of faith doctrines organized by category (marriage, clergy, crimes, etc.)
- Holy sites shown on the map with direct navigation links
- "Create New Faith" button at the bottom for reformation

The faith creation screen is a masterclass in complex configuration UI:
- Step-by-step process: choose base religion > modify tenets > set doctrines > name the faith
- Cost displayed prominently and updates dynamically as choices change
- Each tenet/doctrine choice shows a tooltip explaining mechanical effects
- A summary sidebar shows the current configuration at all times

**How Old World handles religion:**
Old World ties religion to characters and cities rather than a global system. Each city has a dominant religion displayed on its city card. Religious leaders are characters with their own portraits and relationships. This makes religion tangible (it's about people and places, not abstract menus) but limits the depth of religious management UI.

**Gold Standard Pattern for Religion/Faith UI:**
- **Identity panel:** The faith's visual identity (icon, color, name) prominently displayed --- this is "who we are"
- **Doctrine grid:** A structured grid of doctrine categories, each with the current selection and alternatives
- **Modifier summary:** A clear listing of all gameplay effects the current faith provides
- **Geographic view:** Map integration showing where the faith is practiced, where it's spreading, where it's threatened
- **Reformation/creation:** A guided, multi-step configuration wizard with live cost updates and outcome previews

### 4.3 Economy & Trade Dashboards

**How Anno 1800 handles economy:**
Anno 1800's statistics menu (added post-launch, indicating its importance was initially underestimated) contains five tabs:
- **Islands:** Per-island production and consumption overview
- **Population:** Population counts and satisfaction levels by tier
- **Production:** Supply/demand bars for every good, color-coded (green = surplus, red = deficit)
- **Finances:** Income and expenses broken down by category
- **Trade:** Active trade routes with profit/loss per route

The production statistics screen uses green and blue bars as the most accurate indication of whether there is under- or overproduction. This is a small-multiples approach: every good gets the same visual treatment (icon + supply bar + demand bar), and the player scans for anomalies.

**How Victoria 3 handles economy:**
Victoria 3's market screen is the most information-dense economic panel in modern strategy games:
- Goods listed with price, buy orders, sell orders, and balance
- Color coding shows market pressure (too much supply vs. too much demand)
- Filters by goods category, market member, trade status
- Drill-down into any good shows production buildings, trade routes, and price history
- The market is a distinct entity --- different markets have different prices, adding a spatial/strategic dimension

Victoria 3's political economy panels layer additional information:
- Interest group support affected by economic conditions (political consequence of economic decisions)
- Laws panel shows economic policies with their effects on different groups
- Budget panel shows revenue/expense with drill-down into each category

**Gold Standard Pattern for Economy/Trade UI:**
- **Dashboard overview:** Top-level screen showing aggregate indicators (total income, balance, trend)
- **Goods matrix:** Small-multiples view of all goods with supply/demand visualization
- **Trade route manager:** List of active routes with profit/loss, goods transferred, and ship assignments
- **Historical charts:** Line graphs showing key economic metrics over time (accessible via drill-down)
- **Problem detection:** Automatic highlighting of goods in deficit or routes running at a loss
- **Cross-reference links:** Clicking a good shows which buildings produce/consume it; clicking a building shows its economic contribution

**Anti-Patterns:**
- Economy screens that show only current values without trends
- No aggregate "health indicator" forcing the player to check every good individually
- Trade route management that requires navigating to each route's endpoints separately
- Financial information spread across multiple unrelated screens

### 4.4 Character Roster Management

**How CK3 handles character roster:**
CK3's character finder is a filterable, sortable list panel that displays character portraits in a grid. Filters include:
- Relationship to player (vassal, courtier, foreign ruler)
- Traits, skills, religion, culture
- Available for marriage, available for recruitment

Each character in the list shows a portrait, name, key trait icons, and primary title. Clicking opens the full character sheet. The character sheet itself is the deep-dive management interface.

**How Total War: Three Kingdoms handles characters:**
TW:TK's Court panel displays all characters in the faction with their portraits, roles, and satisfaction levels. Characters are grouped by role (generals, administrators, candidates). The character card shows:
- Portrait with emotional expression
- Key stats as small icons with numbers
- Satisfaction level as a color-coded indicator
- Special abilities/traits
- Relationship lines to other characters (visual network)

**Gold Standard Pattern for Character Roster:**
- **Filterable grid view:** Characters displayed as portrait cards in a grid, with powerful filtering and sorting
- **Card design:** Each card shows portrait + name + role + 2-3 key stats + status indicators
- **Relationship visualization:** Lines, icons, or grouped placement showing who is connected to whom
- **Drill-down:** Clicking a character opens a full detail panel without leaving the roster context (split view or side panel)
- **Bulk actions:** Select multiple characters for common operations (assign, dismiss, compare)
- **Sort options:** By skill, by loyalty, by age, by relationship strength

**Anti-Patterns:**
- Character lists that show only names (no portraits or visual differentiation)
- No filtering or sorting options in large rosters (> 20 characters)
- Roster view that closes when you click a character (should split or side-panel)
- Relationship information only available from individual character screens, not visible at the roster level

### 4.5 Excavation & Archaeology Management

**How Stellaris handles archaeology:**
Stellaris's archaeological sites are tied to specific star systems. Each site appears in the Situation Log (a centralized panel for ongoing events/projects). The archaeology UI shows:
- Site name and location (star system, with click-to-navigate)
- Current chapter / total chapters (progress indicator)
- Assigned scientist with their portrait and relevant skill level
- Difficulty rating for the current chapter
- Narrative text for completed chapters (story accumulation)
- "Excavate" button to begin/resume, with cost and time estimate

The multi-chapter structure is key: each chapter completion triggers a narrative event panel that tells part of the story, then the player decides whether to continue excavating. This turns archaeology into a narrative arc, not just a progress bar.

**Gold Standard Pattern for Excavation/Archaeology UI:**
- **Central log:** All active and discovered sites listed in one panel (Stellaris's Situation Log approach)
- **Progress visualization:** Multi-chapter progress shown as a segmented bar or step indicator
- **Narrative integration:** Completed chapters visible as expandable story sections within the site panel
- **Resource assignment:** Assigned personnel shown with portraits and relevant skill contributions
- **Risk/reward display:** Clear indication of what resources are required and what rewards are possible
- **Map integration:** Click-to-navigate from the log entry to the site's location in the game world
- **Discovery notification:** New sites appear as mid-priority notifications with clear "investigate" call-to-action

**Anti-Patterns:**
- Archaeology sites that are only discoverable by manually exploring the map (no log or list)
- No narrative payoff for completing chapters (just resources, no story)
- Progress bars without chapter breakdowns (player can't estimate time commitment)
- Assigned personnel not visible from the site panel (requires navigating to a separate screen)

---

## 5. Progressive Disclosure in Overlays

### 5.1 Summary vs Detail View

**The Three-Tier Model (expanded from Section 1.6):**

**Tier 1 --- Glance (HUD/Tooltip Level):**
- Visible without opening any panel
- 2-5 data points maximum
- Answers: "Do I need to pay attention to this right now?"
- Format: icon + number, color-coded status indicator, alert badge
- Example: CK3's character portrait on the map shows health bar, opinion icon, trait icons

**Tier 2 --- Overview (Panel Level):**
- Visible when the management panel is opened
- 10-25 data points organized into sections
- Answers: "What is the current state and what are my options?"
- Format: tabbed panel with section headers, summary statistics, action buttons
- Example: CK3's character sheet showing all skills, traits, relationships, and available interactions

**Tier 3 --- Deep Dive (Drill-Down Level):**
- Visible when the player clicks into a specific aspect of the overview
- 25-100+ data points with full modifier breakdowns
- Answers: "Why is this value what it is, and how can I change it?"
- Format: expanded section, sub-panel, or full detail view with modifier lists, historical data, and cross-references
- Example: CK3's tooltip on a skill value showing every modifier source (traits, lifestyle perks, spouse bonus, court physician, etc.)

**When each tier should be the default view:**

| Context | Default Tier | Rationale |
|---------|-------------|-----------|
| Player is building/managing | Tier 1 on HUD, Tier 2 on click | Needs to act, not analyze |
| Player is strategizing/planning | Tier 2 on open | Needs to understand state before deciding |
| Player is debugging/optimizing | Tier 3 accessible | Needs full modifier breakdown |
| Player is reading narrative | Tier 2 (event text) | Needs enough context, not all data |

**Gold Standard: Victoria 3's Lens System**
Victoria 3 demonstrates all three tiers seamlessly:
- Tier 1: Map color-coding shows economic health at a glance (green provinces = prosperous)
- Tier 2: Clicking a state opens a panel showing population, production, and key metrics
- Tier 3: Clicking a specific good within the state panel shows full supply chain, modifiers, and contributing buildings

### 5.2 Drill-Down Navigation Patterns

**The key tension:** Drill-down depth vs. context preservation. Every level deeper the player goes, the more they lose sight of where they came from. The solution is to maintain breadcrumbs or visual context at every level.

**Pattern 1: Replace-in-Place (CK3 Model)**
- Clicking a linked entity within a panel replaces the panel content with that entity's information
- Back button (or right-click) returns to the previous view
- History maintained as a stack (like a web browser)
- Pros: Simple, uses existing panel space efficiently
- Cons: Loses the previous context entirely; player must remember what they were looking at

**Pattern 2: Side-by-Side Drill (Anno 1800 Model)**
- Clicking a linked entity opens a secondary panel beside the current one
- Both panels visible simultaneously for comparison
- Closing the secondary returns focus to the primary
- Pros: Preserves context, enables comparison
- Cons: Uses more screen space, can crowd the viewport

**Pattern 3: Inline Expansion (Victoria 3 Model)**
- Clicking an item in a list expands it in-place, pushing other items down
- The expanded section shows Tier 3 detail within the Tier 2 panel
- Clicking again collapses it
- Pros: No navigation, no context loss, spatially logical
- Cons: Can make the panel very long; scroll position must be managed

**Pattern 4: Modal Sub-Panel (Old World Model)**
- Clicking a linked entity opens a focused sub-panel over the current panel
- The sub-panel has its own close button; closing returns to the parent
- Parent panel is dimmed but visible behind the sub-panel
- Pros: Clear hierarchy, focused attention on the drilled-down item
- Cons: Can feel "stacked" with too many levels; maximum 2 deep

**Recommended Hybrid:**
- Use Inline Expansion for Tier 2 > Tier 3 transitions within a panel
- Use Replace-in-Place for entity-to-entity navigation (character A's panel > character B's panel)
- Use Side-by-Side for explicit comparison modes
- Never go more than 3 levels deep without a breadcrumb trail

### 5.3 Breadcrumb Navigation Within Panels

**When breadcrumbs are necessary:**
- The panel supports navigation between entities (character > liege > liege's realm > realm capital)
- The player can reach content that is 3+ clicks from the panel's starting view
- The panel replaces content rather than expanding in-place

**Breadcrumb Design Rules:**

| Rule | Implementation |
|------|----------------|
| Position | Top of the panel, below the title bar, above the content area |
| Format | Entity Type: Name > Entity Type: Name > Current View |
| Clickable | Each breadcrumb segment is clickable to return to that level |
| Truncation | If > 4 segments, collapse middle segments with "..." |
| Visual style | Smaller text, muted color, separator characters (> or /) |
| Keyboard shortcut | Backspace or Alt+Left navigates up one breadcrumb level |

**Gold Standard: CK3 Character Navigation**
CK3 maintains a navigation history when the player follows links between characters. The back arrow in the panel header returns to the previous character. While CK3 uses a back-button rather than a full breadcrumb trail, the principle is the same: the player can retrace their path.

**Anti-Patterns:**
- No way to return to a previous view after drilling down
- Breadcrumbs that show only the current level, not the path
- Drill-down navigation that resets the panel to its root state when closed and reopened
- Deep navigation without any history mechanism (player gets "lost" in the panel)

### 5.4 Expandable Sections vs Separate Sub-Panels

**Decision framework:**

| Choose Expandable Sections When... | Choose Separate Sub-Panels When... |
|------------------------------------|------------------------------------|
| Content relates to the same entity | Content is about a different entity |
| Expanded content is 5-15 lines | Expanded content is 20+ lines |
| Player may want to compare sections | Player wants focused attention |
| Content is additive (more detail) | Content is a different view (different data) |
| 3-6 expandable sections per panel | Sub-panel is a standalone management screen |

**Gold Standard: Stellaris Species Panel**
Stellaris uses expandable sections within the species panel to show traits, rights, and population distribution for each species. The player can expand one species to see details while keeping others collapsed for comparison. This is the correct use of expandable sections: same entity type, moderate detail, comparison-friendly.

**Gold Standard: CK3 Lifestyle Panel**
CK3's Lifestyle panel uses a separate sub-panel approach: clicking a lifestyle focus opens a skill tree view that replaces the panel content. This is the correct use of a sub-panel: the skill tree is a fundamentally different visualization (spatial tree vs. list) that wouldn't work as an expandable section.

---

## 6. Contextual Actions in Overlays

### 6.1 Action Button Placement

**The three viable placement patterns:**

**Pattern A: Fixed Bottom Bar (Recommended Default)**
- Action buttons pinned to the bottom of the panel
- Always visible regardless of scroll position
- Primary action right-aligned, secondary actions left-aligned
- Destructive actions separated by a divider or placed in a "more actions" overflow menu

Best for: Panels with a clear primary action (Confirm, Apply, Build, Assign)
Example: Frostpunk 2's law-passing panel with "Pass Law" button at the bottom

**Pattern B: Inline Actions (Per-Item)**
- Action buttons placed on each item in a list
- Appear on hover or always visible as small icon buttons
- Best for lists where each item has its own set of actions

Best for: Roster management, production chain management, trade route lists
Example: CK3's council panel with "Replace" button on each council position

**Pattern C: Top Action Bar**
- Action buttons in a toolbar at the top of the panel, below the title
- Best for panels that are primarily about acting, not reading
- The toolbar remains visible as content scrolls

Best for: Panels that are tool interfaces rather than information displays
Example: Civ VI's city production queue with "Change Production" prominently at the top

**Placement Rules:**

| Rule | Rationale |
|------|-----------|
| Primary action is visually dominant | Larger, higher contrast, or filled vs. outlined |
| Maximum 1 primary action per panel view | Multiple primaries = no primary (visual competition) |
| Destructive actions are visually distinct | Red text/icon, separated by space, or in overflow menu |
| Actions are contextual to the current view | Tab A shows Tab A's actions, not a global action bar |
| Disabled actions show why they're disabled | Tooltip on disabled button explains what's needed to enable it |

**Anti-Patterns:**
- Action buttons mixed into the information flow (hard to find, easy to accidentally click)
- Actions only available from a right-click context menu with no visual indicator
- All buttons the same visual weight (can't distinguish primary from secondary)
- Actions at the top of a scrolling panel that disappear when you scroll down
- "Apply" buttons without visual indication that changes are pending

### 6.2 Confirmation Patterns for Important Actions

**The Confirmation Spectrum:**

| Action Type | Confirmation Pattern | Example |
|-------------|---------------------|---------|
| Harmless/Reversible | No confirmation | View details, switch tabs, change sorting |
| Moderate/Costly | Inline confirmation | "Are you sure? This costs 200 gold." --- single dialog |
| Significant/Irreversible | Modal confirmation with consequence preview | "Declaring war will: [list of consequences]. Proceed?" |
| Critical/Permanent | Multi-step confirmation with friction | "Type the character's name to confirm execution" (Dwarf Fortress-style) |

**Gold Standard: Crusader Kings III**
CK3 shows consequence previews before confirming major decisions:
- Declaring war: Shows allies that will join, expected war score, potential outcomes
- Executing a prisoner: Shows opinion penalties from all relevant characters
- Disinheriting an heir: Shows succession changes, dynasty opinion impact

The consequence preview IS the confirmation --- it doesn't ask "Are you sure?" in isolation. It asks "Here is what will happen. Do you want this?" This turns the confirmation dialog into an information tool, not just a speed bump.

**Gold Standard: Frostpunk 2 Law Passing**
Frostpunk 2's law-passing panel shows:
- The proposed law and its effects
- Estimated vote breakdown (for/against/undecided)
- Faction reactions and relationship changes
- A "Pass Law" button that is only enabled when the player has enough votes

The entire panel IS the confirmation. The act of reading the panel and deciding to click "Pass" is the confirmation process. No additional "Are you sure?" dialog needed.

**Confirmation Design Rules:**

| Rule | Rationale |
|------|-----------|
| Show consequences, not questions | "This will cost 500 gold and lower opinion by 20" > "Are you sure?" |
| Make consequences scannable | Bullet points or icon + text, not a paragraph |
| Reversible actions need no confirmation | Undo is more elegant than "Are you sure?" for low-stakes actions |
| Don't train players to click past dialogs | If every action asks for confirmation, players stop reading them |
| Destructive confirmation must look different | Red button, warning icon, different panel style than routine confirmations |

**Anti-Patterns:**
- "Are you sure?" with Yes/No and no consequence information
- Confirmation dialogs for routine actions (training players to auto-dismiss)
- No confirmation for genuinely irreversible, consequential actions
- Confirmation text that doesn't explain what will actually happen
- "Are you sure?" followed by another "Are you REALLY sure?" (patronizing, ineffective)

### 6.3 Feedback After Taking an Action Within an Overlay

**The Feedback Trio:** Every action must produce:
1. **Immediate visual acknowledgment** (< 100ms) --- button state change, checkmark animation, highlight flash
2. **State update** (< 500ms) --- the panel content reflects the new state (numbers update, lists change)
3. **World feedback** (contextual) --- if the action affects the game world, the world should reflect it visibly when the player closes the panel

**Feedback Patterns by Action Type:**

| Action Type | Visual Feedback | Audio Feedback | State Feedback |
|------------|-----------------|----------------|----------------|
| Toggle/Selection | Highlight, checkmark | Subtle click | Immediate state change in panel |
| Resource expenditure | Number animates from old to new value | Coin/resource sound | Balance updated, cost greyed out |
| Assignment | Portrait slides into position | Assignment chime | Slot filled, other slots update availability |
| Construction/Queue | Item appears in queue with "new" indicator | Build queue sound | Queue list updates, cost deducted from balance |
| Destructive | Red flash, item removes with animation | Warning/impact sound | Item removed from list, related items update |
| Confirmation (multi-step) | Step indicator advances | Progress chime | Next step appears, previous step marked complete |

**Gold Standard: CK3 Character Interactions**
When the player sends a gift in CK3:
1. Button animates on click (immediate)
2. Gold counter in top bar decreases with a small animation (state)
3. Opinion number on the character updates with a green flash and "+15" floating text (world)
4. A notification toast confirms the action ("Gift sent to Duke Harald") (confirmation)

The player never wonders "did that work?" because feedback is delivered across four channels simultaneously.

**Gold Standard: Stellaris Research Assignment**
When assigning a scientist to a project:
1. Scientist portrait slides into the assignment slot (immediate)
2. Research progress bar appears with estimated completion time (state)
3. The situation log updates to show the active project (world)
4. A brief audio chime confirms the assignment (audio)

**Feedback Rules:**

| Rule | Rationale |
|------|-----------|
| Never leave the player guessing | If nothing visually changed, the player thinks it didn't work |
| Animate state transitions | Numbers that teleport from old to new are harder to parse than animated transitions |
| Use micro-animations, not full animations | 100-300ms transitions, not 1-second cinematic flourishes |
| Color flash for changed values | A brief green/red flash on updated numbers draws the eye to what changed |
| Toast confirmations for non-obvious actions | If the action's effect isn't immediately visible in the panel, add a toast |

**Anti-Patterns:**
- Actions that complete with no visual change (did it work?)
- Feedback that's only auditory (players who play muted miss it entirely)
- State updates that require closing and reopening the panel to see
- Animations that take > 500ms (player waits for feedback instead of continuing)
- Success and failure using the same feedback (can't distinguish outcomes)

---

## 7. Consolidated Scoring Rubric

### 7.1 How to Use This Rubric

Score each area independently on a 1-10 scale using the weighted criteria specified in each section. The final overlay/panel score is an average of the five area scores, weighted as follows:

| Area | Weight | Rationale |
|------|--------|-----------|
| Overlay Architecture | 25% | Foundation: modality, sizing, escape patterns |
| Information Organization | 25% | Core: tabs, sections, data visualization, text density |
| Deep Management Screens | 20% | Depth: domain-specific panel design quality |
| Progressive Disclosure | 15% | Flow: summary > overview > detail transitions |
| Contextual Actions | 15% | Interaction: button placement, confirmation, feedback |

### 7.2 Quick Reference: Score Anchors

**10/10 - Exemplary:**
- CK3-level modality management (non-modal info, modal decisions, appropriate dimming)
- Victoria 3-level information density with 3-tier progressive disclosure
- Old World-level event panel narrative quality with consequence previews
- Stellaris-level centralized management with pinnable monitoring widgets
- Every action produces immediate, visible, multi-channel feedback
- All six expert principles demonstrably applied to every panel

**7-8/10 - Strong:**
- Clear modal/non-modal distinction with appropriate dimming levels
- Tabbed panels with good section organization and collapsible groups
- Deep management screens with drill-down capability and breadcrumb navigation
- Actions produce clear feedback; confirmations show consequences
- Most expert principles applied, minor violations

**5-6/10 - Adequate:**
- Overlays function but modality is inconsistent
- Panels have tabs but internal organization is flat
- Management screens show data but lack drill-down or trend visualization
- Actions produce feedback but confirmations are generic ("Are you sure?")
- Some expert principles applied, some clear violations

**3-4/10 - Below Standard:**
- Modal/non-modal usage is inconsistent or wrong for context
- Panels are dense text blocks with no visual hierarchy
- Management screens are shallow or scattered across multiple locations
- Actions produce minimal feedback; no consequence previews
- Few expert principles applied

**1-2/10 - Failing:**
- Overlays trap the player or can't be closed reliably
- Information is unstructured and overwhelming
- Management tasks require navigating multiple disconnected screens
- Actions have no feedback or confirmation
- Expert principles violated throughout

### 7.3 Cross-Cutting Quality Gates

These must ALL pass for a score above 7 in any overlay/panel area:

| Gate | Requirement |
|------|-------------|
| **Tufte Gate** | Every data display uses the appropriate visualization type. Color carries semantic meaning. Small multiples used for comparable entities. |
| **Norman Gate** | Panel structure mirrors system structure. Cause-and-effect is visually traceable. Every interactive element has visible affordance. |
| **Krug Gate** | Panel purpose identifiable within 5 seconds. Navigation is convention-following (tabs top, close top-right, actions bottom). No "thinking required" for standard operations. |
| **Chen Gate** | Panel transitions < 200ms. Escape always works. Game world remains partially visible for non-fullscreen panels. Audio continues during overlays. |
| **Nielsen Gate** | All 10 heuristics score >= 4/5 for each panel type. Recognition over recall: options shown, not memorized. Error prevention for destructive actions. |
| **Wroblewski Gate** | >= 3 information tiers exist (glance > overview > detail). Each tier accessible in exactly 1 click from the previous tier. |

### 7.4 Per-Panel-Type Checklists

**Character/Entity Detail Panel:**
- [ ] Portrait + name + primary role prominently displayed
- [ ] 3-7 tabs covering distinct information domains
- [ ] Stats shown with modifiers accessible via hover/expand
- [ ] Relationship indicators visible without drilling down
- [ ] Action buttons at panel bottom, primary action visually dominant
- [ ] Navigation to related entities via clickable links
- [ ] Back button or breadcrumb for return navigation
- [ ] Close via Escape, X button, right-click, and hotkey toggle

**List/Roster Management Panel:**
- [ ] Filterable by >= 3 criteria
- [ ] Sortable by >= 3 attributes
- [ ] Card/row items show 3-5 key data points without drilling down
- [ ] Inline actions available per-item (assign, dismiss, inspect)
- [ ] Drill-down opens detail without losing list context (split view or side panel)
- [ ] Aggregate summary shown at top or bottom (total count, average stats)
- [ ] Empty states handled gracefully ("No characters match these filters")

**Dashboard/Statistics Panel:**
- [ ] Aggregate health indicator visible at the top (overall status)
- [ ] Small multiples or comparative bars for individual items
- [ ] Color coding consistent with game-wide conventions
- [ ] Historical data accessible via drill-down (sparklines or full charts)
- [ ] Problem items automatically highlighted or sorted to the top
- [ ] Export or comparison features for advanced players
- [ ] Filters and grouping options for different analytical views

**Event/Decision Panel:**
- [ ] Narrative text is 60-150 words, readable in 15-30 seconds
- [ ] Character portraits and relationship context shown
- [ ] 2-4 choice buttons with clear mechanical previews on each
- [ ] Consequence preview shown before confirmation
- [ ] Backdrop dimming proportional to decision weight
- [ ] Audio continues; ambient sound may shift to match event mood
- [ ] Choice history accessible after the fact (event log)

**Configuration/Creation Wizard Panel:**
- [ ] Step indicator showing current position in the process (step 2 of 5)
- [ ] Live cost/requirement update as options are selected
- [ ] Summary sidebar or section showing current configuration
- [ ] Back navigation to previous steps without losing selections
- [ ] Validation errors shown inline, not in separate dialogs
- [ ] Final confirmation shows complete preview of what will be created
- [ ] Cancel exits the wizard entirely with a single confirmation

---

## Appendix A: Game-Specific Overlay Patterns

### Crusader Kings III (Paradox Development Studio)
- **Key Pattern:** Non-modal side panels for information; modal centered panels for decisions
- **Strength:** Character sheet with multi-tab organization is the genre benchmark
- **Strength:** Right-click interaction menus with context-sensitive options
- **Strength:** Tooltip chains that build from simple to complex (hover > extended hover > click)
- **Weakness:** Tooltip depth can become "tooltips within tooltips within tooltips" for complex modifiers
- **Use for:** Character panel architecture, modal/non-modal balance, escape/close patterns

### Victoria 3 (Paradox Development Studio)
- **Key Pattern:** Right-side pinned situation panels for ongoing processes; left-side panels for management
- **Strength:** Market screen is the deepest economic panel in the genre
- **Strength:** Political lens integrates economic data with political consequences
- **Strength:** Pinned situations as non-blocking monitoring widgets
- **Weakness:** Information density can overwhelm new players; onboarding is steep
- **Use for:** Economic dashboard design, situation monitoring, political consequence displays

### Stellaris (Paradox Development Studio)
- **Key Pattern:** Outliner as persistent monitoring widget; empire interface as tabbed management hub
- **Strength:** Situation Log centralizes active projects, events, and archaeology
- **Strength:** Species panel with expandable sections for traits and rights
- **Strength:** Tab reordering in empire interface for player customization
- **Weakness:** Espionage UI scattered and unsatisfying (developer-acknowledged)
- **Use for:** Centralized log patterns, outliner design, tab customization

### Old World (Mohawk Games)
- **Key Pattern:** Rich narrative event panels with illustrated character cards
- **Strength:** Events are the game's primary storytelling mechanism; UI reflects this with high production value
- **Strength:** Character cards combine portrait, stats, traits, and relationships in a compact visual
- **Strength:** Undo button on game actions (exceptional accessibility)
- **Weakness:** Information density in late game can overwhelm
- **Use for:** Event panel design, character card templates, narrative overlay quality

### Frostpunk 2 (11 bit studios)
- **Key Pattern:** District overlays with heat mapping; faction management through Council UI
- **Strength:** Heatmap overlay uses spatial color coding for immediate comprehension
- **Strength:** Law system UI shows vote breakdown and faction stances before committing
- **Strength:** Command Radial hides HUD and focuses attention during navigation
- **Weakness:** Initial release had readability issues that required patching
- **Use for:** Spatial overlay design, law/voting UI, radial navigation for management modes

### Anno 1800 (Ubisoft Blue Byte)
- **Key Pattern:** Contextual panels from building selection; dedicated statistics menu
- **Strength:** Production statistics with supply/demand bars are the benchmark for economic visualization
- **Strength:** Per-tier population needs create natural progressive disclosure
- **Strength:** Statistics menu with 5 organized tabs covering all economic dimensions
- **Weakness:** Trade route UI complexity remains a pain point even with iteration
- **Use for:** Production chain visualization, economic statistics panels, per-tier information disclosure

### Civilization VI (Firaxis Games)
- **Key Pattern:** Full-screen tech tree; city management overlay; diplomatic overlays
- **Strength:** Tech tree as spatial visualization with era-based organization
- **Strength:** City management overlay combines building placement with resource display
- **Strength:** Diplomatic deal screen with clear offer/counteroffer layout
- **Weakness:** Late-game diplomatic panels can become cluttered with many civilizations
- **Use for:** Tech tree visualization, city overlay patterns, diplomatic deal UI

### Dwarf Fortress (Bay 12 Games / Kitfox Games)
- **Key Pattern:** Deep nested menus with extreme information density
- **Strength (Steam):** Mouse-driven navigation, text filters, tabbed panels
- **Strength:** Unparalleled depth of information access for every game entity
- **Weakness:** Information hierarchy remains challenging despite Steam UI overhaul
- **Weakness:** New player onboarding is the genre's weakest
- **Use for:** Maximum information density reference, accessibility transformation case study

### Total War: Three Kingdoms (Creative Assembly)
- **Key Pattern:** Character cards with emotional portraits; faction diplomacy with attitude masks
- **Strength:** Diplomacy panel uses color-coded masks (green to red) for at-a-glance attitude reading
- **Strength:** Court panel groups characters by role with satisfaction indicators
- **Strength:** Army composition panel with retinue-per-general structure
- **Weakness:** Character panel navigation can feel disconnected from the campaign map
- **Use for:** Character card design, diplomacy attitude visualization, army composition UI

### Oxygen Not Included (Klei Entertainment)
- **Key Pattern:** Toggle-based overlay system for colony analysis
- **Strength:** Overlays transform the entire game view to show one specific system (temperature, gas, disease)
- **Strength:** Overlays unlock through research, creating natural progressive disclosure
- **Strength:** Resource tracking sidebar with expandable categories
- **Weakness:** Overlay count can become overwhelming; no "composite" overlay option
- **Use for:** Toggle-overlay system, research-gated UI disclosure, environmental data visualization

---

## Appendix B: Key Takeaways for Implementation

1. **Default to non-modal.** Only use modal overlays when the decision is irreversible or high-consequence. Information panels must never block game interaction.

2. **Three tiers, always.** Glance (HUD/tooltip) > Overview (management panel) > Deep Dive (drill-down). Each tier accessible in exactly 1 click from the previous. Never dump all data on the first view.

3. **Panels must mirror system structure.** If the game system has a hierarchy (network > agents > operations), the panel must have the same hierarchy. If the structure mismatches, players build wrong mental models.

4. **Backdrop dimming communicates decision weight.** Light dim = optional. Medium dim = important. Heavy dim = irreversible. Train the player to calibrate attention from the dimming alone.

5. **Tabs for categories, collapsible sections for depth.** Use tabs when content is fundamentally different views of the same entity. Use collapsible sections when content is different levels of detail for the same view.

6. **Show consequences, not questions.** "This will cost 500 gold and lower opinion by 20" is a confirmation. "Are you sure?" is a waste of the player's time.

7. **Every action needs the Feedback Trio.** Immediate visual acknowledgment (< 100ms), state update (< 500ms), and world reflection (when panel closes). If even one is missing, the player wonders "did that work?"

8. **Escape is sacred.** Escape always closes the topmost panel. Always. Cascade from innermost to outermost. Never capture Escape for anything else while a panel is open.

9. **Centralize management.** Espionage, archaeology, trade routes --- whatever the system, there must be ONE screen that shows all active instances. Scattering across per-entity panels is the #1 deep management anti-pattern.

10. **Design panels as small multiples.** When displaying lists of entities (characters, goods, provinces), use a repeating card/row template where only the data varies. The player's brain locks onto the pattern and scans for anomalies --- this is how experts process information.

---

## Sources

- [CK3 UI, Petter Lundh (ArtStation)](https://www.artstation.com/artwork/WmKNaN)
- [CK3 Console Dev Diary #3: UI/UX and Controls (Paradox Interactive)](https://www.paradoxinteractive.com/games/crusader-kings-iii/news/ck3-console-dev-diary-3-uiux-and-controls)
- [CK3 Interface Guide (gamepressure.com)](https://www.gamepressure.com/crusader-kings-3/interface-description/z2f0f6)
- [Victoria 3 Dev Diary #30: User Interface Overview (Paradox Forums)](https://forum.paradoxplaza.com/forum/threads/victoria-3-dev-diary-30-user-interface-overview.1507166/)
- [Victoria 3 User Interface Wiki](https://vic3.paradoxwikis.com/User_interface)
- [Victoria 3 UI Art, Kenneth Lim (ArtStation)](https://www.artstation.com/artwork/3Eq0ZJ)
- [Stellaris Empire Interface Wiki](https://stellaris.paradoxwikis.com/index.php?title=Empire_interface)
- [Stellaris Main Interface (Fandom)](https://stellaris.fandom.com/wiki/Main_interface)
- [Stellaris Intelligence/Espionage Wiki](https://stellaris.paradoxwikis.com/Intelligence)
- [Stellaris Espionage Rework (PCGamesN)](https://www.pcgamesn.com/stellaris/espionage-rework-survey)
- [Stellaris Archaeological Sites Wiki](https://stellaris.paradoxwikis.com/Archaeological_site)
- [Old World (Mohawk Games Official)](https://mohawkgames.com/oldworld/)
- [Old World Designer Notes (Soren Johnson)](http://www.designer-notes.com/category/mohawk/)
- [Frostpunk 2 UI Adaptation (Xbox Wire)](https://news.xbox.com/en-us/2025/09/18/adapting-frostpunk-2s-depth-to-a-gamepad/)
- [Frostpunk 2 Game UI Database](https://www.gameuidatabase.com/gameData.php?id=1965)
- [Anno 1800 Production Statistics (Fandom)](https://anno1800.fandom.com/wiki/Statistics)
- [Anno 1800 DevBlog: User Interface (Anno Union)](https://www.anno-union.com/devblog-user-interface-2/)
- [Anno 1800 Game UI Database](https://www.gameuidatabase.com/gameData.php?id=1118)
- [Civilization VI Interface (gamepressure.com)](https://www.gamepressure.com/sidmeierscivilization6/interface/ze92ba)
- [CQUI Community Edition for Civ VI (GitHub)](https://github.com/Azurency/CQUI_Community-Edition)
- [Dwarf Fortress New UI (PC Gamer)](https://www.pcgamer.com/the-new-dwarf-fortress-ui-looks-so-much-better/)
- [Dwarf Fortress Interface Wiki](https://dwarffortresswiki.org/index.php/Interface)
- [Total War: Three Kingdoms Manual (Feral Interactive)](https://www.feralinteractive.com/en/manuals/threekingdomstw/1.0/steam/)
- [Total War: Three Kingdoms Diplomacy (Fandom)](https://totalwar.fandom.com/wiki/Diplomacy_(Total_War:_Three_Kingdoms))
- [Oxygen Not Included Overlays Wiki](https://oxygennotincluded.fandom.com/wiki/Overlays)
- [CK3 Faith Wiki](https://ck3.paradoxwikis.com/Faith)
- [Edward Tufte's Principles (thedoublethink.com)](https://thedoublethink.com/tuftes-principles-for-visualizing-quantitative-information/)
- [Tufte's Design Principles Applied to Games (Wolfire Games)](https://www.wolfire.com/blog/2009/02/design-principles-from-tufte/)
- [Mastering Tufte's Data Visualization Principles (GeeksforGeeks)](https://www.geeksforgeeks.org/data-visualization/mastering-tuftes-data-visualization-principles/)
- [Don Norman's Principles of Interaction Design (Medium)](https://medium.com/@sachinrekhi/don-normans-principles-of-interaction-design-51025a2c0f33)
- [Steve Krug's Don't Make Me Think (Sensible.com)](https://sensible.com/dont-make-me-think/)
- [Steve Krug's Principles (Medium)](https://medium.com/@yashu02raghuwanshi/6-guiding-principles-from-the-dont-make-me-think-by-steve-krug-8dde3797abe6)
- [Jenova Chen: Flow in Games MFA Thesis](https://www.jenovachen.com/flowingames/Flow_in_games_final.pdf)
- [Jenova Chen: Design Flow](http://jenovachen.info/design-flow)
- [Jakob Nielsen: Progressive Disclosure (NN/g)](https://www.nngroup.com/articles/progressive-disclosure/)
- [Luke Wroblewski: Mobile First](https://www.lukew.com/resources/mobile_first.asp)
- [Modal vs Non-Modal in UI Design (Medium)](https://medium.com/design-bootcamp/ux-blueprint-09-modal-and-non-modal-components-in-ui-design-why-they-matter-75e6ffb62946)
- [Best Practices for Modals in Video Games (indieklem)](https://indieklem.substack.com/p/5-best-practices-for-modals-in-video)
- [10 Guidelines for Overlays/Modals (UXforthemasses)](https://www.uxforthemasses.com/overlays/)
- [Nested Tab UI Design (Design Monks)](https://www.designmonks.co/blog/nested-tab-ui)
- [Game UI Database](https://gameuidatabase.com/)
- [Strategy Game Battle UI Analysis (Medium)](https://medium.com/@treeform/strategy-game-battle-ui-3b313ffd3769)
- [Game UI Design Principles (JustInMind)](https://www.justinmind.com/ui-design/game)
- [Confirmation Dialogs Design (UX Planet)](https://uxplanet.org/confirmation-dialogs-how-to-design-dialogues-without-irritation-7b4cf2599956)
