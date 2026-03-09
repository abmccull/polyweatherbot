# Core Gameplay Interaction, Setup, Save/Load & Meta UI: Design Standards Report

## Best-in-Class UI/UX Research for Strategy, Management & Ritual Games

---

## Table of Contents

1. [Core Ritual / Consultation Interaction](#1-core-ritual--consultation-interaction)
2. [Run Setup / New Game Configuration](#2-run-setup--new-game-configuration)
3. [Save / Load / Settings](#3-save--load--settings)
4. [End-of-Run / Legacy Screens](#4-end-of-run--legacy-screens)
5. [Tutorial / Onboarding](#5-tutorial--onboarding)
6. [Design Expert Principles Applied](#6-design-expert-principles-applied)
7. [10/10 Criteria Checklist](#7-1010-criteria-checklist)

---

## 1. Core Ritual / Consultation Interaction

The consultation is The Oracle's signature mechanic -- the equivalent of Slay the Spire's card combat, Hades' chamber encounters, or Inscryption's card matches. This is where the game lives or dies aesthetically and mechanically.

### 1.1 How the Best Games Design Their "Main Action" Screen

#### Gold Standard: Inscryption's Diegetic Table

Inscryption eliminates the boundary between "game UI" and "game world" entirely. The card table IS the interface. A bell ends your turn. Candle flames are your lives. A balance scale shows who is winning. A rulebook explains mechanics. Nothing is an overlay -- everything is a physical object you interact with on the table surface.

**Why it works:**
- Every UI element has a narrative justification (diegetic design)
- Interactions feel tactile -- you ring a bell, flip cards, manipulate physical objects
- The atmosphere is inseparable from the mechanics
- Cognitive load is reduced because spatial memory replaces menu navigation

#### Gold Standard: Cultist Simulator's Verb-Card Workspace

Cultist Simulator uses a flat workspace where cards (representing stats, followers, lore, tools) are dragged into "verb" slots (representing actions like Study, Work, Dream, Talk, Explore). The outputs emerge as new cards. The entire game is played on this single surface.

**Why it works:**
- The combinatorial discovery of dragging cards into verbs creates genuine mystery
- No explicit instructions for most combinations -- the player must experiment
- Timers on verb slots create pacing tension without artificial turn structures
- The flat workspace means everything is visible simultaneously -- no hidden menus

#### Gold Standard: Slay the Spire's Three-Card Reward

After each combat, Slay the Spire presents exactly three card choices plus a "skip" option. This is the game's most important decision point.

**Why it works:**
- Constraint breeds clarity: three options plus skip is cognitively manageable
- Each card shows its full text, energy cost, and type at a glance
- Hovering a card enlarges it and dims the others -- clear focus hierarchy
- The "skip" button is always visible, legitimizing restraint as a strategy
- No time pressure on the decision

#### Anti-Pattern: Hidden Information at Decision Time

FTL occasionally presents event choices where your ship's actual capabilities (crew skills, augmentations) are not visually reflected in the UI. Players with a battle-hardened crew may still lose crewmembers to a random event because the UI doesn't surface how your stats affect outcomes. This creates a disconnect between player investment and perceived fairness.

#### Anti-Pattern: Information Overload at Choice Point

Crusader Kings III event popups sometimes present 4-6 options with complex stat dependencies, trait requirements, and hidden outcome weights all competing for attention. When the player cannot quickly assess what each choice means, the mystical weight of the decision collapses into spreadsheet anxiety.

### 1.2 Mystical / Ritual Atmosphere in UI

#### Principle: Diegetic Over Overlay

The UI should feel like part of the sacred space, not a software toolbar floating above it. For The Oracle, this means the consultation screen should evoke a physical ritual workspace -- a stone altar, an offering arrangement, a divination layout.

**Reference implementations:**

| Game | Diegetic Element | Effect |
|------|-----------------|--------|
| Inscryption | Physical card table, bell, candles, scale | Total immersion in the ritual |
| Strange Horticulture | Desktop with drawers, map, plant book, labels | Tactile puzzle-solving atmosphere |
| Cultist Simulator | Flat desk surface with cards and verb slots | Occult experimentation feeling |
| Return of the Obra Dinn | Logbook with handwritten vs. printed text | Investigation-as-artifact |

**Application to The Oracle:**

The consultation overlay should feel like a ritual preparation surface:
- Word tiles should look like physical tokens (bone, stone, bronze, or fired clay) arranged on a divination surface
- The omen report should appear as something written or etched, not as a pop-up modal
- The Pythia's state (fatigue, clarity) should be represented through visual changes to the ritual space itself -- dimmer candles, more smoke, trembling hands -- not health bars
- Sound design should underscore each tile placement, each prophecy delivery, each consequence arrival

#### Principle: Restraint Creates Reverence (Fumito Ueda)

Ueda's "design by subtraction" philosophy applies directly: remove every UI element that does not serve the ritual's emotional arc. No minimap during consultations. No resource bar. No faction sidebar. The consultation screen should feel like entering a sacred space where worldly concerns recede.

**Concrete guidance:**
- Dim or hide the main HUD when the consultation overlay opens
- Use transition animations (incense smoke, light shift, sound fade) to mark the boundary between management and ritual
- Keep the consultation surface sparse -- fewer elements with more visual weight
- Let empty space on the surface carry meaning (an unused tile slot is a deliberate silence)

### 1.3 Choice Presentation: Tile Selection & Prophecy Crafting

#### Gold Standard: Constrained Choice Sets with Full Visibility

Slay the Spire's three-card reward and Hades' boon selection both present a small number of options (2-4) with complete information visible. The player never needs to scroll or navigate to compare.

**Applied to tile selection:**
- Present the available word tile pool in a single visible layout (no scrolling if possible)
- Each tile should show its visible literary text prominently; hidden semantic tags remain hidden
- The active prophecy composition area should show all currently placed tiles simultaneously
- Available tile slots in the prophecy should be visually distinct empty receptacles, not invisible drop targets
- Allow tiles to be placed AND removed freely before committing -- reversibility prevents anxiety (Nielsen Heuristic #3: User Control and Freedom)

#### Gold Standard: Preview Before Commit

Into the Breach shows enemy attack intentions before your turn. Civilization VI shows tech tree branches before you commit to research. The pattern: let the player see consequences before they lock in.

**Applied to prophecy crafting:**
- Show live scoring guidance as tiles are placed: risk/value shifts, clarity indicators, ambiguity warnings
- Before final delivery, preview the probable consequence space ("This prophecy risks angering Sparta but may elevate Athens' trust")
- The commit action (delivering the prophecy) should feel weighty -- a distinct animation, sound, and moment of irreversibility
- Never auto-commit a prophecy. The player should always take the final deliberate action.

#### Anti-Pattern: Blind Commitment

Cultist Simulator's verb slots sometimes consume cards without adequate preview of what will happen. For a game about experimentation, this is intentional. For The Oracle, where the player is a skilled professional (a priestess running a sanctuary), blind commitment would feel punishing rather than mysterious.

### 1.4 Feedback Loops: How Does the Player Know They Did Well or Poorly?

#### Gold Standard: Multi-Layered Delayed Feedback (Hades + Slay the Spire)

The best feedback systems operate on multiple timescales:

| Timescale | Slay the Spire | Hades | Applied to The Oracle |
|-----------|---------------|-------|----------------------|
| Immediate | Damage numbers, block applied, energy spent | Hit sounds, visual effects, HP change | Tile placement sound, scoring indicator shift, Pythia reaction |
| Short-term | End-of-combat rewards, health remaining | Boon granted, room cleared, dialogue line | Prophecy delivery confirmation, petitioner reaction text, immediate offering received |
| Medium-term | Map path decisions, deck quality evolution | Mirror upgrade progress, relationship deepening | Faction disposition shift, consequence arrival (days later), sanctuary reputation change |
| Long-term | Ascension level unlocked, heart key progress | Epilogue progress, weapon aspect unlocks | Era transition, political legacy, Pythia legend status |

**Concrete guidance:**
- Immediate: When a tile is placed, the ritual surface should subtly react (light shift, smoke curl, score indicator movement). When a tile is removed, the surface settles back. This gives the player continuous micro-feedback during crafting.
- Short-term: Prophecy delivery should show the petitioner's immediate reaction and any instant rewards/costs. This is the "combat result" moment.
- Medium-term: Consequence arrivals (days or weeks later) should surface with narrative weight -- a messenger arriving, a faction leader reacting, a political shift manifesting. These should feel like dramatic reveals, not log entries.
- Long-term: The chronicle should accumulate a visible history that the player can review, creating a sense of legacy.

#### Anti-Pattern: Ambiguous Outcomes

If the player crafts a prophecy and receives only a generic "The petitioner departs" with no indication of quality, the feedback loop is broken. Even in a game about ambiguity, the player's skill expression needs acknowledgment. The game's ambiguity should be in the prophecy's future consequences, not in whether the player did their job well.

### 1.5 Pacing: Building Tension Through UI Design

#### Principle: Jenova Chen's Flow Channel

Chen's flow theory states that engagement requires a dynamic balance between challenge and skill. The UI's pacing mechanisms should modulate this balance.

**Consultation pacing model:**

1. **Approach** (low tension): The petitioner arrives. Context is presented. The player has time to read and understand.
2. **Preparation** (rising tension): The tile pool is revealed. The player surveys options. The Pythia's state is visible.
3. **Crafting** (peak engagement): Tile placement, real-time scoring feedback, iterative refinement. This is the flow state zone.
4. **Commitment** (tension spike): The delivery button. A moment of irreversibility. Brief held pause.
5. **Resolution** (release): The prophecy is spoken. The petitioner reacts. Immediate outcomes land.
6. **Aftermath** (settling): Return to sanctuary management. The world continues. Consequences are pending but not yet visible.

**UI mechanisms for pacing:**
- Approach: Slide-in animation for the petitioner panel. Ambient sound shifts.
- Preparation: Tile pool appears with a brief reveal animation (tiles dealt, not teleported). Music layer adds a subtle underscore.
- Crafting: No artificial timer. Pacing comes from the player's own deliberation. The UI should be responsive and snappy -- no animation delays on tile placement/removal.
- Commitment: The delivery button should require a deliberate action (not just a click -- perhaps a click-and-hold, or a drag-to-deliver gesture). A brief cinematic moment: Pythia speaks, smoke billows, the room shifts.
- Resolution: Petitioner reaction text appears with appropriate pacing (not all at once). Rewards/costs appear sequentially.
- Aftermath: Overlay fades. Main HUD returns. The player is back in the sanctuary. A brief event-feed entry marks the consultation's completion.

#### Reference: Inscryption's Turn-End Bell

Inscryption forces the player to physically ring a bell to end their turn. This tiny ritual action (click the bell, hear the ring, watch the opponent respond) transforms a mundane "end turn" button into a pacing device. The bell is the commitment point. For The Oracle, the equivalent might be lighting a brazier, pouring a libation, or striking a ritual gong.

---

## 2. Run Setup / New Game Configuration

### 2.1 Seed Selection Patterns

#### Gold Standard: Into the Breach's Timeline Reset

Into the Breach frames restarting as a narrative act: your pilots travel back through a timeline breach to try again. The seed is implicit in the timeline selection. Daily/weekly challenge seeds provide community-shared experiences.

**Applied to The Oracle:**
- Seeds should be presented with narrative framing: "The threads of fate align differently..." or a new reading of the stars
- Optional manual seed entry for competitive/shared play
- Daily oracle challenge seeds for community engagement
- Seed should be visible in the save file and end-of-run summary for sharing

#### Gold Standard: Slay the Spire's Custom Seed Entry

Slay the Spire allows manual seed entry alongside random generation. The seed field is unobtrusive but accessible -- it does not clutter the primary flow.

**Applied guidance:**
- Default flow: random seed, no friction
- Advanced flow: seed entry field visible but secondary (collapsed or in an "advanced" section)
- Seed should be copy-pasteable for sharing

### 2.2 Difficulty / Modifier Selection

#### Gold Standard: Hades' Pact of Punishment

Hades presents difficulty modifiers (the Pact of Punishment) as individual toggleable conditions, each with their own heat value. Players can mix and match modifiers to create custom difficulty profiles. Each modifier has a clear name, icon, and numerical impact.

**Why it works:**
- Granular control: players choose which aspects become harder
- Visible reward scaling: higher heat = better rewards
- Progressive unlock: new modifiers appear as the player demonstrates mastery
- Each modifier is described in one sentence with its numerical effect

#### Gold Standard: Slay the Spire's Ascension Levels

Ascension levels are sequential difficulty tiers that stack cumulative modifiers. Each level adds one new constraint. The player must clear Ascension N before accessing Ascension N+1.

**Why it works:**
- Simplicity: one number captures difficulty
- Cumulative complexity: veterans carry the full mental model of all stacked modifiers
- Achievement tracking: each ascension level is a badge of mastery

#### Applied to The Oracle:

Difficulty modifiers should be themed as "divine burdens" or "trials of the oracle":
- Each modifier should have a narrative name ("The Jealous Gods," "Age of Skepticism," "The Pythia's Curse")
- Modifiers should be individually toggleable (Hades model) rather than strictly sequential
- A combined difficulty score should be visible for run comparison
- Modifier descriptions should be one sentence with clear mechanical impact
- A "recommended first run" preset should exist for new players, with no modifiers active

### 2.3 Origin / Faction / Scenario Selection with Preview

#### Gold Standard: Civilization VI's Leader Selection

Civilization VI shows each leader with:
- A large character portrait with personality animation
- Unique ability name and one-paragraph description
- Starting bonuses listed as icons with tooltips
- A difficulty rating for new players

**Why it works:**
- Visual identity: each leader is immediately recognizable
- Scannable: you can compare leaders by scrolling horizontally
- Depth on demand: hover for details, don't front-load all text

#### Gold Standard: FTL's Ship Selection

FTL shows each unlocked ship with:
- A visual ship layout preview
- Starting crew composition
- Starting equipment loadout
- A brief flavor text paragraph
- Locked ships shown as silhouettes with unlock conditions

**Why it works:**
- The ship preview IS the decision-relevant information (layout = strategy)
- Locked silhouettes create aspiration without spoiling content
- Compact presentation: everything fits on one screen

#### Applied to The Oracle:

Scenario selection should show:
- A visual preview of the starting precinct layout or terrain
- The era and starting political landscape summary
- Key starting resources and constraints
- A one-paragraph narrative premise ("The Oracle of Delphi stands at the crossroads of a brewing conflict between Athens and Sparta...")
- Difficulty indicator for new players
- Locked scenarios shown with shadowed previews and unlock conditions

### 2.4 "Start" Button Prominence and Ceremony

#### Principle: The Start Button Is a Commitment Ritual

The "Start Game" button in a roguelike or strategy game is the player's equivalent of crossing a threshold. It should feel deliberate and ceremonial, not like clicking "OK" on a dialog box.

**Gold Standard: Slay the Spire's "Embark" Button**

Slay the Spire uses the word "Embark" (not "Start" or "Play"), placed prominently at the bottom center of the setup screen. The word choice implies journey and adventure.

**Applied to The Oracle:**
- Use thematic language: "Begin the Reading," "Open the Sanctuary," "Receive the First Petitioner"
- The button should be large, centered, and visually distinct from all other controls
- Pressing it should trigger a brief transition sequence (not instant scene swap): a door opening, a flame lighting, a Pythia taking her seat
- The transition should establish the sanctuary's initial state and mood before handing control to the player
- Do NOT gate the start button behind mandatory configuration. Default settings should always produce a valid game.

#### Anti-Pattern: Overwhelming Setup

Civilization VI's game setup screen is powerful but can overwhelm new players with map size, game speed, leader selection, advanced settings, and mod toggles all competing for attention. For The Oracle, the default path should require zero configuration decisions -- just "Begin the Reading."

#### Anti-Pattern: Instant Start

Dropping the player directly into gameplay with no transition feels jarring. Even a 2-3 second atmospheric transition (Hades' opening run sequence, Slay the Spire's map reveal) establishes context and mood.

---

## 3. Save / Load / Settings

### 3.1 Where Do Save/Load Belong?

#### Gold Standard: Layered Access with Pause Menu as Primary Home

The consensus across well-designed games:

| Access Point | Priority | Example |
|-------------|----------|---------|
| Pause menu | Primary | Hades, Slay the Spire, Into the Breach |
| Keyboard shortcut | Power user | F5/F9 (quicksave/load), Ctrl+S |
| Main menu | Secondary | All games -- for pre-session load |
| Auto-save | Background | Every well-designed modern game |

**Applied to The Oracle:**
- Pause menu (Esc) should contain Save, Load, Settings, Return to Main Menu, and Quit
- F5 for quicksave, F9 for quickload (industry-standard shortcuts)
- Main menu should show Continue (most recent save), New Game, Load Game, Settings, Quit
- Auto-save should occur at natural breakpoints: after each consultation delivery, at month boundaries, before major political events

#### Anti-Pattern: Save Available Only from Main Menu

Forcing the player to exit to the main menu to save breaks immersion and punishes players who need to stop quickly. Save should always be accessible from within gameplay.

#### Anti-Pattern: Save as a Gameplay Mechanic (Unless Intentional)

Some roguelikes intentionally restrict saving (permadeath, save-on-quit only). If The Oracle uses this pattern, it should be clearly communicated during setup, not discovered when the player tries to save mid-session.

### 3.2 Auto-Save Visibility

#### Gold Standard: Unobtrusive but Visible Auto-Save Indicator

The best pattern is a small, transient icon that appears briefly when an auto-save occurs:

- A spinning/pulsing save icon in a screen corner (common: lower-right)
- Appears for 1-2 seconds, then fades
- Does NOT interrupt gameplay or require acknowledgment
- The icon should match the game's visual language (for The Oracle: a small bronze seal, a quill mark, or a wax stamp)

**Additional guidance:**
- Auto-save slots should be separate from manual save slots to prevent accidental overwriting
- The number of auto-save slots should be configurable (default: 3 rotating)
- The player should be able to promote an auto-save to a named manual save

#### Anti-Pattern: No Auto-Save Indicator

If the player doesn't know when auto-saves occur, they don't know whether their progress is safe. This creates background anxiety.

#### Anti-Pattern: Full-Screen Save Confirmation

Pausing the game to show "Game Saved!" in a modal dialog is disruptive and unnecessary for auto-saves.

### 3.3 Settings Organization

#### Gold Standard: Categorized Tabs with Immediate Preview

Settings should be organized into clear categories:

| Category | Contents |
|----------|----------|
| **Gameplay** | Difficulty, auto-pause on events, consultation timer toggle, tutorial hints toggle |
| **Display** | Resolution, fullscreen/windowed, UI scale, brightness, colorblind modes |
| **Audio** | Master volume, music volume, SFX volume, ambient volume, voice volume (each with independent sliders) |
| **Controls** | Key bindings (rebindable), mouse sensitivity, scroll speed |
| **Accessibility** | Text size, high contrast mode, screen reader support, reduced motion |

**Applied guidance:**
- Use tabs or a vertical sidebar for categories, not a single scrolling list
- Changes should preview in real-time where possible (audio sliders play sample sounds, display changes apply immediately)
- Provide a "Reset to Defaults" button per category and globally
- Settings should persist across sessions (save to local storage / config file)
- The settings screen should be accessible from BOTH the main menu AND the pause menu, with identical contents

#### Anti-Pattern: Settings in a Single Scrolling List

Mixing audio, display, gameplay, and control settings in one long scroll creates a scavenger hunt. Category tabs solve this.

#### Anti-Pattern: Settings That Require Restart

Modern games should apply settings changes immediately wherever technically possible. If a restart is truly required (rare -- typically only for rendering backend changes), warn the player BEFORE they change the setting.

### 3.4 Quick Save Patterns

#### Gold Standard: Single-Slot Quick Save with Overwrite Warning

- F5 saves to a dedicated quick-save slot
- F9 loads from that slot
- If a quick-save already exists, F5 overwrites it silently (no confirmation dialog -- speed is the point)
- The quick-save slot is visually distinct from manual saves in the Load screen
- Quick-save should capture the full game state identically to a manual save

**Additional patterns worth supporting:**
- Save-on-quit for roguelike-style sessions (auto-saves on exit, deletes save on load)
- Cloud sync for cross-device play (if web + desktop builds share persistence)
- Save file export/import for sharing sanctuary states

---

## 4. End-of-Run / Legacy Screens

### 4.1 Score Presentation

#### Gold Standard: Slay the Spire's Score Breakdown

Slay the Spire's end-of-run screen shows:
- Win or loss clearly stated at top
- Final score as a large number
- Score breakdown by category (floors cleared, elites killed, bosses killed, bonuses)
- Each category animates in sequentially, building to the final total
- The score contributes to cumulative unlock progress

**Why it works:**
- Sequential reveal creates a mini-narrative of the run's accomplishments
- Each category reminds the player of moments from the run
- The cumulative unlock progress gives every run meaning, even losses

#### Gold Standard: Hades' Run Summary

Hades shows:
- Time elapsed
- Chambers cleared
- Darkness (currency) earned
- A highlight of boons and upgrades collected
- Narrative dialogue that advances the story regardless of outcome

**Why it works:**
- The narrative dialogue after death transforms failure into story progress
- The Darkness earned means every run has tangible legacy value
- The summary is brief -- it doesn't overstay its welcome

#### Applied to The Oracle:

End-of-scenario scoring should present:
- **Headline**: "The Oracle's Legacy" or scenario-specific title
- **Primary metric**: Sanctuary reputation / Oracle renown (large, central)
- **Breakdown categories** (revealed sequentially):
  - Prophecies delivered (count and average quality)
  - Faction relationships at conclusion
  - Consequences fulfilled vs. consequences that backfired
  - Political outcomes influenced
  - Sanctuary development achieved (buildings, upgrades)
  - Pythia health and legend status
- **Narrative summary**: A 2-3 sentence authored summary of the Oracle's legacy in this timeline
- **Unlock feedback**: New scenarios, traits, tile sets, or modifiers unlocked
- **Seed display**: For sharing and competitive comparison

### 4.2 Run Summary Narrative

#### Gold Standard: Into the Breach's Timeline Display

Into the Breach shows each timeline attempt as a numbered entry, creating a visual history of the player's attempts. Failed timelines are not erased -- they're acknowledged.

**Applied to The Oracle:**
- Each completed scenario (win or loss) should be recorded in a persistent chronicle
- The chronicle entry should include: scenario name, seed, score, date played, and the authored narrative summary
- Losses should have their own dignified entries ("The Oracle fell silent in the third month when...")
- This chronicle should be accessible from the main menu

### 4.3 Unlock Feedback

#### Principle: Unlocks Should Feel Discovered, Not Granted

The best unlock moments combine surprise with explanation:

1. **Visual fanfare**: The new item/option appears with a distinct reveal animation
2. **Clear label**: "New Scenario Unlocked: The Siege of Corinth"
3. **Brief description**: One sentence explaining what was unlocked
4. **Trigger explanation**: "Achieved by reaching Oracle renown 50 in any scenario"
5. **Path forward**: The unlocked content is marked as "NEW" in the relevant menu

#### Anti-Pattern: Unlock Dump

Showing 8 unlocks at once after a run dilutes the impact of each. If multiple unlocks occur, present them sequentially with brief pauses between each.

#### Anti-Pattern: Silent Unlocks

If content unlocks without any feedback, the player may not notice and may not understand what they did to earn it. Every unlock needs a moment.

### 4.4 "Play Again" Flow

#### Gold Standard: Hades' Seamless Return

After death in Hades, the player's character literally walks back through the House of Hades, talks to NPCs, upgrades at the Mirror, and then walks to the next run entrance. There is no "Play Again" button -- the return to gameplay IS the play-again flow.

**Why it works:**
- Zero friction between runs
- The hub provides meaningful between-run decisions (upgrades, loadout)
- Narrative continues in the hub, so returning is rewarding, not punishing

#### Gold Standard: Slay the Spire's "Main Menu" or "Embark"

After the score screen, Slay the Spire offers two buttons: "Main Menu" (to change character or settings) and "Embark" (to immediately start a new run with the same character). The Embark button is larger and more prominent.

**Applied to The Oracle:**
- After the legacy screen, offer:
  - **"Consult the Fates Again"** (prominent): Start new run with same scenario/settings
  - **"Choose a New Path"** (secondary): Return to scenario selection
  - **"Return to the Agora"** (tertiary): Main menu
- The default/prominent action should have zero friction -- one click to begin again
- Between-run decisions (if any hub mechanics exist) should live between the legacy screen and the next run start

---

## 5. Tutorial / Onboarding

### 5.1 First-Time Player Guidance Patterns

#### Gold Standard: Hades' Invisible Tutorial

Hades teaches through play. The first run is implicitly tutorial-shaped: simpler enemies, fewer mechanics, key NPCs offer contextual dialogue. No tutorial popup ever says "This is how you dash." The dash move is introduced by encounter design that makes dashing necessary.

**Why it works:**
- The player learns by doing, not by reading
- The first run feels like a real run, not a tutorial ghetto
- Mastery comes from repetition across runs, not from a front-loaded lesson

#### Gold Standard: Slay the Spire's Gated Content Unlock

Slay the Spire's first run uses the Ironclad character with a starter deck. Additional characters, relics, and cards unlock through play. The player's first experience is with the simplest possible game state.

**Why it works:**
- Reduced decision space on first encounter
- Each subsequent unlock adds one new element to learn
- Players never feel overwhelmed because complexity scales with experience

#### Gold Standard: Strange Horticulture's Day-by-Day Escalation

Strange Horticulture introduces one new puzzle type per day. Day 1: identify one plant by description. Day 2: use the reference book. Day 3: use the map. Each day adds exactly one new tool or challenge type.

**Why it works:**
- Each session has one clear learning objective
- New tools arrive in context (you need the map because a customer gives you a location clue)
- Mastery of yesterday's tool is assumed before today's tool arrives

#### Applied to The Oracle:

First scenario should be tutorial-shaped without being labeled "tutorial":
- **First consultation**: Present a simple omen with a small tile pool (4-6 tiles, 2-3 slots). The petitioner's needs are straightforward. The scoring feedback is explicit and generous.
- **First consequence**: Arrives quickly (within 1-2 days, not the standard longer delay). The cause-and-effect chain is visible.
- **First faction interaction**: One faction, one clear disposition, one trade option.
- **First building placement**: One available building type, one obvious placement location.
- Progressive unlocking: Additional tile categories, omen complexity, faction interactions, and building types unlock across the first 3-5 consultations.

### 5.2 Contextual Help Systems

#### Gold Standard: Into the Breach's Animated Tooltips

Into the Breach uses animated tooltips that dynamically demonstrate how a weapon works. Instead of "This weapon pushes enemies one tile," the tooltip shows a looping animation of the push happening on a miniature grid. The developers stated: "Sacrifice cool ideas for the sake of clarity every time."

**Why it works:**
- Shows rather than tells
- The animation plays on a miniature version of the actual game interface
- No reading required to understand the core mechanic

#### Gold Standard: Crusader Kings III's Encyclopedia + Tooltips

CK3 uses nested tooltips (tooltips within tooltips) and a searchable encyclopedia. Any bolded term in any tooltip can be hovered to reveal its own tooltip. The encyclopedia provides deeper reference for players who want full mechanical detail.

**Why it works:**
- Information depth is available on demand, never forced
- The nesting means the player can drill down from any starting point
- Bold text signals "this term has a definition" without cluttering the base text

#### Applied to The Oracle:

- **Tile tooltips**: Hovering a word tile should show its visible literary meaning. Advanced tooltip (right-click or long-press) could show mechanical guidance ("This tile tends toward high ambiguity")
- **Score tooltips**: Hovering the scoring indicator during crafting should explain what is contributing positively and negatively
- **Faction tooltips**: Hovering a faction name should show current disposition, recent history, and key interests
- **Building tooltips**: Already covered in the build menu standards document
- **Contextual prompts**: First-time encounters with new mechanics should trigger a brief, dismissible tooltip that explains the mechanic in one sentence and shows a visual example
- **Glossary/Codex**: A searchable reference accessible from the pause menu, organized by category (Prophecy Crafting, Sanctuary Management, Faction Politics, Pythia Management)

### 5.3 Progressive Feature Revelation

#### Principle: Never Show a Button the Player Cannot Use

If a feature is locked, do not show its full UI. Show either nothing or a locked indicator with unlock conditions. This prevents the "what does this button do?" anxiety that comes from visible-but-non-functional UI elements.

**Staged revelation plan for The Oracle:**

| Phase | Mechanics Available | UI Elements Visible |
|-------|-------------------|-------------------|
| First consultation | Basic tiles, simple omens, prophecy delivery | Tile pool, crafting surface, delivery button, score indicator |
| After first consequence | Consequence log, basic faction reactions | Event feed, faction panel (single faction) |
| After 3 consultations | Full tile pool, complex omens, Pythia fatigue | Expanded tile categories, Pythia status panel |
| After first month | Faction politics, trade, monthly events | Full faction watchlist, trade panel, political event popups |
| After first building | Build menu, resource management | Build button, resource bar, storehouse panel |
| After first Pythia rest | Pythia management, rest/purify actions | Pythia action buttons, trait display |

#### Anti-Pattern: Tutorial Pop-Up Cascade

Showing 5-6 tutorial popups in sequence ("Click here to open the build menu. Click here to see your resources. Click here to view factions.") teaches nothing because the player is clicking "OK" reflexively without absorbing information. Each tutorial moment should be separated by actual gameplay.

#### Anti-Pattern: Forced Tutorial With No Skip

Experienced players or returning players should always have the option to skip onboarding content. A "Skip Tutorial" option should be visible on the setup screen and in the first consultation overlay.

### 5.4 "What Should I Do?" Assistance

#### Gold Standard: Civilization VI's Advisor System

Civilization VI offers advisor recommendations ("Science Advisor suggests researching Sailing") that the player can follow or ignore. The advisor suggestions are visible but non-blocking.

**Applied to The Oracle:**

The Oracle has advisors as a content type already. Advisor assistance should work as follows:
- Advisors should offer contextual suggestions in the event feed: "The Pythia grows tired. Consider allowing her to rest."
- During consultations, advisor commentary could appear as brief margin notes: "This petitioner is aligned with Athens. Favorable prophecies may strengthen the alliance."
- Advisor suggestions should be dismissible and toggleable (Settings > Gameplay > Show advisor hints)
- First-time players should have advisor hints enabled by default; the setting should be surfaced after the first few consultations ("Would you like to continue receiving advisor guidance?")

#### Gold Standard: Return of the Obra Dinn's Progress Validation

Obra Dinn validates deductions in sets of three. You can make guesses at any time, but confirmation only arrives when three fates are correctly identified simultaneously. This creates a "check your work" rhythm without hand-holding.

**Applied to The Oracle:**
- After a prophecy is crafted but before delivery, show a quality assessment range (not a precise score): "This prophecy reads as [Cautious / Bold / Reckless]" and "Likely consequences: [Modest / Significant / Severe]"
- This guidance helps the player calibrate without removing uncertainty
- The assessment should use thematic language, not numerical ratings

---

## 6. Design Expert Principles Applied

### 6.1 Jenova Chen: Flow State and Emotional Arc

**Core principle:** The UI must support a dynamic difficulty balance where the player's skill and the game's challenge remain in harmony. When the player is overwhelmed, the UI should simplify (fewer distractions, clearer priorities). When the player is coasting, the UI should surface more complexity (additional tile categories, harder omens, faction complications).

**Specific applications:**
- Consultation difficulty should escalate gradually across a scenario, not spike randomly
- The UI should never create artificial difficulty through poor information design -- all difficulty should come from meaningful strategic decisions
- Emotional pacing: consultations should alternate between high-stakes (political crisis, angry faction) and lower-stakes (routine reading, friendly petitioner) encounters to create an emotional rhythm
- The transition between management mode and consultation mode should itself be a pacing device: a moment of mental gear-shifting that separates strategic planning from ritual performance

### 6.2 Fumito Ueda: Atmosphere Through Restraint

**Core principle:** Remove everything that does not serve the core experience. The UI should feel like it belongs in the world, not like software chrome floating above it.

**Specific applications:**
- The consultation overlay should contain NO elements from the management layer (no resource counts, no build menu, no minimap)
- Use negative space deliberately: an empty tile slot is meaningful
- Sound design should do heavy lifting for atmosphere -- a subtle ambient shift when entering consultation mode is worth more than a visual popup
- Character animation (Pythia's posture, petitioner's demeanor) should convey information that would otherwise require UI labels
- The pause menu should be the simplest possible overlay: minimal chrome, clear options, immediate dismissal

### 6.3 Raph Koster: Fun Through Learning and Pattern Recognition

**Core principle:** Fun arises from mastery. The UI should make learning curves visible and pattern recognition rewarding.

**Specific applications:**
- The scoring system should be learnable: players should be able to form mental models of "what makes a good prophecy" through repeated play
- The consequence system should be partially predictable: experienced players should recognize patterns ("high-ambiguity prophecies to aggressive factions tend to cause problems")
- The tile system should have discoverable synergies that the UI surfaces through subtle feedback (tiles that combine well might produce a visual resonance)
- The progression system (unlocks, ascension) should make the player's growing mastery visible and rewarding
- Avoid randomness that cannot be read -- every random element should have a learnable distribution

### 6.4 Jakob Nielsen: Usability Heuristics

**Most relevant heuristics for The Oracle:**

| Heuristic | Application |
|-----------|-------------|
| **Visibility of system status** | Always show: current day, Pythia fatigue, pending consequences count, active faction dispositions |
| **Match between system and real world** | Use thematic language (not "Score: 47" but "The prophecy resonates with moderate clarity") |
| **User control and freedom** | Tiles can be placed AND removed during crafting. Undo is always available before commitment. |
| **Consistency and standards** | Same visual language for all interactive elements. Tooltips behave identically everywhere. |
| **Error prevention** | Warn before delivering a prophecy with known risks. Confirm before overwriting a save. |
| **Recognition rather than recall** | Show all available tiles simultaneously. Show faction names, not just icons. |
| **Flexibility and efficiency** | Keyboard shortcuts for experienced players. Mouse-only play for casual sessions. |
| **Aesthetic and minimalist design** | No decorative UI elements that don't serve gameplay or atmosphere. |
| **Help users recover from errors** | Allow prophecy revision before delivery. Allow save-scumming unless explicitly roguelike mode. |
| **Help and documentation** | Codex/glossary accessible from pause menu. Contextual tooltips everywhere. |

---

## 7. 10/10 Criteria Checklist

A 10/10 implementation of The Oracle's gameplay interaction and meta UI would satisfy ALL of the following criteria:

### Consultation / Ritual Interaction

- [ ] The consultation screen feels like entering a sacred space, not opening a dialog box
- [ ] Transition in/out of consultation mode includes atmospheric animation and sound
- [ ] The management HUD is dimmed or hidden during consultations
- [ ] Word tiles look and behave like physical ritual objects (tactile placement, satisfying sounds)
- [ ] The tile pool is fully visible without scrolling
- [ ] The crafting surface shows all placed tiles and all empty slots simultaneously
- [ ] Live scoring feedback updates with each tile placed/removed
- [ ] Tiles can be freely placed and removed before commitment (full reversibility)
- [ ] The delivery action feels weighty and ceremonial (not a simple button click)
- [ ] Prophecy delivery includes a brief cinematic moment (Pythia speaks, atmosphere shifts)
- [ ] Petitioner reaction is immediate and narratively authored
- [ ] Consequence arrivals (days later) are dramatic reveals, not log entries
- [ ] No information needed for decision-making is hidden behind navigation

### Run Setup / New Game

- [ ] Default flow requires zero configuration decisions (one click to start playing)
- [ ] Scenario selection shows visual preview, narrative premise, and difficulty indicator
- [ ] Difficulty modifiers are individually toggleable with thematic names and clear descriptions
- [ ] Combined difficulty score is visible for run comparison
- [ ] Locked scenarios show shadowed previews with unlock conditions
- [ ] Seed is visible and shareable; manual seed entry is available but secondary
- [ ] Start button uses thematic language and triggers an atmospheric transition
- [ ] No mandatory configuration gates the start button

### Save / Load / Settings

- [ ] Save is accessible from pause menu (Esc) AND via keyboard shortcut (F5)
- [ ] Load is accessible from pause menu AND via keyboard shortcut (F9)
- [ ] Auto-save occurs at natural breakpoints with a brief, unobtrusive indicator
- [ ] Auto-save slots are separate from manual save slots
- [ ] Settings are organized in categorized tabs (Gameplay, Display, Audio, Controls, Accessibility)
- [ ] Settings changes preview in real-time where possible
- [ ] Settings are accessible from both main menu and pause menu
- [ ] Quick-save overwrites silently (no confirmation dialog)
- [ ] All settings persist across sessions
- [ ] The save/load screen shows scenario name, date, playtime, and a thumbnail/description

### End-of-Run / Legacy

- [ ] Score breakdown reveals categories sequentially (animated, not all-at-once)
- [ ] A narrative summary of the run is authored and displayed prominently
- [ ] Unlocks are presented individually with reveal animation and trigger explanation
- [ ] Every run (win or loss) is recorded in a persistent chronicle
- [ ] The "play again" button is prominent and uses thematic language
- [ ] Returning to a new run requires at most one click from the legacy screen
- [ ] Seed is displayed on the legacy screen for sharing
- [ ] Losses are treated with dignity, not punishment

### Tutorial / Onboarding

- [ ] First scenario teaches through play, not through popup text
- [ ] Mechanics are introduced one at a time with gameplay between each introduction
- [ ] No UI button is visible before the player has the context to use it
- [ ] Contextual tooltips appear at first encounter and are dismissible
- [ ] An advisor system offers optional contextual suggestions
- [ ] A codex/glossary is accessible from the pause menu
- [ ] Experienced players can skip all onboarding content
- [ ] Progressive revelation follows a defined phase plan (see Section 5.3)

### Cross-Cutting

- [ ] All interactive elements have consistent hover/focus/active states
- [ ] Keyboard shortcuts exist for all frequent actions
- [ ] Mouse-only play is fully supported
- [ ] Colorblind mode is available
- [ ] Text is scalable (accessibility)
- [ ] No animation blocks input (all animations can be interrupted or skipped)
- [ ] The game never enters a state where the player's only option is unclear
- [ ] The pacing follows the approach-preparation-crafting-commitment-resolution-aftermath arc

---

## Sources

### Game-Specific References
- [Slay the Spire UI/UX Redesigns Analysis](https://medium.com/@n01578837/final-deliverable-632cfc09e673)
- [Slay the Spire Interface Screenshots](https://interfaceingame.com/games/slay-the-spire/)
- [Slay the Spire Score System](https://slay-the-spire.fandom.com/wiki/Score)
- [Hades UI Database](https://www.gameuidatabase.com/gameData.php?id=534)
- [Hades HUD Redesign Analysis](https://medium.com/@bramhadalvi/hud-redesign-fdc332d05291)
- [Hades Responsive Underworld Design](https://medium.com/@Nat.Rowley/how-hades-creates-a-responsive-underworld-915715a7c2a)
- [Into the Breach: Sacrifice Cool Ideas for Clarity](https://www.gamedeveloper.com/design/-i-into-the-breach-i-dev-on-ui-design-sacrifice-cool-ideas-for-the-sake-of-clarity-every-time-)
- [Into the Breach UI Database](https://www.gameuidatabase.com/gameData.php?id=483)
- [Inscryption's Journey from Game Jam to Cult Classic](https://www.gamedeveloper.com/marketing/-i-inscryption-s-i-journey-from-game-jam-joint-to-cult-classic)
- [Diegetic Interface (TV Tropes)](https://tvtropes.org/pmwiki/pmwiki.php/Main/DiegeticInterface)
- [Cultist Simulator and Cosmic Horror of Board Games](https://medium.com/@gatherer286/confronting-the-unknowable-cultist-simulator-and-the-cosmic-horror-of-board-games-1382620876c3)
- [Cultist Simulator UI Critique](https://steamcommunity.com/app/718670/discussions/0/1737715419894151581/)
- [FTL Event Choice Quality](https://steamcommunity.com/app/212680/discussions/0/412448158150944538/)
- [FTL Designer Review](https://gamedesignstrategies.wordpress.com/2012/09/29/ftl-faster-than-light-designer-review/)
- [Strange Horticulture Deep Dive](https://www.gamedeveloper.com/design/deep-dive-strange-horticulture)
- [Return of the Obra Dinn Design Innovation](https://www.kokutech.com/blog/gamedev/design-patterns/unique-mechanics/return-of-the-obra-dinn)
- [Return of the Obra Dinn Lateral Information](https://atomicbobomb.home.blog/2020/03/21/return-of-the-obra-dinn-lateral-information/)
- [Return of the Obra Dinn Deduction Analysis](https://frostilyte.ca/2019/05/16/a-look-at-deduction-in-return-of-the-obra-dinn/)
- [Crusader Kings III Console UI/UX Dev Diary](https://www.paradoxinteractive.com/games/crusader-kings-iii/news/ck3-console-dev-diary-3-uiux-and-controls)
- [Civilization VI Interface Tips](https://www.gamepressure.com/sidmeierscivilization6/interface/ze92ba)

### Design Theory References
- [Jenova Chen: Flow in Games (MFA Thesis)](https://www.jenovachen.com/flowingames/Flow_in_games_final.pdf)
- [Jenova Chen: Flow in Games (ACM)](http://jenovachen.com/flowingames/p31-chen.pdf)
- [Jenova Chen on Game Design](https://famousaspect.com/what-is-game-design-with-jenova-chen/)
- [Fumito Ueda: Design by Subtraction](https://demasiri.wordpress.com/2017/01/22/design-by-subtraction/)
- [Fumito Ueda: Art of Subtraction in Shadow of the Colossus](https://indiegamesdevel.com/fumito-ueda-and-the-language-of-subtraction/)
- [Raph Koster: A Theory of Fun for Game Design](https://www.theoryoffun.com/)
- [Jakob Nielsen: 10 Usability Heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/)
- [Nielsen's Heuristics Applied to Complex Applications](https://www.nngroup.com/articles/usability-heuristics-complex-applications/)

### General Game UI/UX References
- [Game UI Database](https://gameuidatabase.com/)
- [Interface In Game](https://interfaceingame.com/)
- [Diegetic Interfaces in Game Design](https://www.wayline.io/blog/diegetic-interfaces-game-design)
- [UI Strategy Game Design Dos and Don'ts](https://www.gamedeveloper.com/design/ui-strategy-game-design-dos-and-don-ts)
- [Game UX Best Practices for Onboarding](https://inworld.ai/blog/game-ux-best-practices-for-video-game-onboarding)
- [Progressive Disclosure in Game Design](https://medium.com/the-acagamic-tip-tuesday/the-acagamic-tip-tuesday-11-progressive-disclosure-e3822dac00a)
- [Roguelike Seed Systems](https://www.gridsagegames.com/blog/2017/05/working-seeds/)
- [Designing Fair RNG in Roguelikes](https://medium.com/@JeongHyeonUk/designing-fair-rng-in-roguelikes-balancing-luck-and-skill-7b967230e961)
