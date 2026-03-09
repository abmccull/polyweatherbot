# The Oracle — UI/UX Design Standards

Best-in-class visual design, color systems, typography, and aesthetic cohesion for a historical ancient-Greek-themed strategy and management game.

---

## Governing Philosophy

Three design thinkers anchor this system:

- **Dieter Rams** — "Good design is as little design as possible." Every element must earn its place. If it does not aid comprehension or atmosphere, remove it.
- **Edward Tufte** — Maximize the data-ink ratio. Every pixel of UI chrome that is not conveying game state is chartjunk. Theme the chrome, but never let it obscure the data.
- **Jan Tschichold** — Classical typography demands proportion, not decoration. Golden-ratio spacing, generous margins, and restrained ornament produce gravitas without noise.

**The Oracle rule**: The UI should feel like a well-preserved bronze tablet — warm, legible, authoritative, and worn just enough to prove it has been used.

---

## 1. Color System

### 1.1 Palette Construction

The palette draws from historically attested Greek pigments and materials: ochre, terracotta, malachite green, azurite blue, lamp-black, and bronze. The existing token set is sound; this section formalizes it into a complete system.

#### Foundation Layer (backgrounds and surfaces)

| Token | Hex | Role |
|---|---|---|
| `--stone-dk` | `#1a1408` | Deepest background — the "night stone" behind everything |
| `--surface` | `rgba(46,34,16,0.92)` | Panel interiors, card backgrounds |
| `--surface-hover` | `rgba(60,44,20,0.95)` | Interactive hover state |
| `--surface-light` | `rgba(200,151,42,0.08)` | Subtle highlight wash on pills, chips |
| `--ink` | `#2a1f0e` | Dense foreground surface (tooltip bg, dropdown bg) |

#### Text Layer

| Token | Hex | Role |
|---|---|---|
| `--text` | `#f5e6c8` | Primary body text — warm parchment |
| `--text-mid` | `rgba(245,230,200,0.80)` | Secondary labels, timestamps |
| `--text-dim` | `rgba(245,230,200,0.64)` | Tertiary captions, disabled text |
| `--parchment` | `#f5e6c8` | Full-strength parchment for emphasis |

#### Accent Layer (the bronze family)

| Token | Hex | Role |
|---|---|---|
| `--gold` | `#c8972a` | Primary accent — titles, active borders, selected state |
| `--gold-dim` | `#8a6a1e` | Muted accent — inactive tabs, ornamental lines |
| `--gold-bright` | `#e8c060` | Highlight accent — hover glow, notification pulse |

#### Semantic Layer

| Token | Hex | Role | Mnemonic |
|---|---|---|---|
| `--green` | `#5a9e4a` | Positive — income, growth, success, healing | Olive grove thriving |
| `--red` | `#c0513a` | Negative — expense, danger, damage, decline | Terracotta fired too hot |
| `--blue` | `#5a8abe` | Neutral-informational — navigation, links, faction diplomacy | Aegean sea |
| `--amber` | `#d4a032` | Warning — approaching threshold, caution | Beeswax lamplight |
| `--purple` | `#8b6bb0` | Sacred/divine — prophecy power, omen strength, Pythia | Tyrian dye |

### 1.2 Palette Rules

**Gold standard**: Old World and Crusader Kings III both limit their palettes to 5-7 core hues. The Oracle uses exactly 5 semantic colors (green, red, blue, amber, purple) on top of the bronze-gold accent family and the stone-parchment foundation. This is the ceiling.

**Principle** (Tufte): "Use color as a data tool. Highlight one series or one key category. De-emphasize everything else." Color must encode meaning, not decorate.

**Anti-patterns**:
- Rainbow dashboards with 10+ hues competing for attention
- Using color as the *sole* differentiator (always pair with icon or label)
- Neon or saturated modern accent colors (electric blue, hot pink) — these break the ancient-world atmosphere
- Using red and green as the only status pair without luminance difference (colorblind users cannot distinguish them)

**10/10 criteria**:
- [ ] No more than 3 colors visible simultaneously in any single panel
- [ ] Every semantic color has a text label or icon backup (never color-only encoding)
- [ ] Accent color (gold) is used for exactly one purpose per context: either "selected" or "important," never both simultaneously
- [ ] Semantic colors maintain 4.5:1 contrast ratio against their background
- [ ] The palette could be described in one sentence: "Bronze and parchment on dark stone, with olive, terracotta, and sea-blue for status"

### 1.3 Contrast Requirements

Following WCAG 2.1 AA as a floor:

| Element | Minimum Ratio | Target Ratio |
|---|---|---|
| Body text on `--surface` | 4.5:1 | 7:1 |
| Large text (>=18px) on `--surface` | 3:1 | 4.5:1 |
| Icon/graphic on background | 3:1 | 4.5:1 |
| Interactive component border | 3:1 | 3:1 |
| `--text` (#f5e6c8) on `--stone-dk` (#1a1408) | ~14:1 | Excellent |
| `--text-dim` on `--stone-dk` | ~8:1 | Good |
| `--gold` (#c8972a) on `--stone-dk` | ~6:1 | Good |
| `--red` (#c0513a) on `--stone-dk` | ~4.2:1 | Acceptable — do not go dimmer |

**Rule**: Never render `--text-dim` at font sizes below 14px. At that opacity and size, it becomes illegible on textured backgrounds.

---

## 2. Typography

### 2.1 Font Selection

**Primary typeface**: Georgia (serif) — already in use. This is a strong choice.

Georgia was designed for screen rendering, has excellent hinting at small sizes, and carries a warm, classical personality without the stiffness of Times New Roman. It reads as "literate and historical" rather than "academic newspaper."

**Rationale by reference**:
- *Old World* uses a warm serif with classical proportions for body text
- *Crusader Kings III* uses serif for narrative/flavor text and sans-serif for data
- *Hades* uses bold display serifs for titles and clean sans for HUD numbers
- *Civilization VI* uses clean sans-serif for data readability with serif for cultural text

**Recommended stack**:

| Role | Font | Fallback | Weight |
|---|---|---|---|
| Display/title | Georgia | "Times New Roman", serif | 700 (bold) |
| Body/narrative | Georgia | "Times New Roman", serif | 400 (regular) |
| Data/numbers | system-ui | -apple-system, sans-serif | 600 (semi-bold) |
| Labels/caps | Georgia | "Times New Roman", serif | 400 + `letter-spacing: 0.08em` + `text-transform: uppercase` |
| Monospace (debug) | "Courier New" | monospace | 400 |

**Why sans-serif for numbers**: Numeric data in resource pills, stat bars, and counters must be instantly scannable. Sans-serif numerals have uniform width (tabular figures) and higher x-height, making columns align and comparisons faster. This is the Civ VI / Slay the Spire principle — theme the words, clarify the math.

**Future upgrade path**: If the project adopts custom fonts, consider:
- **Cinzel** (display) — designed specifically for classical/ancient aesthetic, free on Google Fonts
- **Cormorant Garamond** (body) — elegant oldstyle serif with Greek character support
- **Inter** or **DM Sans** (data) — highly legible, tabular numerals, open source

### 2.2 Type Scale

Use a **Major Third** ratio (1.250) for moderate contrast appropriate to information-dense strategy panels. The Golden Ratio (1.618) creates too much jump between levels for dense data screens; Minor Third (1.200) provides too little hierarchy for a game with narrative and atmospheric text.

| Level | Name | Size | Weight | Usage |
|---|---|---|---|---|
| 1 | Display | 28px | 700 | Screen titles ("The Oracle of Delphi"), overlay headings |
| 2 | Heading | 22px | 700 | Panel titles ("Consultation"), section headers |
| 3 | Subheading | 18px | 600 | Sub-section titles, faction names, building names |
| 4 | Body | 14px | 400 | Narrative text, event descriptions, flavor text |
| 5 | Body-small | 13px | 400 | Dense panel text, tooltips, secondary descriptions |
| 6 | Label | 11px | 400 | Resource labels, chip text, uppercase micro-labels |
| 7 | Caption | 10px | 400 | Timestamps, debug info — absolute minimum size |

**Floor**: 10px is the absolute minimum for any text. Below 10px, no serif font remains legible on screen. The existing `0.58rem` labels (~9.3px) should be raised to `0.69rem` (11px).

**Line height**:
- Body text: 1.5 (generous — enables reading in narrow panels)
- Labels and captions: 1.15-1.25 (tighter — these are scanned, not read)
- Headings: 1.1-1.2 (tight — display text should feel solid)

### 2.3 Type Hierarchy Rules

**Gold standard** (Müller-Brockmann): A well-designed page uses exactly 3 levels of emphasis. More than 4 levels in a single panel means the information architecture needs restructuring, not more type sizes.

**Principle** (Lupton): "Hierarchy is conveyed by contrast — size, weight, color, and spacing all contribute. Never rely on size alone."

**Anti-patterns**:
- More than 4 font sizes in one panel
- Bold and italic and color all applied simultaneously
- Decorative/script fonts for body text or data
- Centered body text (left-align everything except headings)
- Justified text in narrow panels (creates rivers of whitespace)

**10/10 criteria**:
- [ ] Any panel can be understood at a glance by reading only the largest text
- [ ] Numbers are immediately distinguishable from labels — different weight or font family
- [ ] No text smaller than 10px anywhere in the UI
- [ ] Line length never exceeds 65 characters (optimal is 45-55)
- [ ] Every piece of text has exactly one role: title, label, value, or description — never ambiguous

---

## 3. Iconography

### 3.1 Icon System Philosophy

The existing UI uses emoji/Unicode for icons (olive branch, flame, laurel, grain, amphora). This is functional for rapid prototyping but not a final system. The target is a **cohesive pictographic icon set** with the following properties:

**Style**: Filled silhouette icons with 2px stroke, inspired by:
- Greek pottery black-figure silhouettes (Apotheon's treatment)
- Coin engravings and seal impressions (simple, bold, readable at small sizes)
- The warm, heavy line quality of bronze engraving rather than thin modern line icons

**Not**: Outlined/line icons (too modern, too "SaaS"), 3D icons (too mobile-game), or photorealistic icons (too noisy at small sizes).

### 3.2 Icon Sizing

| Context | Icon Size | Touch/Click Target | Padding |
|---|---|---|---|
| Resource pill (inline) | 14-16px | 28px | 4px around |
| Panel toolbar button | 20px | 36px | 8px around |
| Primary action button | 24px | 44px | 10px around |
| Status badge | 12px | n/a (not interactive) | 2px around |
| Map overlay | 32px | 48px | 8px around |

**Minimum interactive target**: 36x36px for mouse-primary desktop games. This is tighter than the 48px mobile standard but appropriate for the platform. Console/controller ports would need 48px minimum.

### 3.3 Icon + Text Rules

**Gold standard** (Civ VI): Icons paired with text labels on first encounter, icon-only after player has learned the system. Tooltips always available.

**Principle**: Icons accelerate recognition for experts; labels ensure comprehension for novices. Both must coexist.

| Pattern | When to use |
|---|---|
| Icon + label | First 3 encounters with a concept; all menu items; all buttons with consequences |
| Icon + value | Resource counts, stat displays, compact HUD elements (after player is oriented) |
| Icon only | Repeating elements in a learned grid (e.g., building palette after tutorial) |
| Label only | Long-form narrative, event descriptions, dialogue |

**Anti-patterns**:
- Icon-only buttons with no tooltip (the "mystery meat navigation" problem)
- Different icons for the same concept in different panels
- More than 1 icon per label (icon stacking creates visual noise)
- Icons that require color to be understood (must be distinguishable in monochrome)

**10/10 criteria**:
- [ ] Every icon has a tooltip that appears within 300ms of hover
- [ ] Icons are distinguishable at 14px (the smallest they will ever appear)
- [ ] The same concept uses the same icon everywhere (olive = oil, amphora = storage, flame = sacred fire)
- [ ] A new player can identify at least 80% of icons without tooltips after 15 minutes of play
- [ ] Icons maintain silhouette clarity against both light and dark backgrounds

---

## 4. Component Design Language

### 4.1 Buttons

Four tiers, styled to feel like bronze-age interface artifacts rather than modern web buttons:

#### Primary (consecrate, confirm, build)
```
Background:     linear-gradient(180deg, #c8972a, #8a6a1e)
Border:         1px solid #e8c060
Text:           #1a1408 (dark ink on gold — high contrast)
Border-radius:  6px
Shadow:         0 2px 8px rgba(0,0,0,0.3)
Hover:          brighten gradient 10%, border glow
Active:         inset shadow, slight darken
```
Feel: A stamped bronze token. Important actions feel weighty.

#### Secondary (open panel, navigate, filter)
```
Background:     var(--surface)
Border:         1px solid var(--panel-border)
Text:           var(--text)
Hover:          border color shifts to --gold-dim
```
Feel: An inscribed stone tile. Visible but not demanding attention.

#### Destructive (abandon, demolish, refuse)
```
Background:     transparent
Border:         1px solid var(--red)
Text:           var(--red)
Hover:          background fills with rgba(192,81,58,0.15)
```
Feel: A warning scratched into stone. Not inviting, deliberately restrained.

#### Disabled
```
Background:     var(--surface-light)
Border:         1px solid rgba(200,151,42,0.10)
Text:           var(--text-dim)
Cursor:         not-allowed
Opacity:        0.5
```
Feel: Weathered away. Clearly inert.

**Anti-patterns**:
- More than 1 primary button visible at a time per panel
- Ghost buttons (no border, no background) for important actions
- Buttons that look like text links (underlined, no border/background)
- Animated/pulsing buttons for non-critical actions
- Button text longer than 3 words

### 4.2 Panels and Cards

Panels are the primary container. They should feel like parchment or bronze tablets floating over the game world.

**Panel anatomy**:
```
Border:         1px solid var(--panel-border)
Background:     var(--panel-bg) — gradient from #2e2210 to #1e180a
Shadow:         0 8px 32px rgba(0,0,0,0.45)
Border-radius:  14px (--radius-lg) for major panels, 10px for sub-cards
Padding:        16px body, 12px header
```

**Panel header pattern**:
- Left-aligned title in Heading size (22px, bold, `--gold`)
- Optional subtitle in Body-small (`--text-mid`)
- Close/collapse button right-aligned, 36px target
- 1px `--panel-border` separator below header
- 8px gap between header separator and body content

**Card pattern** (for items within panels — buildings, factions, prophecies):
- Background: `var(--surface)` (slightly lighter than panel)
- Border: `1px solid var(--panel-border)` — appears on hover only (initially borderless)
- Border-radius: `var(--radius-md)` (10px)
- Padding: 10px
- Hover: border appears, background shifts to `--surface-hover`
- Selected: border becomes `--gold`, left accent line 3px solid `--gold`

**Reference**: Crusader Kings III uses subtle parchment texture gradients with gold-leaf borders on major panels and borderless cards that gain definition on hover. This is the target feel.

### 4.3 Badges and Status Chips

Status chips appear on faction cards, building states, and resource conditions.

**Chip anatomy**:
```
Padding:        2px 8px
Border-radius:  var(--radius-sm) (6px)
Font-size:      11px (Label tier)
Text-transform: uppercase
Letter-spacing: 0.08em
Font-weight:    600
```

**Chip variants**:

| Variant | Background | Text | Usage |
|---|---|---|---|
| Positive | `rgba(90,158,74,0.15)` | `--green` | "Active," "Surplus," "Allied" |
| Negative | `rgba(192,81,58,0.15)` | `--red` | "Damaged," "Hostile," "Deficit" |
| Warning | `rgba(212,160,50,0.15)` | `--amber` | "Low Stock," "Tensions," "Waning" |
| Neutral | `var(--surface-light)` | `--text-mid` | "Idle," "Pending," "Standard" |
| Sacred | `rgba(139,107,176,0.15)` | `--purple` | "Prophesied," "Consecrated," "Divine" |

**Principle**: Chips use tinted backgrounds at 15% opacity with full-color text. This creates readable, thematic labels without the visual weight of solid-color badges, and maintains contrast on the dark UI.

### 4.4 Progress Bars and Meters

**Bar anatomy**:
```
Track:          var(--surface-light), 6px height, full border-radius
Fill:           semantic color gradient (left bright, right dim)
Border:         none on track; 1px solid rgba(color, 0.3) on fill
Height:         6px for inline, 8px for standalone, 12px for major meters
Border-radius:  full (half of height)
```

**Meter patterns** by game context:

| Meter | Fill Color | Empty State | Usage |
|---|---|---|---|
| Resource quantity | `--gold` | `--surface-light` | Storehouse levels, reserve amounts |
| Health/integrity | `--green` -> `--amber` -> `--red` | Grayed fill | Building condition |
| Sacred power | `--purple` | Dim purple wash | Pythia's clarity, omen strength |
| Progress/completion | `--blue` | `--surface-light` | Construction, research, preparation |
| Faction standing | `--green` (allied) / `--red` (hostile) | Center-anchored | Diplomatic relationship |

**Anti-patterns**:
- Bars without numeric labels (the number must always be visible)
- Color-only bar state changes (pair with icon or label)
- Bars thinner than 4px (unreadable, especially for partially-filled states)
- Animated bars for non-changing values (animation implies activity)

### 4.5 Stat Displays

Strategy games live and die by stat readability. Three canonical patterns:

**Pattern A: Icon + Number** (resource pills)
```
[icon] 247
```
Compact. Use for HUD top-bar, inline mentions. The existing `resource-pill` pattern does this well.

**Pattern B: Label / Value stack** (building detail panels)
```
Reputation          87
Monthly Income      +12
Upkeep              -4
```
Left-aligned label, right-aligned value. Monospace or tabular-figure font for values. Add `--green`/`--red` to +/- values.

**Pattern C: Bar + Number overlay** (meters)
```
[████████░░] 78/100
```
For quantities with known caps. The number sits right-aligned beside or inside the bar.

**Gold standard** (Slay the Spire): Critical stats (health, energy) are large and prominent. Secondary stats (relic counts, deck size) are smaller and accessible. Tertiary stats are hidden behind tooltips. This three-tier visibility model applies directly to The Oracle: piety/gold/prestige at tier 1, individual resource counts at tier 2, per-building stats at tier 3.

---

## 5. Atmospheric Design

### 5.1 Making It Feel Ancient Greek Without Being Kitschy

This is the hardest design challenge. The line between "evocative" and "theme park" is thin.

**Gold standard references**:
- *Old World*: Warm, understated, historically grounded — never costuming the UI
- *Hades*: Bold and stylized, but internally consistent — every element speaks the same visual language
- *Apotheon*: Radical commitment to a single historical art form (pottery), but adapted for gameplay clarity
- *Frostpunk*: Proves that UI atmosphere comes from *tone* (color temperature, edge treatment, material quality), not from literal depictions of the setting

**The Oracle principle**: The UI should feel like it was *made by* an ancient Greek craftsman, not like it is *depicting* ancient Greece. The difference is between a bronze tablet (the UI IS the artifact) and a picture of a bronze tablet (the UI is a frame around a picture).

### 5.2 Texture and Material Usage

**Do**:
- Subtle noise/grain overlay on panel backgrounds (2-4% opacity) to break up flat CSS gradients
- Very slight warm-to-cool gradient on major surfaces (simulates bronze patina)
- Hairline inner glow on panels (simulates the slight catch of light on a raised metal edge)
- Slight vignette darkening at panel corners (natural, not dramatic)

**Do not**:
- Full photographic parchment textures as backgrounds (overwhelming, reduces text legibility, screams "2005 Flash game")
- Stone-crack overlays or marble veining (too literal, too noisy)
- Leather or stitching textures (medieval, not Greek)
- Wood grain (medieval tavern, not Delphi sanctuary)
- Torn/burned parchment edges (too dramatic, too literal)

**Material reference table**:

| Material | How to evoke it in CSS/UI | Where to use it |
|---|---|---|
| Bronze | Warm gold-to-dim gradient, subtle noise, sharp 1px highlight border | Panel borders, button faces, active states |
| Parchment | Warm cream color (#f5e6c8), very slight yellow noise | Text on dark bg, narrative overlays |
| Stone | Dark cool-warm gradient, no texture, matte feel | Background, deepest layer |
| Terracotta | Warm red-orange accent (#c0513a family) | Warning states, fired/consumed indicators |
| Olive wood | Not literally — just the warm brown tones of `--surface` family | Panel interiors, card backgrounds |

### 5.3 Borders and Ornament

**The ornament budget**: One ornamental element per screen. Not per panel — per screen.

Ornament in The Oracle should follow the Greek **meander** (key pattern) tradition: geometric, repeatable, and structural. Not floral, not figurative, not curvilinear.

**Permitted ornamental elements**:
- A thin meander/key-pattern border on the top bar or overlay headers
- Small laurel-leaf or olive-branch corner marks on major panel headers
- A single column-capital or palmette motif as a section divider in long scroll views
- Subtle wave/scroll line as a horizontal rule

**Budget enforcement**: If you are adding a second decorative element to a screen, the first one must be removed or the two must be consolidated into one. This prevents ornament creep.

**Anti-patterns**:
- Column/pillar graphics flanking every panel (too literal, wastes space)
- Greek key border on EVERY panel (ornament fatigue — it becomes wallpaper)
- Figurative Greek art (warriors, gods, vases) as UI decoration (save for game content, not chrome)
- Ornate corner scrollwork (baroque/medieval, not Greek)
- Gold filigree borders on every card (too "mobile RPG gacha")

### 5.4 Motion and Transitions

Atmosphere lives in motion quality, not quantity.

**Principle**: Transitions should feel like stone settling into place or bronze catching the light — heavy, deliberate, warm. Not bouncy, not snappy, not elastic.

| Element | Duration | Easing | Notes |
|---|---|---|---|
| Panel open | 250ms | `ease-out` | Slide up + fade, slight scale from 0.98 |
| Panel close | 200ms | `ease-in` | Faster than open (disappearing is less important) |
| Tooltip appear | 150ms | `ease-out` | Quick — tooltips must not feel sluggish |
| Button hover | 120ms | `linear` | Border glow + color shift |
| State change (chip, badge) | 300ms | `ease-in-out` | Color transitions feel like temperature change |
| Number increment | 400ms | `ease-out` | Roll/count animation for resource changes |
| Notification enter | 350ms | `ease-out` | Slide from edge + fade |

**Anti-patterns**:
- Spring/bounce easing (too playful for the tone)
- Transitions longer than 500ms (feels sluggish in a strategy game)
- Parallax or depth effects on UI panels (distracting, breaks the tablet metaphor)
- Particle effects on UI elements (sacred fire particles belong in the game world, not the HUD)

### 5.5 The Atmosphere Checklist

A panel achieves atmospheric integration when:

- [ ] It could exist as a real artifact from the Archaic-to-Classical Greek world (conceptually, not literally)
- [ ] Removing all text and icons, the colors and proportions alone suggest "ancient, warm, serious"
- [ ] No element in the panel could be mistaken for a modern web application
- [ ] No element in the panel could be mistaken for a medieval European aesthetic
- [ ] The panel is comfortable to read for 30+ minutes without eye strain
- [ ] A screenshot of the panel, shown to someone unfamiliar with the game, would prompt them to say "that looks Greek" rather than "that looks old-fashioned" or "that looks like fantasy"

---

## 6. Cross-Cutting Standards

### 6.1 Information Density

Strategy games are information-dense by nature. The Oracle manages prophecies, resources, factions, buildings, personnel, and narrative events simultaneously.

**Principle** (Tufte): "Clutter and confusion are not attributes of information, they are failures of design."

**Density tiers**:

| Tier | Max items visible | Example |
|---|---|---|
| Glance (HUD) | 5-7 data points | Top bar: gold, piety, prestige, day, month, speed |
| Scan (panel) | 12-18 data points | Building detail: name, type, condition, staff, output, upkeep |
| Study (overlay) | 25-40 data points | Consultation: tiles, risk, value, prophecy text, consequence preview |
| Reference (full screen) | Unlimited with scroll | Faction compendium, building encyclopedia |

Each tier uses progressively smaller type, denser layout, and more abbreviation. But no tier drops below 10px text or 36px interactive targets.

### 6.2 Spacing System

Use an 8px base grid (the most common in professional design systems, aligning with Müller-Brockmann's grid principles):

| Token | Value | Usage |
|---|---|---|
| `--space-xs` | 4px | Inner icon padding, tight inline gaps |
| `--space-sm` | 8px | Chip padding, label-to-value gap, list item separation |
| `--space-md` | 16px | Panel body padding, section separation |
| `--space-lg` | 24px | Major section breaks, panel-to-panel margin |
| `--space-xl` | 32px | Screen-level layout gutters |

### 6.3 Z-Index Layer Map

| Layer | Z-Index | Contents |
|---|---|---|
| Game canvas | 0 | Phaser scene |
| HUD overlay | 1 | Top bar, sidebar, bottom bar |
| Floating panels | 2 | Building detail, faction card |
| Modal overlay | 3 | Consultation, event, dialogue |
| Tooltip | 4 | Hover tooltips |
| Notification | 5 | Toast messages, alerts |
| Debug | 100 | Debug panel (dev only) |

### 6.4 Responsive Considerations

The Oracle is desktop-first, but panels must remain functional at:
- **Minimum**: 1280x720 (common laptop resolution)
- **Target**: 1920x1080 (standard desktop)
- **Wide**: 2560x1440 (large monitor — panels should not stretch, use max-width)

At 1280x720, sidebar panels should collapse to icon-only or become toggleable overlays. The top bar should compress resource pills into a scrollable row.

---

## 7. Audit of Current Implementation

The existing `styles.css` is a strong foundation. Key improvements to consider:

| Issue | Current | Recommended |
|---|---|---|
| Minimum font size | `0.58rem` (~9.3px) labels | `0.69rem` (11px) — raise to meet readability floor |
| Missing semantic colors | No amber or purple tokens | Add `--amber: #d4a032` and `--purple: #8b6bb0` |
| No spacing tokens | Ad-hoc padding/gap values | Add `--space-{xs,sm,md,lg,xl}` to `:root` |
| No type scale tokens | Inline `font-size` everywhere | Add `--text-{display,heading,subheading,body,body-sm,label,caption}` |
| Number font | Same Georgia for text and numbers | Add `font-family: system-ui` for `.resource-pill-value` and stat numbers |
| Button system | Implicit, varies by context | Formalize `.btn-primary`, `.btn-secondary`, `.btn-destructive`, `.btn-disabled` |
| Chip system | Not yet formalized | Add `.chip` with `.chip--positive`, `.chip--negative`, `.chip--warning`, `.chip--neutral`, `.chip--sacred` |

---

## 8. Reference Index

### Games Studied

| Game | Key Takeaway for The Oracle |
|---|---|
| Old World | Warm serif typography, restrained classical palette, transparency as a design value |
| Crusader Kings III | Information-dense panels with parchment warmth; border/hover patterns for card containers |
| Civilization VI | Icon system consistency; clean data hierarchy; type-scale discipline across many panel types |
| Hades | Bold color coding per divine domain; proves Greek mythological UI can be vibrant without being kitschy |
| Assassin's Creed Odyssey | Color in ancient Greek architecture was functional (contrast/legibility), not decorative — apply same logic to UI |
| Total War: Rome II | Faction-specific UI tinting; historical scroll/engraving chrome instead of holographic panels |
| Apotheon | Radical proof that one consistent Greek art vocabulary (pottery) can drive an entire interface language |
| Pharaoh: A New Era | Clean categorical organization of dense building menus; warm gold-blue as Egyptian but adjacent to Greek bronze |
| Frostpunk | Atmospheric UI comes from tone (color temperature, edge quality, motion) not literal depiction |
| Slay the Spire | Three-tier stat visibility model; card animation as comprehension aid; managing density through progressive disclosure |

### Design Principles Applied

| Expert | Principle Applied |
|---|---|
| Dieter Rams | "As little design as possible" — one ornament per screen, every element must earn its place |
| Josef Müller-Brockmann | 8px grid system, maximum 3-4 hierarchy levels per panel, left-aligned text |
| Edward Tufte | Data-ink ratio — theme the chrome but never obscure the data; color as data tool, not decoration |
| Ellen Lupton | Hierarchy through contrast (size, weight, color, spacing), not decoration; type serves the content |
| Jan Tschichold | Classical proportions; Golden-section spacing; restraint in ornament; type that breathes |
| Massimo Vignelli | Limited palette discipline (5 semantic + 3 accent colors); typographic consistency across all contexts |
