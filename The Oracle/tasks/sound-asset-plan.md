# The Oracle — Sound Asset Acquisition Plan

## API Reality Check

| Source | Has API? | License | Notes |
|--------|----------|---------|-------|
| Pixabay | NO (images/videos only) | Royalty-free, no attribution | Manual download, MP3, 120k+ SFX |
| Freesound.org | YES (full REST API) | CC0 / CC-BY | API key required, .wav/.ogg/.mp3 |
| ElevenLabs | YES (AI generation) | Commercial use OK | $5/mo, generate custom SFX from text prompts |

**Recommendation**: Pixabay for hand-picked hero sounds + Freesound API script for bulk discovery.

---

## Sound Asset Manifest

### Tier 1: Core Loop (ship first — 18 sounds)

| ID | Sound | Pixabay Search Term | Freesound Query | Duration |
|----|-------|--------------------|--------------------|----------|
| `ui-click` | Button click | "wooden button click" | "click wood UI" | <0.3s |
| `ui-confirm` | Confirm action | "confirm chime" | "confirm positive" | <0.5s |
| `ui-error` | Error/denied | "error buzz" | "error negative" | <0.5s |
| `ui-hover` | Hover hint | "soft tick" | "tick subtle" | <0.2s |
| `build-place` | Generic building placed | "stone place impact" | "stone drop thud" | 0.5-1s |
| `build-road` | Road segment placed | "gravel footstep" | "gravel crunch" | 0.3-0.5s |
| `build-locked` | Locked building attempt | "lock rattle" | "lock denied" | 0.3s |
| `day-advance` | New day bell | "temple bell single" | "bell single ring" | 1-2s |
| `speed-change` | Game speed toggle | "clock tick" | "clock mechanism" | <0.3s |
| `pause` | Game paused | "pause swoosh" | "swoosh down" | 0.3s |
| `consult-start` | Consultation begins | "mystical reveal" | "mystical shimmer" | 1-2s |
| `consult-tile` | Prophecy tile placed | "stone tablet click" | "stone click ceramic" | 0.3-0.5s |
| `consult-deliver` | Prophecy delivered | "dramatic reveal gong" | "gong deep reverb" | 2-3s |
| `trade-buy` | Trade purchase | "coin drop pouch" | "coins money bag" | 0.5-1s |
| `repair` | Building repaired | "hammer stone repair" | "hammer chisel stone" | 0.5-1s |
| `priest-assign` | Priest assigned | "chant single note" | "monk chant short" | 0.5-1s |
| `pythia-rest` | Pythia rests | "gentle exhale calm" | "breath calm sigh" | 1s |
| `pythia-purify` | Pythia purified | "water splash ritual" | "water pour splash" | 1-1.5s |

### Tier 2: Building-Specific (20 sounds)

| ID | Sound | Search Terms | Duration |
|----|-------|-------------|----------|
| `build-quarters` | Quarters placed | "wooden door close" | 0.5-1s |
| `build-storehouse` | Storehouse placed | "heavy chest close" | 0.5-1s |
| `build-spring` | Spring placed | "water spring bubbling" | 1s |
| `build-sanctum` | Sanctum placed | "deep stone door temple" | 1-2s |
| `build-brazier` | Brazier placed | "fire ignite whoosh" | 0.5-1s |
| `build-altar` | Altar placed | "stone altar thud" | 0.5-1s |
| `build-pen` | Animal pen placed | "wooden fence" | 0.5s |
| `build-granary` | Granary placed | "grain pour" | 0.5-1s |
| `build-kitchen` | Kitchen placed | "pot clang" | 0.3-0.5s |
| `build-olive-press` | Olive press placed | "millstone grind" | 0.5-1s |
| `build-incense-store` | Incense store placed | "wooden crate" | 0.5s |
| `build-agora` | Market placed | "crowd murmur short" | 1s |
| `build-xenon` | Xenon placed | "door creak open" | 0.5-1s |
| `build-grain-field` | Grain field placed | "wheat rustle" | 0.5-1s |
| `build-olive-grove` | Olive grove placed | "tree leaves rustle" | 0.5-1s |
| `build-incense-workshop` | Workshop placed | "mortar pestle" | 0.5-1s |
| `build-reed-bed` | Reed bed placed | "reeds water" | 0.5-1s |
| `build-scriptorium` | Scriptorium placed | "quill writing parchment" | 0.5-1s |
| `build-library` | Library placed | "book heavy place" | 0.5-1s |
| `build-degrade-warn` | Building degradation | "crumble crack" | 0.5s |

### Tier 3: Ambient Loops (8 loops)

| ID | Sound | Search Terms | Duration |
|----|-------|-------------|----------|
| `amb-spring` | Spring season | "birds spring morning nature" | 30-60s loop |
| `amb-summer` | Summer season | "cicadas summer heat" | 30-60s loop |
| `amb-autumn` | Autumn season | "autumn wind leaves falling" | 30-60s loop |
| `amb-winter` | Winter season | "cold wind winter desolate" | 30-60s loop |
| `amb-temple` | Base precinct | "temple ambient drone" | 30-60s loop |
| `amb-fire` | Brazier active | "fire crackling campfire" | 15-30s loop |
| `amb-water` | Spring active | "water stream gentle" | 15-30s loop |
| `amb-market` | Agora active | "marketplace distant crowd" | 15-30s loop |

### Tier 4: Events & Feedback (12 sounds)

| ID | Sound | Search Terms | Duration |
|----|-------|-------------|----------|
| `event-crisis` | Crisis event | "war horn distant" | 1-2s |
| `event-consequence` | Prophecy resolved | "fate reveal" | 1s |
| `event-vindicated` | Prophecy correct | "triumph fanfare short" | 1-2s |
| `event-broken` | Prophecy failed | "ominous thud failure" | 1s |
| `event-tier-up` | Reputation tier gained | "achievement ascend chime" | 2-3s |
| `event-envoy` | Envoy approaching | "horse hooves approach" | 1-2s |
| `event-rival` | Rival oracle action | "whisper sinister" | 1s |
| `advisor-warn` | Advisor warning | "subtle alert tone" | 0.5s |
| `advisor-critical` | Advisor critical | "urgent warning bell" | 0.5-1s |
| `scroll-bonus` | Scroll clarity bonus | "page flip shimmer" | 0.5-1s |
| `season-change` | Season transition | "wind transition swoosh" | 1-2s |
| `save-confirm` | Game saved | "soft chime confirm" | 0.5s |

### Tier 5: Music (3 tracks, lowest priority)

| ID | Sound | Search Terms | Duration |
|----|-------|-------------|----------|
| `music-menu` | Main menu | "ancient greek lyre calm" | 2-3min loop |
| `music-play` | Gameplay | "ambient mediterranean ancient" | 3-5min loop |
| `music-consult` | Consultation | "mystical oracle trance" | 2-3min loop |

---

## Total: ~61 sound assets

| Tier | Count | Priority |
|------|-------|----------|
| 1: Core Loop | 18 | Ship first |
| 2: Building-Specific | 20 | Second pass |
| 3: Ambient Loops | 8 | Third pass |
| 4: Events & Feedback | 12 | Fourth pass |
| 5: Music | 3 | Polish |

---

## Acquisition Script Plan (Freesound API)

Since Pixabay has no audio API, use Freesound.org for programmatic discovery:

```
GET https://freesound.org/apiv2/search/text/
  ?query={search_term}
  &filter=duration:[0 TO 3] license:"Creative Commons 0"
  &fields=id,name,tags,duration,previews,download
  &sort=rating_desc
  &page_size=5
  &token={API_KEY}
```

### Script: `scripts/audio/fetch-sounds.mjs`

1. Read the manifest above as JSON
2. For each sound ID, query Freesound with the search terms
3. Download the top-rated CC0 preview (.mp3) to `apps/web/public/audio/{id}.mp3`
4. Generate an `audio-manifest.json` mapping IDs to file paths + metadata
5. Log gaps (no results) for manual Pixabay download

### Manual Pixabay Workflow (for gaps)

1. Go to https://pixabay.com/sound-effects/search/{term}
2. Preview candidates, download MP3
3. Rename to `{id}.mp3` and place in `apps/web/public/audio/`
4. Update `audio-manifest.json`

---

## Audio System Integration (implementation sketch)

### File structure
```
apps/web/public/audio/
  ui-click.mp3
  build-place.mp3
  amb-spring.mp3
  ...
  audio-manifest.json

packages/core/src/audio/
  audioEvents.ts        -- maps GameEvent/GameCommand -> sound ID

apps/web/src/audio/
  AudioManager.ts       -- Howler.js or Web Audio wrapper
  useGameAudio.ts       -- React hook binding state changes to sounds
```

### Event-to-Sound Mapping (`audioEvents.ts`)
```typescript
export const EVENT_SOUNDS: Record<string, string> = {
  BuildingPlaced: "build-place",
  DayAdvanced: "day-advance",
  ConsultationStarted: "consult-start",
  ProphecyDelivered: "consult-deliver",
  TradePurchased: "trade-buy",
  ConsequenceResolved: "event-consequence",
  CredibilityChanged: "event-vindicated", // conditional on delta
};

export const COMMAND_SOUNDS: Record<string, string> = {
  SetGameSpeedCommand: "speed-change",
  RestPythiaCommand: "pythia-rest",
  PurifyPythiaCommand: "pythia-purify",
  RepairBuildingCommand: "repair",
  AssignPriestCommand: "priest-assign",
};

export const BUILDING_PLACE_SOUNDS: Record<string, string> = {
  sacred_way: "build-road",
  castalian_spring: "build-spring",
  inner_sanctum: "build-sanctum",
  // ... one per building
};

export const SEASON_AMBIENT: Record<string, string> = {
  Spring: "amb-spring",
  Summer: "amb-summer",
  Autumn: "amb-autumn",
  Winter: "amb-winter",
};
```

---

## Implementation Order

```
Step 1: Get Freesound API key (free signup)
Step 2: Build fetch-sounds.mjs script
Step 3: Run script -> download Tier 1 sounds
Step 4: Manual Pixabay gap-fill for anything missing
Step 5: Build AudioManager + useGameAudio hook
Step 6: Wire into app.tsx runtime event listener
Step 7: Add volume/mute controls to UI
Step 8: Iterate through Tiers 2-5
```
