# The Oracle — UI Foundation Refactoring (Wave 1) - COMPLETE

## Foundation Components
- [x] `GameOverlay.tsx` — shared overlay shell
- [x] `GameDispatchContext.tsx` — dispatch context to eliminate prop drilling
- [x] `OverlayTriggerStrip.tsx` — 40px icon strip replacing right sidebar

## Overlay Panel Extractions
- [x] `OracleOverlayPanel.tsx` — Oracle tab content
- [x] `StoresOverlayPanel.tsx` — Stores tab content
- [x] `PriestsOverlayPanel.tsx` — Priests tab content
- [x] `WorldOverlayPanel.tsx` — World tab content
- [x] `RecordOverlayPanel.tsx` — Record tab content

## Integration
- [x] Update `app.tsx` — activeOverlay state, dispatch context, keyboard handler
- [x] Update `OracleHud.tsx` — remove RightSidebar, add trigger strip, collapsible palette
- [x] Update `styles.css` — new layout, overlay styles, z-index, typography
- [x] Update `index.ts` — exports
- [x] Delete `RightSidebar.tsx`

---

# Wave 2: Surface Orphaned Backend Systems - COMPLETE

## 2A. New Selectors
- [x] `selectPhilosopherThreats` — per-faction philosopher threats
- [x] `selectConsequenceTracker` — pending/resolved consequences
- [x] `selectCrisisChains` — active crisis chains

## 2B. New UI Panels
- [x] `PhilosopherThreatsPanel.tsx` — in World overlay
- [x] `CrisisChainsPanel.tsx` — in World overlay
- [x] `CharactersPanel.tsx` — in World overlay (spotlight + roster with relationship bars)
- [x] `RivalOraclesPanel.tsx` — in World overlay
- [x] `ConsequenceTrackerPanel.tsx` — in Oracle overlay
- [x] `CarrierDetailPanel.tsx` — in Stores overlay

## 2C. Priest Commands
- [x] `IssuePriestDecreeCommand` (calm/reform/investigate)
- [x] `DismissPriestCommand`
- [x] `EndorseBlocCommand`
- [x] Reducer handlers for all 3 commands
- [x] UI buttons in PriestsOverlayPanel

## 2D. Ghost Priest Roles
- [x] `dream_priest` and `astronomer` added to PriestRole type

## Verification
- [x] TypeScript compiles clean
- [x] All 71 tests pass

---

# Wave 3: R0 Foundation Completion - COMPLETE
(collapsed for brevity — all items complete)

---

# Wave 4: R1 Long Campaign - COMPLETE
(collapsed for brevity — all items complete)

---

# Wave 5: R2-R4 - COMPLETE
(collapsed for brevity — 86 tests, all passing)

---

# 10/10 UI Redesign

## Wave 0: Design System Foundation — IN PROGRESS
- [ ] Lane A: CSS tokens + font-size migration (styles.css)
- [ ] Lane B: Inline fontSize elimination (17 TSX files)
- [ ] Verification: typecheck + test + build pass

## Wave 1: Navigation Architecture
- [ ] Top bar redesign (remove title, move speed controls)
- [ ] Escape menu (new component)
- [ ] Bottom toolbar (new component, replaces left sidebar)
- [ ] Wire into OracleHud + app.tsx keyboard handler
- [ ] Verification

## Wave 2: Overlay Trigger Strip Completion
- [ ] Add 3 missing overlays (espionage, legacy, lineage)
- [ ] Add kbd badges + notification dots
- [ ] Add I key for lineage
- [ ] Verification

## Wave 3: Overlay Decomposition + Stat Unification
- [ ] Oracle overlay 4-tab split
- [ ] World overlay sub-tabs
- [ ] Stat naming unification (Anchors -> Sanctity, add Shelter)
- [ ] Verification

## Wave 4: Per-Screen Polish
- [ ] Consultation: remove Save/Load, add Cancel, tile animation
- [ ] Sacred Record: side-by-side layout, search
- [ ] Event log urgency tiers
- [ ] Minimap toggle (M key)
- [ ] Verification

## Wave 5: Meta-Progression Polish
- [ ] Legacy: ceremony animation, Start New Run
- [ ] Lineage: unlock grid, burden toggles, Launch Run
- [ ] Legendary: progress dots, stage gating
- [ ] Verification

## Wave 6: Icon System + CSS Cleanup
- [ ] SVG icon system (Icons.tsx)
- [ ] SharedComponents.tsx (extract duplicates)
- [ ] Icon replacement across all files
- [ ] CSS dead code removal + responsive breakpoints
- [ ] Inline style cleanup
- [ ] Verification

## Wave 7: Accessibility + Help + Final QA
- [ ] Accessibility pass
- [ ] Help overlay (? key)
- [ ] Performance check
- [ ] Final QA walkthrough
- [ ] All 86 tests still pass
