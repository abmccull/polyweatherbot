# The Oracle Balance Workbook

This folder contains the `SHEET-1` deliverable from [docs/parallel-implementation-plan.md](/Users/tsc-001/station_sniper/The Oracle/docs/parallel-implementation-plan.md).

Files:
- `oracle-balance.xlsx`: workbook with tabs for buildings, resource rates, envoys, Pythia pacing, reputation thresholds, and `Rising Oracle` pacing.
- `oracle-balance.tables.json`: machine-readable export of the same tables for future content seeding or scripted import.

Grounding:
- Extracted from the current repo: `packages/content/src/data.ts`, `packages/core/src/state/initialState.ts`, `packages/core/src/simulation/updateDay.ts`, and `packages/core/src/simulation/events.ts`.
- Proposed, not yet implemented in code: reputation tiers, treasury dedication thresholds, crisis-chain gates, and most `Rising Oracle` pacing targets.

Important caveats:
- Generic building upkeep is not applied uniformly in the current simulation. The clearest example is `priest_quarters`: it has grain upkeep in content data, but `updateDay.ts` does not currently consume it.
- `inner_sanctum` uses hardcoded idle/open demand instead of only the content-def upkeep values.
- Consultation-open sink rates only matter if the simulation is allowed to keep ticking while a consultation remains open.
- The workbook includes target rates for planned DVG-1/DVG-3 content so implementation agents have a single reference point, but those rows should be treated as balancing proposals rather than established canon.
