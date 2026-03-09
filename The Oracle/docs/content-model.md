# The Oracle Content Model

Content is referenced by stable IDs from simulation code.

## Primary content sets

- `buildingDefs`: placement rules, role slots, upkeep, and render metadata
- `resourceDefs`: display names, caps, and economic weight
- `factionDefs`: political posture, agendas, and trade affinity
- `omenDefs`: report templates and tag extraction rules
- `wordTileDefs`: visible prophecy tiles with hidden semantic tags
- `traitDefs`: Pythia modifiers
- `advisorDefs`: advisor personas and warning domains
- `scenarioDefs`: starting state and scripted pacing
- `politicalEventDefs`: monthly world map drivers and consequence outputs

## Semantic prophecy tags

Every prophecy tile carries hidden tags across these axes:

- target
- action
- polarity
- ambiguity
- time horizon
- domain

These tags power clarity/value scoring and delayed consequence resolution. Visible tile text stays literary while scoring remains deterministic.
