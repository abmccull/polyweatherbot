# The Oracle Architecture

The Oracle is structured as a browser-first game workspace with a desktop shell layered on afterward.

## Runtime split

- `apps/web`: primary development target. Hosts Phaser world rendering, React UI overlays, debug hooks, and the deterministic simulation runtime.
- `apps/desktop`: Electron wrapper around the web bundle with native save paths and future Steam integration ports.
- `packages/core`: pure simulation state, commands, reducers, selectors, event scheduling, and deterministic time stepping.
- `packages/content`: typed data definitions for buildings, resources, factions, omens, questions, and political events.
- `packages/persistence`: snapshot schema and save repositories for web and desktop.
- `packages/ui`: React presentation components and consultation builder UI.
- `packages/testkit`: helpers for deterministic rendering checks and browser automation.

## Core loop

Simulation advances in integer ticks. Each tick runs:

1. world clock progression
2. command queue processing
3. walker AI target selection
4. path movement
5. building processing
6. resource transfer jobs
7. needs and degradation
8. event scheduling and consequence resolution
9. autosave signal emission

## Persistence

`GameSnapshot` is the single serialized save contract. Browser builds persist snapshots into IndexedDB. Desktop builds persist the same snapshot payload plus recent domain events into SQLite.
