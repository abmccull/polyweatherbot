# The Oracle Meshy-Assisted Art Production Plan

## Purpose

This document defines how `The Oracle` should use the existing local skills plus Meshy AI to create a scalable visual asset pipeline for buildings, characters, terrain, flora, fauna, and visual set dressing.

The goal is **not** to replace final art direction with one-click generation. The goal is to build a production pipeline that:

- creates strong concept anchors quickly
- preserves consistency across variants and camera angles
- generates reusable 3D proxy assets where helpful
- converts those proxies into readable 2D game assets
- keeps every asset testable inside the actual game camera and UI

This plan treats Meshy as a **3D reference and proxy generator** that complements `imagegen`, not as the final source of truth for production-ready pixel art.

## Recommended Tool Stack

| Tool / Skill | Primary Use in This Pipeline |
|---|---|
| `imagegen` | Style exploration, concept anchors, faction variants, portraits, ornaments, item sheets, and character turnarounds |
| Meshy API | 3D proxy generation for buildings, props, landmarks, humanoid NPC bases, retexturing, and fixed-camera render sources |
| `proper-pixel-art` | Converting generated or rendered source images into lower-resolution, cleanup-friendly pixel-art outputs |
| `develop-web-game` | Asset preview scenes, import helpers, fixed-camera render checks, palette/readability validation surfaces |
| `playwright` | Automated visual regression on asset preview scenes and in-context scene placement |
| `playwright-interactive` | Fast debugging of anchoring, z-sort, tile occupancy, animation alignment, and scene readability |
| `screenshot` | Headed artifact capture when canvas/full-page automation is not enough |
| `spreadsheet` | Asset manifest, prompt tracking, batch status, QA notes, palette and footprint bookkeeping |

## Production Principles

1. Lock style before scale.
2. Use 3D for consistency and 2D for final look.
3. Validate every asset at gameplay zoom.
4. Batch work by family.
5. Keep prompts, tasks, and outputs auditable.
6. Prefer reusable kits over isolated hero assets.

## Best Use of Meshy

Use Meshy primarily for:

- buildings and monuments
- large environment props
- humanoid NPC bases
- variant generation for upgrades, damage states, and patron-specific skins

Do not treat Meshy as the final step for:

- seamless terrain autotiles
- final strict pixel-art sprite sheets without cleanup
- non-humanoid creature animation

## Best Use of `proper-pixel-art`

Use `proper-pixel-art` after concept art or Meshy-based renders have already been shaped toward the desired silhouette and camera angle.

It is best used for:

- converting rendered source images into lower-resolution pixel art
- preserving transparent outputs for sprite work
- producing cleanup-friendly bases for buildings, props, portraits, and static poses

It should not be treated as a complete substitute for human cleanup on final production assets.

## Asset Family Order

1. current precinct slice replacement kit
2. core building roster expansion kit
3. first walker role kit
4. terrain, flora, and landmark prop kit
5. portrait and consultation ornament kit

## Required Supporting Files

- `.env.example`
- `.env`
- `docs/asset-pipeline-setup.md`
- `scripts/art/meshy.mjs`
- `scripts/art/pixel-art.mjs`

## First Practical Goal

Replace the live slice graybox assets for:

- Sacred Way props
- Castalian Spring
- Eternal Flame Brazier
- Priest Quarters
- Storehouse
- Inner Sanctum
- first priest / pilgrim / envoy silhouettes

That is enough to prove the pipeline before opening larger art batches.
