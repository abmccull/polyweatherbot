# Blender MCP Integration

This project now treats Blender as the bridge between 3D proxy generation and final 2D/pixel-art source renders.

## Why Blender MCP

- Meshy gives us fast proxy geometry and reusable base forms.
- Blender gives us consistent camera, scale, materials, shadow direction, and multi-pass renders.
- MCP lets us script that setup repeatedly instead of hand-staging every asset.

## Global Setup Targets

Use Blender MCP as a **global** tool, not a repo-local one:

- Codex global MCP config
- Claude Desktop global MCP config
- optional home-level `.mcp.json` fallback for clients that read user-root MCP config

## Blender Stage In The Pipeline

1. Generate or download a 3D proxy from Meshy.
2. Import it into Blender.
3. Normalize origin, rotation, and scale.
4. Apply sanctuary-safe material presets: limestone, marble, terracotta, aged bronze.
5. Snap to the fixed 3/4 isometric camera.
6. Render clean extraction-friendly PNG passes.
7. Convert or clean up those renders for final game usage.

## Era Variants

Building and major precinct assets should support era-aware variants from the same base setup:

- `archaic`
- `classical`
- `hellenistic`
- `romanized`

The goal is not to fully replace buildings each era. Instead, use one shared footprint and evolve:

- masonry refinement
- ornament density
- bronze details
- stair treatment
- roof treatment
- votive clutter and monument dressing

## First Recommended Blender Automation Tasks

- import and render `castalian_spring`
- import and render `eternal_flame_brazier`
- import and render `temple_of_apollo`
- produce `archaic` and `classical` variants from the same normalized scene file

## Output Conventions

- source models: `output/art/meshy/downloads/`
- Blender renders: `output/art/renders/`
- pixel-converted outputs: `output/art/pixel/`

## Tracked Template Assets

- Blender bootstrap script: `scripts/art/blender/oracle-scene-template.py`
- Meshy import/render helper: `scripts/art/blender/render-meshy-asset.py`
- Manual blockout helper: `scripts/art/blender/manual-blockout-asset.py`
- Saved reusable template: `output/art/blender-templates/oracle_isometric_template.blend`
- Template metadata: `output/art/blender-templates/oracle_isometric_template.json`
- Validation render: `output/art/renders/template/oracle_template_probe.png`

Run the bootstrap script inside Blender MCP when you need to recreate the baseline scene. It resets the scene, rebuilds the fixed orthographic camera and sun rig, restores the sanctuary-safe material presets, writes a transparent PNG validation render, and saves a clean `.blend` template with the reference shrine hidden from final renders.

Use the Meshy import helper after a GLB download is complete and the template scene is open. It imports the GLB, clears any prior imported asset collection, normalizes footprint and ground placement, renders a transparent PNG, and saves a per-asset `.blend` plus metadata JSON for later cleanup passes.

Use the manual blockout helper when Meshy credits are exhausted or external Blender add-on generators are unavailable. It creates a stylized sanctuary-safe proxy directly from primitives inside the saved Oracle template, renders a transparent PNG, and saves a matching `.blend` plus metadata JSON. Run one asset per Blender MCP call; batched renders were prone to timing out the bridge even when the files eventually finished writing, and a few single-asset runs also wrote their files after the tool-layer timeout.

Current manual profiles on disk include:

- the full tracked building family in `art-library/generated/asset-manifest.json`, except `castalian_spring`, `eternal_flame_brazier`, and `temple_of_apollo`, which already exist as imported Meshy-first scenes
- manual render count currently on disk: `32`
- render output directory: `output/art/renders/manual/`

## Review Rules

- keep camera and sun direction fixed across a family
- verify footprint readability before adding detail
- avoid drifting into oversized palace or temple-complex silhouettes
- preserve the Delphi-inspired sacred-precinct tone from `docs/art-style-bible.md`
