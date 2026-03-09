"""
Import a Meshy GLB into the already-open Oracle template scene, normalize it,
and render a transparent PNG.

Expected globals before execution:
- ASSET_ID: stable asset id
- GLB_PATH: absolute path to the downloaded GLB
- FOOTPRINT_METERS: target footprint size in template units

Optional globals:
- OUTPUT_RENDER_PATH
- OUTPUT_BLEND_PATH
- OUTPUT_METADATA_PATH
- ROTATE_Z_DEGREES

This script assumes the Oracle template scene is already loaded.
"""

from __future__ import annotations

import json
import math
import os

import bpy
from mathutils import Vector


REPO_ROOT = "/Users/tsc-001/station_sniper/The Oracle"
ASSET_COLLECTION_PREFIX = "ASSET_"
TEMPLATE_COLLECTION_NAMES = {"ORACLE_RENDER", "ORACLE_GUIDES", "ORACLE_REFERENCE"}


def require_global(name: str):
    if name not in globals():
        raise RuntimeError(f"Missing required global: {name}")
    return globals()[name]


ASSET_ID = require_global("ASSET_ID")
GLB_PATH = require_global("GLB_PATH")
FOOTPRINT_METERS = float(require_global("FOOTPRINT_METERS"))

ROTATE_Z_DEGREES = float(globals().get("ROTATE_Z_DEGREES", 0.0))
OUTPUT_RENDER_PATH = globals().get(
    "OUTPUT_RENDER_PATH",
    os.path.join(REPO_ROOT, "output", "art", "renders", "meshy", f"{ASSET_ID}.png"),
)
OUTPUT_BLEND_PATH = globals().get(
    "OUTPUT_BLEND_PATH",
    os.path.join(REPO_ROOT, "output", "art", "blender-scenes", f"{ASSET_ID}.blend"),
)
OUTPUT_METADATA_PATH = globals().get(
    "OUTPUT_METADATA_PATH",
    os.path.join(REPO_ROOT, "output", "art", "blender-scenes", f"{ASSET_ID}.json"),
)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def remove_collection_tree(collection: bpy.types.Collection) -> None:
    for child in list(collection.children):
        remove_collection_tree(child)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def cleanup_previous_imports() -> None:
    for collection in list(bpy.data.collections):
        if collection.name.startswith(ASSET_COLLECTION_PREFIX):
            remove_collection_tree(collection)


def imported_objects(before_names: set[str]) -> list[bpy.types.Object]:
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def create_asset_collection() -> bpy.types.Collection:
    collection = bpy.data.collections.new(f"{ASSET_COLLECTION_PREFIX}{ASSET_ID}")
    bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(
    objects: list[bpy.types.Object], collection: bpy.types.Collection
) -> None:
    for obj in objects:
        for parent in list(obj.users_collection):
            parent.objects.unlink(obj)
        collection.objects.link(obj)


def mesh_objects(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    return [obj for obj in objects if obj.type == "MESH"]


def world_bbox(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world_corner.x)
            mins.y = min(mins.y, world_corner.y)
            mins.z = min(mins.z, world_corner.z)
            maxs.x = max(maxs.x, world_corner.x)
            maxs.y = max(maxs.y, world_corner.y)
            maxs.z = max(maxs.z, world_corner.z)
    return mins, maxs


def translate_objects(objects: list[bpy.types.Object], offset: Vector) -> None:
    for obj in objects:
        obj.location += offset


def set_uniform_scale(objects: list[bpy.types.Object], factor: float) -> None:
    for obj in objects:
        obj.scale *= factor
    bpy.context.view_layer.update()


def rotate_objects_z(objects: list[bpy.types.Object], radians: float) -> None:
    for obj in objects:
        obj.rotation_euler.z += radians
    bpy.context.view_layer.update()


def normalize_to_origin(objects: list[bpy.types.Object]) -> dict[str, float]:
    mins, maxs = world_bbox(objects)
    center_xy = Vector(((mins.x + maxs.x) * 0.5, (mins.y + maxs.y) * 0.5, 0.0))
    translate_objects(objects, Vector((-center_xy.x, -center_xy.y, -mins.z)))
    bpy.context.view_layer.update()

    mins, maxs = world_bbox(objects)
    width = maxs.x - mins.x
    depth = maxs.y - mins.y
    height = maxs.z - mins.z
    return {"width": width, "depth": depth, "height": height}


def scale_to_footprint(objects: list[bpy.types.Object]) -> dict[str, float]:
    dims = normalize_to_origin(objects)
    max_span = max(dims["width"], dims["depth"], 0.001)
    target_span = FOOTPRINT_METERS * 0.82
    factor = target_span / max_span
    set_uniform_scale(objects, factor)
    dims = normalize_to_origin(objects)
    dims["scale_factor"] = factor
    return dims


def save_metadata(dimensions: dict[str, float]) -> None:
    metadata = {
        "asset_id": ASSET_ID,
        "glb_path": GLB_PATH,
        "render_path": OUTPUT_RENDER_PATH,
        "blend_path": OUTPUT_BLEND_PATH,
        "footprint_meters": FOOTPRINT_METERS,
        "rotate_z_degrees": ROTATE_Z_DEGREES,
        "dimensions": dimensions,
    }
    ensure_parent(OUTPUT_METADATA_PATH)
    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def frame_camera(dimensions: dict[str, float]) -> None:
    camera = bpy.data.objects.get("ORACLE_IsometricCamera")
    target = bpy.data.objects.get("ORACLE_AssetOrigin")
    if camera is None or target is None:
        return

    target.location = Vector((0.0, 0.0, max(dimensions["height"] * 0.42, 0.28)))
    camera.data.ortho_scale = max(
        FOOTPRINT_METERS * 1.6,
        max(dimensions["width"], dimensions["depth"]) * 1.4,
        dimensions["height"] * 2.2,
    )


def main() -> dict[str, object]:
    if not os.path.exists(GLB_PATH):
        raise FileNotFoundError(GLB_PATH)

    scene = bpy.context.scene
    if scene.name != "OracleAssetTemplate":
        raise RuntimeError(
            f"Expected OracleAssetTemplate scene to be open, found {scene.name!r}"
        )

    cleanup_previous_imports()

    before = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.gltf(filepath=GLB_PATH)
    imported = imported_objects(before)
    meshes = mesh_objects(imported)
    if not meshes:
        raise RuntimeError(f"No mesh objects were imported from {GLB_PATH}")

    asset_collection = create_asset_collection()
    move_to_collection(imported, asset_collection)

    if ROTATE_Z_DEGREES:
        rotate_objects_z(imported, math.radians(ROTATE_Z_DEGREES))

    dimensions = scale_to_footprint(imported)
    frame_camera(dimensions)

    reference_collection = bpy.data.collections.get("ORACLE_REFERENCE")
    if reference_collection is not None:
        reference_collection.hide_render = True

    ensure_parent(OUTPUT_RENDER_PATH)
    ensure_parent(OUTPUT_BLEND_PATH)
    scene.render.filepath = OUTPUT_RENDER_PATH
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND_PATH, check_existing=False)
    save_metadata(dimensions)

    summary = {
        "asset_id": ASSET_ID,
        "mesh_count": len(meshes),
        "render_path": OUTPUT_RENDER_PATH,
        "blend_path": OUTPUT_BLEND_PATH,
        "metadata_path": OUTPUT_METADATA_PATH,
        "dimensions": dimensions,
    }
    print(json.dumps(summary, indent=2))
    return summary


RESULT = main()
