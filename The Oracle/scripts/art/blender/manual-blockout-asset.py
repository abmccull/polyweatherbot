"""
Create a manual Blender blockout asset inside the Oracle template scene and
render it to a transparent PNG.

Expected globals before execution:
- ASSET_ID
- PROFILE_KEY
- FOOTPRINT_METERS

Optional globals:
- OUTPUT_RENDER_PATH
- OUTPUT_BLEND_PATH
- OUTPUT_METADATA_PATH

The Oracle template scene should already be open.
"""

from __future__ import annotations

import json
import math
import os

import bpy
from mathutils import Vector


REPO_ROOT = "/Users/tsc-001/station_sniper/The Oracle"
ASSET_COLLECTION_PREFIX = "ASSET_"


def require_global(name: str):
    if name not in globals():
        raise RuntimeError(f"Missing required global: {name}")
    return globals()[name]


ASSET_ID = require_global("ASSET_ID")
PROFILE_KEY = require_global("PROFILE_KEY")
FOOTPRINT_METERS = float(require_global("FOOTPRINT_METERS"))

OUTPUT_RENDER_PATH = globals().get(
    "OUTPUT_RENDER_PATH",
    os.path.join(REPO_ROOT, "output", "art", "renders", "manual", f"{ASSET_ID}.png"),
)
OUTPUT_BLEND_PATH = globals().get(
    "OUTPUT_BLEND_PATH",
    os.path.join(REPO_ROOT, "output", "art", "blender-scenes", f"{ASSET_ID}_manual.blend"),
)
OUTPUT_METADATA_PATH = globals().get(
    "OUTPUT_METADATA_PATH",
    os.path.join(REPO_ROOT, "output", "art", "blender-scenes", f"{ASSET_ID}_manual.json"),
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


def create_asset_collection() -> bpy.types.Collection:
    collection = bpy.data.collections.new(f"{ASSET_COLLECTION_PREFIX}{ASSET_ID}")
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_only_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for parent in list(obj.users_collection):
        parent.objects.unlink(obj)
    collection.objects.link(obj)


def ensure_material(name: str, rgba: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is not None:
        shader.inputs["Base Color"].default_value = rgba
        shader.inputs["Roughness"].default_value = 0.7
    return material


def sanctuary_materials() -> dict[str, bpy.types.Material]:
    return {
        "limestone": ensure_material("ORACLE_Limestone", (0.84, 0.79, 0.70, 1.0)),
        "marble": ensure_material("ORACLE_Marble", (0.92, 0.91, 0.88, 1.0)),
        "terracotta": ensure_material("ORACLE_Terracotta", (0.64, 0.37, 0.21, 1.0)),
        "bronze": ensure_material("ORACLE_AgedBronze", (0.44, 0.31, 0.14, 1.0)),
        "wood": ensure_material("ORACLE_BarkWood", (0.34, 0.24, 0.14, 1.0)),
        "grain": ensure_material("ORACLE_GrainGold", (0.78, 0.68, 0.26, 1.0)),
        "olive": ensure_material("ORACLE_OliveLeaf", (0.3, 0.4, 0.18, 1.0)),
        "cypress": ensure_material("ORACLE_CypressDark", (0.12, 0.19, 0.12, 1.0)),
        "reed": ensure_material("ORACLE_ReedGreen", (0.5, 0.62, 0.32, 1.0)),
        "water": ensure_material("ORACLE_SpringWater", (0.34, 0.58, 0.68, 1.0)),
    }


def apply_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def create_primitive(operator, **kwargs) -> bpy.types.Object:
    before = {obj.name_full for obj in bpy.data.objects}
    operator(**kwargs)
    created = [obj for obj in bpy.data.objects if obj.name_full not in before]
    if not created:
        raise RuntimeError("Blender primitive creation did not produce a detectable object.")
    obj = created[-1]
    bpy.context.view_layer.update()
    return obj


def add_cube(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    obj = create_primitive(bpy.ops.mesh.primitive_cube_add, location=location)
    obj.name = name
    obj.scale = scale
    link_only_to_collection(obj, collection)
    apply_material(obj, material)
    return obj


def add_cylinder(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    vertices: int = 24,
) -> bpy.types.Object:
    obj = create_primitive(
        bpy.ops.mesh.primitive_cylinder_add,
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
    )
    obj.name = name
    link_only_to_collection(obj, collection)
    apply_material(obj, material)
    return obj


def add_plane(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    size: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    obj = create_primitive(bpy.ops.mesh.primitive_plane_add, location=location, size=size)
    obj.name = name
    link_only_to_collection(obj, collection)
    apply_material(obj, material)
    return obj


def add_cone(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    radius1: float,
    radius2: float,
    depth: float,
    material: bpy.types.Material,
    vertices: int = 24,
) -> bpy.types.Object:
    obj = create_primitive(
        bpy.ops.mesh.primitive_cone_add,
        vertices=vertices,
        radius1=radius1,
        radius2=radius2,
        depth=depth,
        location=location,
    )
    obj.name = name
    link_only_to_collection(obj, collection)
    apply_material(obj, material)
    return obj


def add_uv_sphere(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material,
    segments: int = 24,
    ring_count: int = 12,
) -> bpy.types.Object:
    obj = create_primitive(
        bpy.ops.mesh.primitive_uv_sphere_add,
        segments=segments,
        ring_count=ring_count,
        radius=radius,
        location=location,
    )
    obj.name = name
    link_only_to_collection(obj, collection)
    apply_material(obj, material)
    return obj


def add_steps(
    collection: bpy.types.Collection,
    prefix: str,
    widths: list[tuple[float, float, float, float]],
    material: bpy.types.Material,
) -> None:
    for index, (x_scale, y_scale, z_scale, z_loc) in enumerate(widths, start=1):
        add_cube(
            collection,
            f"{prefix}_Step_{index}",
            location=(0.0, 0.0, z_loc),
            scale=(x_scale, y_scale, z_scale),
            material=material,
        )


def build_gatehouse(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Gatehouse",
        [
            (1.35, 1.0, 0.12, 0.12),
            (1.18, 0.82, 0.08, 0.32),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Gatehouse_LeftPylon",
        location=(-0.72, 0.0, 0.95),
        scale=(0.28, 0.46, 0.62),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Gatehouse_RightPylon",
        location=(0.72, 0.0, 0.95),
        scale=(0.28, 0.46, 0.62),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Gatehouse_CrossLintel",
        location=(0.0, 0.0, 1.65),
        scale=(0.96, 0.34, 0.12),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Gatehouse_BackWall",
        location=(0.0, 0.44, 0.92),
        scale=(1.12, 0.14, 0.58),
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Gatehouse_Roof",
        location=(0.0, 0.0, 2.0),
        scale=(1.18, 0.72, 0.14),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Gatehouse_Threshold",
        location=(0.0, -0.42, 0.42),
        scale=(0.88, 0.22, 0.05),
        material=materials["limestone"],
    )
    add_cylinder(
        collection,
        "Gatehouse_LaurelUrn_Left",
        location=(-1.05, -0.16, 0.55),
        radius=0.12,
        depth=0.48,
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Gatehouse_LaurelUrn_Right",
        location=(1.05, -0.16, 0.55),
        radius=0.12,
        depth=0.48,
        material=materials["bronze"],
    )


def build_inner_sanctum(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Sanctum",
        [
            (1.22, 1.02, 0.14, 0.14),
            (1.0, 0.82, 0.09, 0.37),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Sanctum_Cella",
        location=(0.0, 0.06, 1.0),
        scale=(0.82, 0.62, 0.48),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Sanctum_PorticoSlab",
        location=(0.0, -0.6, 0.82),
        scale=(0.86, 0.26, 0.08),
        material=materials["limestone"],
    )
    for index, x_coord in enumerate((-0.5, 0.0, 0.5), start=1):
        add_cylinder(
            collection,
            f"Sanctum_Column_{index}",
            location=(x_coord, -0.58, 0.98),
            radius=0.07,
            depth=0.72,
            material=materials["marble"],
        )
    add_cube(
        collection,
        "Sanctum_Roof",
        location=(0.0, -0.02, 1.64),
        scale=(1.02, 0.78, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Sanctum_OmphalosPlinth",
        location=(0.0, 0.18, 0.62),
        scale=(0.22, 0.18, 0.14),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Sanctum_VaporVent",
        location=(0.0, 0.66, 0.46),
        radius=0.12,
        depth=0.2,
        material=materials["bronze"],
        vertices=16,
    )


def build_storehouse(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Storehouse",
        [
            (1.26, 0.92, 0.12, 0.12),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Storehouse_Body",
        location=(0.0, 0.0, 0.82),
        scale=(0.98, 0.68, 0.56),
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Storehouse_Roof",
        location=(0.0, 0.0, 1.58),
        scale=(1.1, 0.78, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Storehouse_Door",
        location=(0.0, -0.74, 0.78),
        scale=(0.2, 0.06, 0.34),
        material=materials["bronze"],
    )
    for offset, name in [((-0.78, 0.54, 0.44), "Storehouse_JarsLeft"), ((0.84, 0.44, 0.44), "Storehouse_JarsRight")]:
        for index in range(3):
            add_cylinder(
                collection,
                f"{name}_{index+1}",
                location=(offset[0] + index * 0.18, offset[1], offset[2]),
                radius=0.08,
                depth=0.34,
                material=materials["terracotta"],
                vertices=18,
            )


def build_priest_quarters(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Quarters",
        [
            (1.34, 1.0, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Quarters_LeftWing",
        location=(-0.46, 0.0, 0.72),
        scale=(0.46, 0.62, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Quarters_RightWing",
        location=(0.46, 0.0, 0.72),
        scale=(0.46, 0.62, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Quarters_BackWall",
        location=(0.0, 0.46, 0.72),
        scale=(0.92, 0.12, 0.46),
        material=materials["marble"],
    )
    add_plane(
        collection,
        "Quarters_Courtyard",
        location=(0.0, -0.08, 0.18),
        size=0.84,
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Quarters_Roof_Left",
        location=(-0.46, 0.0, 1.28),
        scale=(0.54, 0.72, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Quarters_Roof_Right",
        location=(0.46, 0.0, 1.28),
        scale=(0.54, 0.72, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Quarters_Roof_Back",
        location=(0.0, 0.46, 1.28),
        scale=(0.92, 0.16, 0.12),
        material=materials["terracotta"],
    )
    add_cylinder(
        collection,
        "Quarters_CentralShrine",
        location=(0.0, -0.08, 0.42),
        radius=0.12,
        depth=0.42,
        material=materials["bronze"],
        vertices=16,
    )


def build_temple_of_athena(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Athena",
        [
            (1.42, 1.12, 0.14, 0.14),
            (1.22, 0.94, 0.1, 0.38),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Athena_Cella",
        location=(0.0, 0.16, 1.0),
        scale=(0.92, 0.62, 0.5),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Athena_PorticoSlab",
        location=(0.0, -0.74, 0.84),
        scale=(1.0, 0.24, 0.08),
        material=materials["limestone"],
    )
    for index, x_coord in enumerate((-0.72, -0.24, 0.24, 0.72), start=1):
        add_cylinder(
            collection,
            f"Athena_Column_{index}",
            location=(x_coord, -0.72, 1.02),
            radius=0.08,
            depth=0.84,
            material=materials["marble"],
        )
    add_cube(
        collection,
        "Athena_Roof",
        location=(0.0, 0.02, 1.72),
        scale=(1.18, 0.86, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Athena_OwlPlinth_Left",
        location=(-0.7, -0.18, 0.54),
        scale=(0.16, 0.14, 0.18),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Athena_OwlPlinth_Right",
        location=(0.7, -0.18, 0.54),
        scale=(0.16, 0.14, 0.18),
        material=materials["bronze"],
    )


def build_tholos(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cylinder(
        collection,
        "Tholos_BaseRing",
        location=(0.0, 0.0, 0.14),
        radius=1.18,
        depth=0.28,
        material=materials["limestone"],
        vertices=32,
    )
    add_cylinder(
        collection,
        "Tholos_UpperRing",
        location=(0.0, 0.0, 0.36),
        radius=0.96,
        depth=0.16,
        material=materials["marble"],
        vertices=32,
    )
    add_cylinder(
        collection,
        "Tholos_Cella",
        location=(0.0, 0.0, 0.92),
        radius=0.58,
        depth=0.82,
        material=materials["marble"],
        vertices=32,
    )
    for index, angle_deg in enumerate(range(0, 360, 45), start=1):
        radians = math.radians(angle_deg)
        add_cylinder(
            collection,
            f"Tholos_Column_{index}",
            location=(0.82 * math.cos(radians), 0.82 * math.sin(radians), 0.94),
            radius=0.07,
            depth=0.9,
            material=materials["marble"],
            vertices=20,
        )
    add_cone(
        collection,
        "Tholos_Roof",
        location=(0.0, 0.0, 1.76),
        radius1=0.98,
        radius2=0.1,
        depth=0.64,
        material=materials["terracotta"],
        vertices=32,
    )
    add_cylinder(
        collection,
        "Tholos_Finial",
        location=(0.0, 0.0, 2.14),
        radius=0.08,
        depth=0.18,
        material=materials["bronze"],
        vertices=16,
    )


def build_treasury_vault(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Treasury",
        [
            (1.26, 0.94, 0.12, 0.12),
            (1.08, 0.76, 0.08, 0.32),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Treasury_Body",
        location=(0.0, 0.0, 0.92),
        scale=(0.9, 0.54, 0.52),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Treasury_DoorFrame",
        location=(0.0, -0.6, 0.86),
        scale=(0.3, 0.08, 0.38),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Treasury_DoorLeaf_Left",
        location=(-0.12, -0.66, 0.82),
        scale=(0.1, 0.04, 0.3),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Treasury_DoorLeaf_Right",
        location=(0.12, -0.66, 0.82),
        scale=(0.1, 0.04, 0.3),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Treasury_Roof",
        location=(0.0, 0.0, 1.62),
        scale=(1.02, 0.66, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Treasury_Plaque_Left",
        location=(-0.56, -0.58, 1.08),
        scale=(0.18, 0.04, 0.18),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Treasury_Plaque_Right",
        location=(0.56, -0.58, 1.08),
        scale=(0.18, 0.04, 0.18),
        material=materials["bronze"],
    )


def build_kitchen(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Kitchen",
        [
            (1.34, 0.96, 0.1, 0.1),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Kitchen_MainHall",
        location=(0.0, -0.04, 0.78),
        scale=(0.96, 0.66, 0.48),
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Kitchen_ServiceWing",
        location=(0.74, 0.24, 0.64),
        scale=(0.34, 0.34, 0.34),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Kitchen_Roof_Main",
        location=(0.0, -0.04, 1.46),
        scale=(1.08, 0.78, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Kitchen_Roof_Wing",
        location=(0.74, 0.24, 1.16),
        scale=(0.42, 0.42, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Kitchen_Oven",
        location=(-0.68, 0.2, 0.62),
        scale=(0.22, 0.22, 0.28),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Kitchen_SmokeVent",
        location=(-0.68, 0.2, 1.1),
        radius=0.1,
        depth=0.54,
        material=materials["bronze"],
        vertices=16,
    )
    add_cube(
        collection,
        "Kitchen_BreadTable",
        location=(0.1, -0.76, 0.54),
        scale=(0.34, 0.08, 0.08),
        material=materials["marble"],
    )


def build_xenon(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Xenon",
        [
            (1.54, 1.02, 0.12, 0.12),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Xenon_LeftWing",
        location=(-0.72, 0.0, 0.76),
        scale=(0.48, 0.62, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Xenon_RightWing",
        location=(0.72, 0.0, 0.76),
        scale=(0.48, 0.62, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Xenon_BackHall",
        location=(0.0, 0.54, 0.84),
        scale=(1.12, 0.14, 0.54),
        material=materials["marble"],
    )
    add_plane(
        collection,
        "Xenon_Courtyard",
        location=(0.0, 0.0, 0.22),
        size=1.1,
        material=materials["limestone"],
    )
    for index, x_coord in enumerate((-0.42, 0.0, 0.42), start=1):
        add_cylinder(
            collection,
            f"Xenon_Arcade_{index}",
            location=(x_coord, -0.52, 0.8),
            radius=0.08,
            depth=0.72,
            material=materials["marble"],
        )
    add_cube(
        collection,
        "Xenon_Roof_Left",
        location=(-0.72, 0.0, 1.42),
        scale=(0.56, 0.74, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Xenon_Roof_Right",
        location=(0.72, 0.0, 1.42),
        scale=(0.56, 0.74, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Xenon_Roof_Back",
        location=(0.0, 0.54, 1.48),
        scale=(1.16, 0.18, 0.12),
        material=materials["terracotta"],
    )


def build_purification_font(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Font",
        [
            (0.94, 1.08, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Font_Basin",
        location=(0.0, 0.06, 0.38),
        scale=(0.54, 0.78, 0.18),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Font_Water",
        location=(0.0, 0.06, 0.58),
        scale=(0.42, 0.66, 0.04),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Font_BackStele",
        location=(0.0, 0.66, 0.92),
        scale=(0.24, 0.08, 0.48),
        material=materials["marble"],
    )
    add_cylinder(
        collection,
        "Font_Spout",
        location=(0.0, 0.5, 0.88),
        radius=0.05,
        depth=0.18,
        material=materials["bronze"],
        vertices=16,
    )
    add_cylinder(
        collection,
        "Font_Lamp_Left",
        location=(-0.56, -0.26, 0.34),
        radius=0.08,
        depth=0.32,
        material=materials["bronze"],
        vertices=16,
    )
    add_cylinder(
        collection,
        "Font_Lamp_Right",
        location=(0.56, -0.26, 0.34),
        radius=0.08,
        depth=0.32,
        material=materials["bronze"],
        vertices=16,
    )


def build_sacrificial_altar(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Altar",
        [
            (1.18, 0.98, 0.12, 0.12),
            (0.96, 0.76, 0.08, 0.32),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Altar_Block",
        location=(0.0, 0.0, 0.72),
        scale=(0.48, 0.42, 0.28),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Altar_Channel",
        location=(0.0, -0.22, 0.86),
        scale=(0.1, 0.12, 0.04),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Altar_Table",
        location=(-0.7, 0.18, 0.54),
        scale=(0.26, 0.18, 0.08),
        material=materials["marble"],
    )
    add_cylinder(
        collection,
        "Altar_Brazier_Left",
        location=(0.64, -0.18, 0.58),
        radius=0.12,
        depth=0.36,
        material=materials["bronze"],
        vertices=18,
    )
    add_cylinder(
        collection,
        "Altar_Brazier_Right",
        location=(0.64, 0.18, 0.58),
        radius=0.12,
        depth=0.36,
        material=materials["bronze"],
        vertices=18,
    )
    add_cube(
        collection,
        "Altar_KnifeTray",
        location=(-0.18, 0.42, 0.52),
        scale=(0.16, 0.08, 0.04),
        material=materials["bronze"],
    )


def build_omphalos_stone(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Omphalos",
        [
            (0.82, 0.82, 0.1, 0.1),
        ],
        materials["limestone"],
    )
    add_cylinder(
        collection,
        "Omphalos_Plinth",
        location=(0.0, 0.0, 0.36),
        radius=0.36,
        depth=0.22,
        material=materials["marble"],
        vertices=20,
    )
    add_uv_sphere(
        collection,
        "Omphalos_Stone",
        location=(0.0, 0.0, 0.78),
        radius=0.3,
        material=materials["marble"],
        segments=18,
        ring_count=10,
    )
    add_cylinder(
        collection,
        "Omphalos_Fillet_Left",
        location=(-0.22, 0.0, 0.78),
        radius=0.03,
        depth=0.52,
        material=materials["bronze"],
        vertices=12,
    )
    add_cylinder(
        collection,
        "Omphalos_Fillet_Right",
        location=(0.22, 0.0, 0.78),
        radius=0.03,
        depth=0.52,
        material=materials["bronze"],
        vertices=12,
    )
    add_cylinder(
        collection,
        "Omphalos_Offering_Bowl",
        location=(0.0, -0.46, 0.28),
        radius=0.1,
        depth=0.12,
        material=materials["bronze"],
        vertices=16,
    )


def build_temple_of_hermes(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Hermes",
        [
            (1.32, 1.08, 0.14, 0.14),
            (1.12, 0.88, 0.1, 0.38),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Hermes_Cella",
        location=(0.0, 0.14, 0.94),
        scale=(0.86, 0.58, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Hermes_Portico",
        location=(0.0, -0.72, 0.82),
        scale=(0.94, 0.22, 0.08),
        material=materials["limestone"],
    )
    for index, x_coord in enumerate((-0.54, 0.0, 0.54), start=1):
        add_cylinder(
            collection,
            f"Hermes_Column_{index}",
            location=(x_coord, -0.72, 0.98),
            radius=0.07,
            depth=0.74,
            material=materials["marble"],
        )
    add_cube(
        collection,
        "Hermes_Roof",
        location=(0.0, 0.0, 1.62),
        scale=(1.06, 0.8, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Hermes_Wing_Left",
        location=(-0.76, -0.12, 1.08),
        scale=(0.18, 0.08, 0.1),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Hermes_Wing_Right",
        location=(0.76, -0.12, 1.08),
        scale=(0.18, 0.08, 0.1),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Hermes_CourierStone",
        location=(0.0, 0.58, 0.52),
        radius=0.14,
        depth=0.32,
        material=materials["bronze"],
        vertices=16,
    )


def build_gymnasium(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Gymnasium",
        [
            (1.54, 1.0, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_plane(
        collection,
        "Gymnasium_Court",
        location=(0.0, 0.0, 0.18),
        size=1.58,
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Gymnasium_BackHall",
        location=(0.0, 0.66, 0.72),
        scale=(1.02, 0.14, 0.44),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Gymnasium_LeftWing",
        location=(-1.0, 0.0, 0.66),
        scale=(0.16, 0.72, 0.4),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Gymnasium_RightWing",
        location=(1.0, 0.0, 0.66),
        scale=(0.16, 0.72, 0.4),
        material=materials["marble"],
    )
    for index, x_coord in enumerate((-0.66, -0.22, 0.22, 0.66), start=1):
        add_cylinder(
            collection,
            f"Gymnasium_Colonnade_{index}",
            location=(x_coord, -0.72, 0.74),
            radius=0.07,
            depth=0.62,
            material=materials["marble"],
        )
    add_cube(
        collection,
        "Gymnasium_Roof_Back",
        location=(0.0, 0.66, 1.24),
        scale=(1.08, 0.18, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Gymnasium_Bench",
        location=(0.0, -0.1, 0.32),
        scale=(0.44, 0.1, 0.08),
        material=materials["bronze"],
    )


def build_scriptorium(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Scriptorium",
        [
            (1.22, 0.96, 0.1, 0.1),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Scriptorium_Hall",
        location=(0.0, 0.02, 0.82),
        scale=(0.92, 0.62, 0.5),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Scriptorium_Roof_Left",
        location=(-0.34, 0.02, 1.56),
        scale=(0.48, 0.74, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Scriptorium_Roof_Right",
        location=(0.34, 0.02, 1.56),
        scale=(0.48, 0.74, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Scriptorium_Skylight",
        location=(0.0, 0.02, 1.48),
        scale=(0.08, 0.66, 0.06),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Scriptorium_ScrollRack_Left",
        location=(-0.72, -0.28, 0.56),
        scale=(0.14, 0.12, 0.24),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Scriptorium_ScrollRack_Right",
        location=(0.72, -0.28, 0.56),
        scale=(0.14, 0.12, 0.24),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Scriptorium_Desk",
        location=(0.0, -0.74, 0.46),
        scale=(0.22, 0.08, 0.08),
        material=materials["marble"],
    )


def build_animal_pen(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Pen",
        [
            (1.22, 0.94, 0.06, 0.06),
        ],
        materials["limestone"],
    )
    add_plane(
        collection,
        "Pen_Ground",
        location=(0.0, 0.0, 0.14),
        size=1.24,
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Pen_Fence_Back",
        location=(0.0, 0.68, 0.42),
        scale=(0.96, 0.04, 0.24),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Pen_Fence_Front",
        location=(0.0, -0.68, 0.42),
        scale=(0.7, 0.04, 0.24),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Pen_Fence_Left",
        location=(-0.96, 0.0, 0.42),
        scale=(0.04, 0.64, 0.24),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Pen_Fence_Right",
        location=(0.96, 0.0, 0.42),
        scale=(0.04, 0.64, 0.24),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Pen_Awning",
        location=(-0.34, 0.0, 0.96),
        scale=(0.48, 0.36, 0.08),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Pen_Trough",
        location=(0.34, 0.0, 0.24),
        scale=(0.26, 0.1, 0.08),
        material=materials["marble"],
    )
    add_cylinder(
        collection,
        "Pen_Stock",
        location=(-0.16, -0.18, 0.36),
        radius=0.14,
        depth=0.34,
        material=materials["marble"],
        vertices=14,
    )
    add_cylinder(
        collection,
        "Pen_Stock_Companion",
        location=(0.18, 0.2, 0.34),
        radius=0.12,
        depth=0.3,
        material=materials["marble"],
        vertices=14,
    )


def build_granary(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Granary",
        [
            (1.22, 0.92, 0.1, 0.1),
        ],
        materials["limestone"],
    )
    for x_coord in (-0.54, 0.0, 0.54):
        add_cylinder(
            collection,
            f"Granary_Pier_{x_coord:+.2f}",
            location=(x_coord, 0.0, 0.3),
            radius=0.1,
            depth=0.4,
            material=materials["marble"],
            vertices=16,
        )
    add_cube(
        collection,
        "Granary_Bin",
        location=(0.0, 0.0, 0.88),
        scale=(0.88, 0.58, 0.38),
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Granary_Roof",
        location=(0.0, 0.0, 1.46),
        scale=(0.98, 0.7, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Granary_Door",
        location=(0.0, -0.62, 0.84),
        scale=(0.18, 0.04, 0.28),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Granary_Sack_Left",
        location=(-0.72, 0.36, 0.34),
        radius=0.1,
        depth=0.26,
        material=materials["terracotta"],
        vertices=14,
    )
    add_cylinder(
        collection,
        "Granary_Sack_Right",
        location=(0.74, 0.3, 0.34),
        radius=0.1,
        depth=0.26,
        material=materials["terracotta"],
        vertices=14,
    )


def build_grain_field(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cube(
        collection,
        "GrainField_Terrace",
        location=(0.0, 0.0, 0.12),
        scale=(1.48, 0.98, 0.08),
        material=materials["limestone"],
    )
    for index, x_coord in enumerate((-0.92, -0.46, 0.0, 0.46, 0.92), start=1):
        add_cube(
            collection,
            f"GrainField_Row_{index}",
            location=(x_coord, 0.0, 0.28),
            scale=(0.12, 0.84, 0.12),
            material=materials["grain"],
        )
    add_cube(
        collection,
        "GrainField_Sheaf",
        location=(1.06, 0.34, 0.36),
        scale=(0.12, 0.12, 0.18),
        material=materials["grain"],
    )
    add_cube(
        collection,
        "GrainField_Wall",
        location=(-1.12, 0.0, 0.24),
        scale=(0.08, 0.92, 0.08),
        material=materials["marble"],
    )


def build_olive_press(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Press",
        [
            (1.24, 0.96, 0.1, 0.1),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Press_Hall",
        location=(-0.18, 0.04, 0.76),
        scale=(0.74, 0.58, 0.44),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Press_Roof",
        location=(-0.18, 0.04, 1.42),
        scale=(0.84, 0.7, 0.1),
        material=materials["terracotta"],
    )
    add_cylinder(
        collection,
        "Press_Wheel",
        location=(0.78, 0.12, 0.54),
        radius=0.28,
        depth=0.18,
        material=materials["bronze"],
        vertices=24,
    )
    add_cube(
        collection,
        "Press_Beam",
        location=(0.52, 0.12, 0.86),
        scale=(0.58, 0.08, 0.08),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Press_Jar_Left",
        location=(0.58, -0.44, 0.34),
        radius=0.1,
        depth=0.28,
        material=materials["terracotta"],
        vertices=16,
    )
    add_cylinder(
        collection,
        "Press_Jar_Right",
        location=(0.88, -0.28, 0.34),
        radius=0.1,
        depth=0.28,
        material=materials["terracotta"],
        vertices=16,
    )


def build_olive_grove(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cube(
        collection,
        "OliveGrove_Terrace",
        location=(0.0, 0.0, 0.12),
        scale=(1.42, 0.98, 0.08),
        material=materials["limestone"],
    )
    for index, (x_coord, y_coord, canopy_radius) in enumerate(((-0.64, -0.18, 0.38), (0.0, 0.18, 0.42), (0.68, -0.08, 0.36)), start=1):
        add_cylinder(
            collection,
            f"OliveGrove_Trunk_{index}",
            location=(x_coord, y_coord, 0.42),
            radius=0.06,
            depth=0.5,
            material=materials["terracotta"],
            vertices=12,
        )
        add_cone(
            collection,
            f"OliveGrove_Canopy_{index}",
            location=(x_coord, y_coord, 0.86),
            radius1=canopy_radius,
            radius2=0.06,
            depth=0.52,
            material=materials["olive"],
            vertices=18,
        )
    add_cylinder(
        collection,
        "OliveGrove_Jar",
        location=(1.0, 0.4, 0.26),
        radius=0.1,
        depth=0.18,
        material=materials["bronze"],
        vertices=16,
    )


def build_incense_store(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Incense",
        [
            (1.2, 0.92, 0.1, 0.1),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Incense_Body",
        location=(0.0, 0.0, 0.78),
        scale=(0.84, 0.58, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Incense_Roof",
        location=(0.0, 0.0, 1.44),
        scale=(0.94, 0.7, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Incense_Door",
        location=(0.0, -0.62, 0.76),
        scale=(0.16, 0.04, 0.28),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Incense_Censer_Left",
        location=(-0.54, -0.22, 0.56),
        radius=0.08,
        depth=0.26,
        material=materials["bronze"],
        vertices=16,
    )
    add_cylinder(
        collection,
        "Incense_Censer_Right",
        location=(0.54, -0.22, 0.56),
        radius=0.08,
        depth=0.26,
        material=materials["bronze"],
        vertices=16,
    )
    add_cube(
        collection,
        "Incense_Seal",
        location=(0.0, 0.54, 0.98),
        scale=(0.26, 0.04, 0.18),
        material=materials["bronze"],
    )


def build_incense_workshop(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "IncenseWorkshop",
        [
            (1.1, 0.92, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "IncenseWorkshop_Hall",
        location=(-0.12, 0.0, 0.7),
        scale=(0.7, 0.48, 0.4),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "IncenseWorkshop_Roof",
        location=(-0.12, 0.0, 1.28),
        scale=(0.82, 0.6, 0.08),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "IncenseWorkshop_Bench",
        location=(0.72, -0.12, 0.34),
        scale=(0.24, 0.18, 0.08),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "IncenseWorkshop_Jar_A",
        location=(0.58, 0.34, 0.34),
        radius=0.08,
        depth=0.18,
        material=materials["terracotta"],
        vertices=16,
    )
    add_cylinder(
        collection,
        "IncenseWorkshop_Jar_B",
        location=(0.84, 0.26, 0.3),
        radius=0.07,
        depth=0.16,
        material=materials["terracotta"],
        vertices=16,
    )
    add_cylinder(
        collection,
        "IncenseWorkshop_Censer",
        location=(-0.74, 0.34, 0.42),
        radius=0.08,
        depth=0.24,
        material=materials["bronze"],
        vertices=16,
    )


def build_agora_market(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Agora",
        [
            (1.56, 1.02, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_plane(
        collection,
        "Agora_Plaza",
        location=(0.0, 0.0, 0.16),
        size=1.62,
        material=materials["limestone"],
    )
    stall_specs = [
        ("Agora_Stall_Left", (-0.78, 0.12, 0.56)),
        ("Agora_Stall_Center", (0.0, -0.18, 0.56)),
        ("Agora_Stall_Right", (0.78, 0.12, 0.56)),
    ]
    for name, location in stall_specs:
        add_cube(
            collection,
            f"{name}_Counter",
            location=location,
            scale=(0.28, 0.22, 0.18),
            material=materials["marble"],
        )
        add_cube(
            collection,
            f"{name}_Canopy",
            location=(location[0], location[1], 1.04),
            scale=(0.42, 0.28, 0.08),
            material=materials["terracotta"],
        )
    add_cube(
        collection,
        "Agora_CentralMarker",
        location=(0.0, 0.54, 0.36),
        scale=(0.18, 0.18, 0.18),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Agora_VendorJar_Left",
        location=(-0.38, 0.62, 0.26),
        radius=0.08,
        depth=0.18,
        material=materials["bronze"],
        vertices=14,
    )
    add_cylinder(
        collection,
        "Agora_VendorJar_Right",
        location=(0.38, 0.62, 0.26),
        radius=0.08,
        depth=0.18,
        material=materials["bronze"],
        vertices=14,
    )


def build_sacred_way_kit(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_plane(
        collection,
        "SacredWay_Path",
        location=(0.0, 0.0, 0.12),
        size=1.8,
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "SacredWay_Border_Left",
        location=(-0.82, 0.0, 0.2),
        scale=(0.08, 0.86, 0.06),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "SacredWay_Border_Right",
        location=(0.82, 0.0, 0.2),
        scale=(0.08, 0.86, 0.06),
        material=materials["marble"],
    )
    add_cylinder(
        collection,
        "SacredWay_Brazier",
        location=(-0.44, 0.46, 0.38),
        radius=0.12,
        depth=0.28,
        material=materials["bronze"],
        vertices=18,
    )
    add_cylinder(
        collection,
        "SacredWay_Marker",
        location=(0.48, -0.38, 0.34),
        radius=0.1,
        depth=0.36,
        material=materials["marble"],
        vertices=16,
    )
    add_cube(
        collection,
        "SacredWay_PavingInset",
        location=(0.0, 0.0, 0.14),
        scale=(0.34, 0.52, 0.02),
        material=materials["bronze"],
    )


def build_papyrus_reed_bed(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cube(
        collection,
        "PapyrusBed_Basin",
        location=(0.0, 0.0, 0.1),
        scale=(1.4, 0.9, 0.06),
        material=materials["limestone"],
    )
    add_plane(
        collection,
        "PapyrusBed_Water",
        location=(0.0, 0.0, 0.18),
        size=2.4,
        material=materials["water"],
    )
    for index, (x_coord, y_coord, height) in enumerate(((-0.72, -0.14, 0.52), (-0.22, 0.18, 0.62), (0.26, -0.22, 0.58), (0.74, 0.16, 0.56)), start=1):
        add_cylinder(
            collection,
            f"PapyrusBed_Reed_{index}",
            location=(x_coord, y_coord, height * 0.5 + 0.18),
            radius=0.05,
            depth=height,
            material=materials["reed"],
            vertices=10,
        )
        add_cone(
            collection,
            f"PapyrusBed_Crown_{index}",
            location=(x_coord, y_coord, height + 0.24),
            radius1=0.12,
            radius2=0.02,
            depth=0.2,
            material=materials["grain"],
            vertices=12,
        )
    add_cube(
        collection,
        "PapyrusBed_Bundle",
        location=(1.0, -0.34, 0.28),
        scale=(0.16, 0.1, 0.12),
        material=materials["reed"],
    )


def build_theater(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cylinder(
        collection,
        "Theater_OuterSeating",
        location=(0.0, 0.28, 0.16),
        radius=1.18,
        depth=0.24,
        material=materials["limestone"],
        vertices=28,
    )
    add_cylinder(
        collection,
        "Theater_InnerSeating",
        location=(0.0, 0.28, 0.32),
        radius=0.94,
        depth=0.16,
        material=materials["marble"],
        vertices=28,
    )
    add_cube(
        collection,
        "Theater_Stage",
        location=(0.0, -0.66, 0.32),
        scale=(0.78, 0.26, 0.12),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Theater_Skene",
        location=(0.0, -1.0, 0.72),
        scale=(0.86, 0.12, 0.32),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Theater_Canopy",
        location=(0.0, -1.0, 1.16),
        scale=(0.92, 0.16, 0.08),
        material=materials["terracotta"],
    )


def build_bath_house(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Bath",
        [
            (1.46, 1.0, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Bath_MainHall",
        location=(-0.44, 0.0, 0.76),
        scale=(0.54, 0.62, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Bath_SteamRoom",
        location=(0.58, 0.22, 0.72),
        scale=(0.38, 0.38, 0.42),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Bath_Roof_Main",
        location=(-0.44, 0.0, 1.42),
        scale=(0.64, 0.74, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Bath_Roof_Steam",
        location=(0.58, 0.22, 1.32),
        scale=(0.46, 0.46, 0.08),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Bath_Basin",
        location=(0.62, -0.52, 0.34),
        scale=(0.32, 0.18, 0.12),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Bath_Fixture",
        location=(-0.82, -0.38, 0.32),
        radius=0.08,
        depth=0.22,
        material=materials["bronze"],
        vertices=16,
    )


def build_votive_offering_rack(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "VotiveRack",
        [
            (0.98, 1.18, 0.06, 0.06),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "VotiveRack_Frame_Left",
        location=(-0.36, 0.0, 0.66),
        scale=(0.04, 0.74, 0.46),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "VotiveRack_Frame_Right",
        location=(0.36, 0.0, 0.66),
        scale=(0.04, 0.74, 0.46),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "VotiveRack_TopBeam",
        location=(0.0, 0.0, 1.14),
        scale=(0.42, 0.76, 0.04),
        material=materials["bronze"],
    )
    for index, y_coord in enumerate((-0.44, -0.12, 0.2, 0.52), start=1):
        add_cube(
            collection,
            f"VotiveRack_Tablet_{index}",
            location=(0.0, y_coord, 0.72),
            scale=(0.18, 0.06, 0.18),
            material=materials["marble"],
        )


def build_courier_station(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Courier",
        [
            (1.18, 0.92, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Courier_Office",
        location=(-0.24, 0.0, 0.72),
        scale=(0.62, 0.54, 0.42),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Courier_Roof",
        location=(-0.24, 0.0, 1.34),
        scale=(0.72, 0.66, 0.1),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Courier_PostMarker",
        location=(0.74, -0.16, 0.7),
        scale=(0.08, 0.08, 0.48),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Courier_TackBench",
        location=(0.58, 0.34, 0.34),
        scale=(0.22, 0.12, 0.08),
        material=materials["marble"],
    )


def build_shadow_office(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "ShadowOffice",
        [
            (1.12, 0.92, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "ShadowOffice_Body",
        location=(0.0, 0.06, 0.72),
        scale=(0.82, 0.52, 0.42),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "ShadowOffice_Roof_Left",
        location=(-0.26, 0.06, 1.34),
        scale=(0.34, 0.62, 0.08),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "ShadowOffice_Roof_Right",
        location=(0.26, 0.06, 1.34),
        scale=(0.34, 0.62, 0.08),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "ShadowOffice_HiddenDoor",
        location=(0.0, -0.56, 0.72),
        scale=(0.14, 0.04, 0.26),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "ShadowOffice_Alcove",
        location=(-0.6, 0.42, 0.58),
        scale=(0.12, 0.1, 0.16),
        material=materials["bronze"],
    )


def build_library(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Library",
        [
            (1.22, 0.98, 0.1, 0.1),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Library_Hall",
        location=(0.0, 0.08, 0.82),
        scale=(0.9, 0.58, 0.46),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Library_Roof",
        location=(0.0, 0.08, 1.48),
        scale=(1.0, 0.7, 0.1),
        material=materials["terracotta"],
    )
    for index, x_coord in enumerate((-0.44, 0.0, 0.44), start=1):
        add_cylinder(
            collection,
            f"Library_Column_{index}",
            location=(x_coord, -0.62, 0.82),
            radius=0.07,
            depth=0.54,
            material=materials["marble"],
        )
    add_cube(
        collection,
        "Library_Shelf_Left",
        location=(-0.68, 0.38, 0.54),
        scale=(0.12, 0.12, 0.22),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Library_Shelf_Right",
        location=(0.68, 0.38, 0.54),
        scale=(0.12, 0.12, 0.22),
        material=materials["bronze"],
    )


def build_quarry(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cube(
        collection,
        "Quarry_Terrace_Low",
        location=(0.0, 0.0, 0.16),
        scale=(1.32, 1.0, 0.14),
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Quarry_Terrace_High",
        location=(0.44, 0.28, 0.46),
        scale=(0.76, 0.58, 0.16),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Quarry_Block_One",
        location=(-0.66, -0.2, 0.34),
        scale=(0.22, 0.16, 0.16),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Quarry_Block_Two",
        location=(-0.24, -0.42, 0.3),
        scale=(0.18, 0.14, 0.14),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Quarry_Crane_Post",
        location=(0.92, -0.34, 0.74),
        scale=(0.06, 0.06, 0.46),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Quarry_Crane_Beam",
        location=(0.68, -0.34, 1.14),
        scale=(0.32, 0.04, 0.04),
        material=materials["bronze"],
    )


def build_stonemason(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "Stonemason",
        [
            (1.46, 0.96, 0.08, 0.08),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "Stonemason_Shed",
        location=(0.74, 0.14, 0.68),
        scale=(0.36, 0.42, 0.38),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Stonemason_Roof",
        location=(0.74, 0.14, 1.24),
        scale=(0.44, 0.5, 0.08),
        material=materials["terracotta"],
    )
    add_cube(
        collection,
        "Stonemason_Block_A",
        location=(-0.56, 0.18, 0.3),
        scale=(0.22, 0.16, 0.16),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Stonemason_Block_B",
        location=(-0.18, -0.18, 0.26),
        scale=(0.18, 0.14, 0.14),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Stonemason_Worktable",
        location=(0.08, 0.48, 0.34),
        scale=(0.28, 0.14, 0.08),
        material=materials["bronze"],
    )


def build_excavation_trench(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cube(
        collection,
        "Excavation_Rim",
        location=(0.0, 0.0, 0.14),
        scale=(1.32, 1.0, 0.12),
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "Excavation_Cut",
        location=(0.0, 0.0, 0.02),
        scale=(0.82, 0.62, 0.08),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "Excavation_WallFragment",
        location=(-0.28, 0.12, 0.36),
        scale=(0.22, 0.12, 0.18),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "Excavation_Ladder",
        location=(0.62, -0.18, 0.34),
        scale=(0.06, 0.22, 0.22),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "Excavation_Debris",
        location=(-0.64, -0.34, 0.24),
        radius=0.12,
        depth=0.18,
        material=materials["marble"],
        vertices=14,
    )


def build_relic_vault(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_steps(
        collection,
        "RelicVault",
        [
            (1.16, 0.94, 0.1, 0.1),
            (0.94, 0.72, 0.08, 0.28),
        ],
        materials["limestone"],
    )
    add_cube(
        collection,
        "RelicVault_Entrance",
        location=(0.0, 0.0, 0.66),
        scale=(0.74, 0.42, 0.34),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "RelicVault_Door",
        location=(0.0, -0.46, 0.68),
        scale=(0.16, 0.04, 0.24),
        material=materials["bronze"],
    )
    add_cube(
        collection,
        "RelicVault_Lintel",
        location=(0.0, -0.38, 0.98),
        scale=(0.32, 0.08, 0.06),
        material=materials["bronze"],
    )
    add_cylinder(
        collection,
        "RelicVault_Lamp_Left",
        location=(-0.48, -0.12, 0.44),
        radius=0.08,
        depth=0.22,
        material=materials["bronze"],
        vertices=16,
    )
    add_cylinder(
        collection,
        "RelicVault_Lamp_Right",
        location=(0.48, -0.12, 0.44),
        radius=0.08,
        depth=0.22,
        material=materials["bronze"],
        vertices=16,
    )


def build_buried_chamber(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cube(
        collection,
        "BuriedChamber_Floor",
        location=(0.0, 0.0, 0.14),
        scale=(1.08, 0.86, 0.08),
        material=materials["limestone"],
    )
    add_cube(
        collection,
        "BuriedChamber_Wall_Left",
        location=(-0.82, 0.0, 0.54),
        scale=(0.08, 0.62, 0.32),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "BuriedChamber_Wall_Back",
        location=(0.0, 0.62, 0.54),
        scale=(0.74, 0.08, 0.32),
        material=materials["marble"],
    )
    add_cube(
        collection,
        "BuriedChamber_CollapsedLintel",
        location=(0.18, 0.12, 0.86),
        scale=(0.48, 0.12, 0.08),
        material=materials["terracotta"],
    )
    add_cylinder(
        collection,
        "BuriedChamber_Fragment_Left",
        location=(-0.22, -0.24, 0.28),
        radius=0.12,
        depth=0.2,
        material=materials["marble"],
        vertices=14,
    )
    add_cylinder(
        collection,
        "BuriedChamber_Fragment_Right",
        location=(0.42, -0.14, 0.24),
        radius=0.1,
        depth=0.18,
        material=materials["marble"],
        vertices=14,
    )


def build_cypress_tree(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cylinder(
        collection,
        "Cypress_Trunk",
        location=(0.0, 0.0, 0.32),
        radius=0.08,
        depth=0.64,
        material=materials["wood"],
        vertices=12,
    )
    lower = add_cone(
        collection,
        "Cypress_Lower",
        location=(0.0, 0.0, 0.82),
        radius1=0.34,
        radius2=0.06,
        depth=1.02,
        material=materials["cypress"],
        vertices=16,
    )
    lower.rotation_euler.x = math.radians(2)
    upper = add_cone(
        collection,
        "Cypress_Upper",
        location=(0.0, 0.0, 1.36),
        radius1=0.22,
        radius2=0.02,
        depth=0.78,
        material=materials["cypress"],
        vertices=16,
    )
    upper.rotation_euler.x = math.radians(-3)
    add_cylinder(
        collection,
        "Cypress_RootRock",
        location=(0.0, 0.0, 0.08),
        radius=0.18,
        depth=0.12,
        material=materials["limestone"],
        vertices=10,
    )


def build_laurel_shrub(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cylinder(
        collection,
        "Laurel_BaseRock",
        location=(0.0, 0.0, 0.08),
        radius=0.3,
        depth=0.12,
        material=materials["limestone"],
        vertices=10,
    )
    add_cylinder(
        collection,
        "Laurel_Trunk",
        location=(0.0, 0.0, 0.18),
        radius=0.06,
        depth=0.22,
        material=materials["wood"],
        vertices=12,
    )
    for name, location, radius in [
        ("Laurel_Crown_A", (-0.14, 0.06, 0.42), 0.22),
        ("Laurel_Crown_B", (0.12, 0.08, 0.48), 0.24),
        ("Laurel_Crown_C", (-0.02, -0.14, 0.46), 0.2),
        ("Laurel_Crown_D", (0.14, -0.08, 0.4), 0.18),
        ("Laurel_Crown_E", (0.0, 0.0, 0.58), 0.2),
    ]:
        add_uv_sphere(
            collection,
            name,
            location=location,
            radius=radius,
            material=materials["olive"],
            segments=18,
            ring_count=10,
        )


def build_dry_grass_cluster(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cylinder(
        collection,
        "DryGrass_Rock",
        location=(0.0, 0.0, 0.05),
        radius=0.2,
        depth=0.08,
        material=materials["limestone"],
        vertices=9,
    )
    blade_specs = [
        ("DryGrass_Blade_A", (-0.08, -0.02, 0.22), 14),
        ("DryGrass_Blade_B", (0.0, 0.08, 0.24), -10),
        ("DryGrass_Blade_C", (0.1, -0.04, 0.2), 22),
        ("DryGrass_Blade_D", (-0.02, 0.0, 0.28), -18),
        ("DryGrass_Blade_E", (0.06, 0.02, 0.26), 8),
        ("DryGrass_Blade_F", (-0.12, 0.06, 0.18), -24),
    ]
    for name, location, tilt in blade_specs:
        blade = add_cone(
            collection,
            name,
            location=location,
            radius1=0.03,
            radius2=0.003,
            depth=0.42,
            material=materials["grain"],
            vertices=8,
        )
        blade.rotation_euler.x = math.radians(tilt)
        blade.rotation_euler.y = math.radians(tilt * 0.2)


def build_rocky_outcrop(
    collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cylinder(
        collection,
        "Outcrop_Footing",
        location=(0.0, 0.0, 0.05),
        radius=0.32,
        depth=0.08,
        material=materials["limestone"],
        vertices=8,
    )
    for name, location, scale in [
        ("Outcrop_Main", (-0.12, 0.0, 0.14), (0.12, 0.1, 0.1)),
        ("Outcrop_Secondary", (0.14, -0.02, 0.1), (0.09, 0.08, 0.08)),
        ("Outcrop_Top", (0.02, 0.08, 0.18), (0.07, 0.05, 0.05)),
        ("Outcrop_Spall", (-0.2, -0.1, 0.08), (0.05, 0.04, 0.04)),
    ]:
        rock = add_cube(
            collection,
            name,
            location=location,
            scale=scale,
            material=materials["limestone"],
        )
        rock.rotation_euler.z = math.radians(18 if "Main" in name else -12)
        rock.rotation_euler.x = math.radians(7)
    add_cone(
        collection,
        "Outcrop_Grass",
        location=(0.18, 0.04, 0.18),
        radius1=0.04,
        radius2=0.01,
        depth=0.16,
        material=materials["grain"],
        vertices=8,
    )


PROFILES = {
    "sacred_way_kit": build_sacred_way_kit,
    "gatehouse_entrance": build_gatehouse,
    "inner_sanctum": build_inner_sanctum,
    "storehouse": build_storehouse,
    "priest_quarters": build_priest_quarters,
    "purification_font": build_purification_font,
    "sacrificial_altar": build_sacrificial_altar,
    "omphalos_stone": build_omphalos_stone,
    "temple_of_athena": build_temple_of_athena,
    "temple_of_hermes": build_temple_of_hermes,
    "tholos": build_tholos,
    "treasury_vault": build_treasury_vault,
    "gymnasium": build_gymnasium,
    "scriptorium": build_scriptorium,
    "animal_pen": build_animal_pen,
    "granary": build_granary,
    "grain_field": build_grain_field,
    "kitchen": build_kitchen,
    "olive_grove": build_olive_grove,
    "olive_press": build_olive_press,
    "incense_store": build_incense_store,
    "incense_workshop": build_incense_workshop,
    "papyrus_reed_bed": build_papyrus_reed_bed,
    "xenon": build_xenon,
    "agora_market": build_agora_market,
    "theater": build_theater,
    "bath_house": build_bath_house,
    "votive_offering_rack": build_votive_offering_rack,
    "courier_station": build_courier_station,
    "shadow_office": build_shadow_office,
    "library": build_library,
    "quarry": build_quarry,
    "stonemason": build_stonemason,
    "excavation_trench": build_excavation_trench,
    "relic_vault": build_relic_vault,
    "buried_chamber": build_buried_chamber,
    "cypress_tree": build_cypress_tree,
    "laurel_shrub": build_laurel_shrub,
    "dry_grass_cluster": build_dry_grass_cluster,
    "rocky_outcrop": build_rocky_outcrop,
}


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


def normalize_objects(objects: list[bpy.types.Object]) -> dict[str, float]:
    mins, maxs = world_bbox(objects)
    center_xy = Vector(((mins.x + maxs.x) * 0.5, (mins.y + maxs.y) * 0.5, 0.0))
    for obj in objects:
        obj.location += Vector((-center_xy.x, -center_xy.y, -mins.z))
    bpy.context.view_layer.update()

    mins, maxs = world_bbox(objects)
    return {
        "width": maxs.x - mins.x,
        "depth": maxs.y - mins.y,
        "height": maxs.z - mins.z,
    }


def frame_camera(dimensions: dict[str, float]) -> None:
    camera = bpy.data.objects.get("ORACLE_IsometricCamera")
    target = bpy.data.objects.get("ORACLE_AssetOrigin")
    if camera is None or target is None:
        return
    target.location = Vector((0.0, 0.0, max(dimensions["height"] * 0.42, 0.3)))
    camera.data.ortho_scale = max(
        FOOTPRINT_METERS * 1.55,
        max(dimensions["width"], dimensions["depth"]) * 1.35,
        dimensions["height"] * 2.1,
    )


def save_metadata(dimensions: dict[str, float]) -> None:
    metadata = {
        "asset_id": ASSET_ID,
        "profile": PROFILE_KEY,
        "footprint_meters": FOOTPRINT_METERS,
        "render_path": OUTPUT_RENDER_PATH,
        "blend_path": OUTPUT_BLEND_PATH,
        "dimensions": dimensions,
    }
    ensure_parent(OUTPUT_METADATA_PATH)
    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> dict[str, object]:
    if PROFILE_KEY not in PROFILES:
        raise RuntimeError(f"Unknown profile {PROFILE_KEY!r}")
    if bpy.context.scene.name != "OracleAssetTemplate":
        raise RuntimeError("Open the Oracle template scene before running manual blockouts.")

    cleanup_previous_imports()
    collection = create_asset_collection()
    materials = sanctuary_materials()

    PROFILES[PROFILE_KEY](collection, materials)

    meshes = [obj for obj in collection.objects if obj.type == "MESH"]
    dimensions = normalize_objects(meshes)
    frame_camera(dimensions)

    reference_collection = bpy.data.collections.get("ORACLE_REFERENCE")
    if reference_collection is not None:
        reference_collection.hide_render = True

    scene = bpy.context.scene
    ensure_parent(OUTPUT_RENDER_PATH)
    ensure_parent(OUTPUT_BLEND_PATH)
    scene.render.filepath = OUTPUT_RENDER_PATH
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND_PATH, check_existing=False)
    save_metadata(dimensions)

    summary = {
        "asset_id": ASSET_ID,
        "profile": PROFILE_KEY,
        "render_path": OUTPUT_RENDER_PATH,
        "blend_path": OUTPUT_BLEND_PATH,
        "metadata_path": OUTPUT_METADATA_PATH,
        "dimensions": dimensions,
    }
    print(json.dumps(summary, indent=2))
    return summary


RESULT = main()
