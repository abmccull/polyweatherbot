"""
Build the reusable Oracle Blender render template.

Run this inside Blender's Python context. The script creates:
- a fixed orthographic isometric camera
- a fixed sun/shadow direction
- transparent PNG render output
- reusable limestone, marble, terracotta, and aged bronze materials
- footprint guides and a hidden reference shrine used for validation renders
"""

from __future__ import annotations

import json
import math
import os

import bpy


REPO_ROOT = "/Users/tsc-001/station_sniper/The Oracle"
TEMPLATE_BLEND_PATH = os.path.join(
    REPO_ROOT, "output/art/blender-templates/oracle_isometric_template.blend"
)
METADATA_PATH = os.path.join(
    REPO_ROOT, "output/art/blender-templates/oracle_isometric_template.json"
)
VALIDATION_RENDER_PATH = os.path.join(
    REPO_ROOT, "output/art/renders/template/oracle_template_probe.png"
)

CAMERA_NAME = "ORACLE_IsometricCamera"
SUN_NAME = "ORACLE_SunKey"
ORIGIN_NAME = "ORACLE_AssetOrigin"


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def clear_scene() -> None:
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(scene.collection.children):
        scene.collection.children.unlink(collection)
        bpy.data.collections.remove(collection)

    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)

    bpy.ops.outliner.orphans_purge(do_recursive=True)


def link_only_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for parent in list(obj.users_collection):
        parent.objects.unlink(obj)
    collection.objects.link(obj)


def new_collection(name: str, hide_render: bool = False) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    collection.hide_render = hide_render
    return collection


def configure_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    scene.name = "OracleAssetTemplate"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    scene.frame_start = 1
    scene.frame_end = 1

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.preview_samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1536
    scene.render.resolution_y = 1536
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.filepath = VALIDATION_RENDER_PATH

    scene.display_settings.display_device = "sRGB"
    scene.view_settings.exposure = 0.15

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("OracleWorld")
        scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (320, 0)
    background = nodes.new("ShaderNodeBackground")
    background.location = (0, 0)
    background.inputs["Color"].default_value = (0.95, 0.92, 0.87, 1.0)
    background.inputs["Strength"].default_value = 0.35
    links.new(background.outputs["Background"], output.inputs["Surface"])

    return scene


def create_origin(guides_collection: bpy.types.Collection) -> bpy.types.Object:
    origin = bpy.data.objects.new(ORIGIN_NAME, None)
    origin.empty_display_type = "PLAIN_AXES"
    origin.empty_display_size = 0.5
    guides_collection.objects.link(origin)
    return origin


def create_camera(
    render_collection: bpy.types.Collection, target: bpy.types.Object
) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new(CAMERA_NAME)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 7.5
    camera_data.clip_start = 0.1
    camera_data.clip_end = 500.0

    camera = bpy.data.objects.new(CAMERA_NAME, camera_data)
    camera.location = (10.0, -10.0, 10.0)
    render_collection.objects.link(camera)

    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = camera
    return camera


def create_sun(
    render_collection: bpy.types.Collection, target: bpy.types.Object
) -> bpy.types.Object:
    sun_data = bpy.data.lights.new(SUN_NAME, type="SUN")
    sun_data.energy = 2.4
    sun_data.angle = math.radians(6.5)
    sun_data.color = (1.0, 0.94, 0.84)

    sun = bpy.data.objects.new(SUN_NAME, sun_data)
    sun.location = (7.0, -16.0, 18.0)
    render_collection.objects.link(sun)

    constraint = sun.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    return sun


def create_shadow_catcher(render_collection: bpy.types.Collection) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "ORACLE_ShadowCatcher"
    link_only_to_collection(plane, render_collection)
    plane.display_type = "WIRE"

    try:
        plane.is_shadow_catcher = True
    except AttributeError:
        plane.cycles.is_shadow_catcher = True

    return plane


def add_noise_setup(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    color_a: tuple[float, float, float, float],
    color_b: tuple[float, float, float, float],
    scale: float,
    roughness: float,
):
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-880, 220)
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = roughness

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-620, 220)
    ramp.color_ramp.elements[0].color = color_a
    ramp.color_ramp.elements[1].color = color_b
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (-360, 20)
    bump.inputs["Strength"].default_value = 0.05
    links.new(noise.outputs["Fac"], bump.inputs["Height"])

    return ramp, bump, noise


def build_limestone_material() -> bpy.types.Material:
    material = bpy.data.materials.new("ORACLE_Limestone")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (380, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (80, 0)
    shader.inputs["Roughness"].default_value = 0.83
    shader.inputs["Specular IOR Level"].default_value = 0.35

    ramp, bump, noise = add_noise_setup(
        nodes,
        links,
        (0.77, 0.71, 0.58, 1.0),
        (0.93, 0.89, 0.78, 1.0),
        scale=9.0,
        roughness=0.55,
    )
    noise.inputs["Scale"].default_value = 7.0
    bump.inputs["Strength"].default_value = 0.08

    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_marble_material() -> bpy.types.Material:
    material = bpy.data.materials.new("ORACLE_Marble")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (440, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (140, 0)
    shader.inputs["Roughness"].default_value = 0.58
    shader.inputs["Specular IOR Level"].default_value = 0.48

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-980, 240)
    noise.inputs["Scale"].default_value = 5.5
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.45

    wave = nodes.new("ShaderNodeTexWave")
    wave.location = (-980, -20)
    wave.wave_type = "BANDS"
    wave.bands_direction = "DIAGONAL"
    wave.inputs["Scale"].default_value = 9.5
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 4.0

    mix = nodes.new("ShaderNodeMixRGB")
    mix.location = (-720, 120)
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.55
    links.new(noise.outputs["Color"], mix.inputs["Color1"])
    links.new(wave.outputs["Color"], mix.inputs["Color2"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-440, 120)
    ramp.color_ramp.elements[0].color = (0.83, 0.82, 0.79, 1.0)
    ramp.color_ramp.elements[1].color = (0.96, 0.95, 0.93, 1.0)
    links.new(mix.outputs["Color"], ramp.inputs["Fac"])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (-160, -40)
    bump.inputs["Strength"].default_value = 0.035
    links.new(mix.outputs["Color"], bump.inputs["Height"])

    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_terracotta_material() -> bpy.types.Material:
    material = bpy.data.materials.new("ORACLE_Terracotta")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (380, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (80, 0)
    shader.inputs["Roughness"].default_value = 0.88
    shader.inputs["Specular IOR Level"].default_value = 0.26

    ramp, bump, noise = add_noise_setup(
        nodes,
        links,
        (0.49, 0.25, 0.14, 1.0),
        (0.75, 0.43, 0.22, 1.0),
        scale=12.0,
        roughness=0.35,
    )
    noise.inputs["Detail"].default_value = 6.0
    bump.inputs["Strength"].default_value = 0.03

    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_bronze_material() -> bpy.types.Material:
    material = bpy.data.materials.new("ORACLE_AgedBronze")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (420, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (120, 0)
    shader.inputs["Metallic"].default_value = 1.0
    shader.inputs["Roughness"].default_value = 0.42

    ramp, bump, noise = add_noise_setup(
        nodes,
        links,
        (0.28, 0.20, 0.10, 1.0),
        (0.61, 0.44, 0.18, 1.0),
        scale=18.0,
        roughness=0.4,
    )
    noise.inputs["Detail"].default_value = 10.0
    bump.inputs["Strength"].default_value = 0.018

    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_materials() -> dict[str, bpy.types.Material]:
    return {
        "limestone": build_limestone_material(),
        "marble": build_marble_material(),
        "terracotta": build_terracotta_material(),
        "aged_bronze": build_bronze_material(),
    }


def apply_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def add_footprint_guide(
    guides_collection: bpy.types.Collection,
    size: float,
    location: tuple[float, float, float],
    name: str,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    guide = bpy.context.active_object
    guide.name = name
    link_only_to_collection(guide, guides_collection)
    guide.display_type = "WIRE"
    guide.hide_render = True
    guide.show_in_front = True
    return guide


def add_cube(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
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
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=radius, depth=depth, location=location
    )
    obj = bpy.context.active_object
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
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.active_object
    obj.name = name
    link_only_to_collection(obj, collection)
    apply_material(obj, material)
    return obj


def build_reference_shrine(
    reference_collection: bpy.types.Collection, materials: dict[str, bpy.types.Material]
) -> None:
    add_cube(
        reference_collection,
        "ORACLE_Probe_Base",
        location=(0.0, 0.0, 0.28),
        scale=(1.9, 1.6, 0.28),
        material=materials["limestone"],
    )
    add_cube(
        reference_collection,
        "ORACLE_Probe_Step",
        location=(0.0, 0.0, 0.62),
        scale=(1.45, 1.2, 0.08),
        material=materials["limestone"],
    )
    add_cube(
        reference_collection,
        "ORACLE_Probe_Cella",
        location=(0.0, 0.0, 1.18),
        scale=(1.15, 0.9, 0.46),
        material=materials["marble"],
    )
    add_cube(
        reference_collection,
        "ORACLE_Probe_Roof",
        location=(0.0, 0.0, 1.84),
        scale=(1.32, 1.04, 0.12),
        material=materials["terracotta"],
    )
    add_cube(
        reference_collection,
        "ORACLE_Probe_Roof_Cap",
        location=(0.0, 0.0, 2.06),
        scale=(1.08, 0.84, 0.06),
        material=materials["terracotta"],
    )

    for idx, x_coord in enumerate((-0.82, -0.28, 0.28, 0.82), start=1):
        add_cylinder(
            reference_collection,
            f"ORACLE_Probe_Column_{idx}",
            location=(x_coord, -0.62, 1.08),
            radius=0.08,
            depth=0.92,
            material=materials["marble"],
        )

    add_cube(
        reference_collection,
        "ORACLE_Probe_DoorFrame",
        location=(0.0, 0.82, 1.02),
        scale=(0.26, 0.1, 0.4),
        material=materials["aged_bronze"],
    )
    add_uv_sphere(
        reference_collection,
        "ORACLE_Probe_Finial",
        location=(0.0, 0.0, 2.34),
        radius=0.18,
        material=materials["aged_bronze"],
    )

    for index, x_coord in enumerate((-5.5, -4.0, -2.5, -1.0), start=1):
        swatch = add_uv_sphere(
            reference_collection,
            f"ORACLE_Swatch_{index}",
            location=(x_coord, 5.2, 0.5),
            radius=0.42,
            material=list(materials.values())[index - 1],
        )
        swatch.display_type = "TEXTURED"


def write_metadata(
    camera: bpy.types.Object,
    sun: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    metadata = {
        "template_blend": TEMPLATE_BLEND_PATH,
        "validation_render": VALIDATION_RENDER_PATH,
        "camera": {
            "name": camera.name,
            "type": camera.data.type,
            "location": list(camera.location),
            "ortho_scale": camera.data.ortho_scale,
            "target": ORIGIN_NAME,
        },
        "sun": {
            "name": sun.name,
            "location": list(sun.location),
            "energy": sun.data.energy,
            "angle_radians": sun.data.angle,
            "target": ORIGIN_NAME,
        },
        "materials": list(materials.keys()),
        "render": {
            "engine": bpy.context.scene.render.engine,
            "resolution": [
                bpy.context.scene.render.resolution_x,
                bpy.context.scene.render.resolution_y,
            ],
            "transparent_background": bpy.context.scene.render.film_transparent,
            "format": bpy.context.scene.render.image_settings.file_format,
        },
    }

    ensure_parent(METADATA_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> dict[str, object]:
    ensure_parent(TEMPLATE_BLEND_PATH)
    ensure_parent(METADATA_PATH)
    ensure_parent(VALIDATION_RENDER_PATH)

    clear_scene()
    scene = configure_scene()

    render_collection = new_collection("ORACLE_RENDER")
    guides_collection = new_collection("ORACLE_GUIDES", hide_render=True)
    reference_collection = new_collection("ORACLE_REFERENCE")

    origin = create_origin(guides_collection)
    camera = create_camera(render_collection, origin)
    sun = create_sun(render_collection, origin)
    create_shadow_catcher(render_collection)

    add_footprint_guide(
        guides_collection, size=1.0, location=(-5.5, 0.0, 0.01), name="ORACLE_Footprint_1x1"
    )
    add_footprint_guide(
        guides_collection, size=2.0, location=(-2.5, 0.0, 0.01), name="ORACLE_Footprint_2x2"
    )
    add_footprint_guide(
        guides_collection, size=3.0, location=(1.5, 0.0, 0.01), name="ORACLE_Footprint_3x3"
    )

    materials = build_materials()
    build_reference_shrine(reference_collection, materials)

    scene["oracle_template_version"] = "2026-03-06"
    scene["oracle_camera_profile"] = "fixed_orthographic_isometric"
    scene["oracle_render_output"] = "png_rgba_transparent"
    scene["oracle_material_presets"] = json.dumps(sorted(materials.keys()))

    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)

    reference_collection.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=TEMPLATE_BLEND_PATH, check_existing=False)
    write_metadata(camera, sun, materials)

    summary = {
        "scene": scene.name,
        "template_blend": TEMPLATE_BLEND_PATH,
        "validation_render": VALIDATION_RENDER_PATH,
        "metadata": METADATA_PATH,
        "materials": sorted(materials.keys()),
        "camera_ortho_scale": camera.data.ortho_scale,
    }
    print(json.dumps(summary, indent=2))
    return summary


RESULT = main()
