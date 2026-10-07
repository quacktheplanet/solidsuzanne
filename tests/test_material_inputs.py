"""Everything a material reads survives virtualization: renders of the original and of its VGEO cut match.

    blender -b --factory-startup --python tests/test_material_inputs.py

One mesh carries every input at once: the active UV map under its own name (an image-style checker), a
second UV map (another checker), a colour attribute, a float attribute, and Generated coordinates (each
streamed chunk is its own mesh; their texture space has to be the source's). Rendered with EEVEE before and
after Virtualize Mesh, per input, and through Scatter Instances (level meshes).
"""
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "addons"))
import vgeo  # noqa: E402
from vgeo import native  # noqa: E402

OUT = tempfile.mkdtemp(prefix="vgeo_inputs_")
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("  ok  " if ok else "  FAIL ") + name + (f" [{detail}]" if detail else ""), flush=True)


INPUTS = ("uv", "uv2", "color", "float", "generated")


def material(which):
    m = bpy.data.materials.new(f"m_{which}")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    if which in ("uv", "uv2", "generated"):
        ck = nt.nodes.new("ShaderNodeTexChecker")
        ck.inputs["Scale"].default_value = 10
        if which == "generated":
            src = nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
        else:
            uvn = nt.nodes.new("ShaderNodeUVMap")
            uvn.uv_map = "Scan UV" if which == "uv" else "Second"
            src = uvn.outputs["UV"]
        nt.links.new(src, ck.inputs["Vector"])
        nt.links.new(ck.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_name = "paint" if which == "color" else "wetness"
        nt.links.new(at.outputs["Color" if which == "color" else "Fac"], bsdf.inputs["Base Color"])
    return m


def source(which, x):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=2, location=(x, 0, 0))
    o = bpy.context.active_object
    o.name = f"src_{which}"
    bpy.ops.object.shade_smooth()
    me = o.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lv = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lv)
    a = me.uv_layers.new(name="Scan UV")
    b = me.uv_layers.new(name="Second")
    # the scan UV has seams: a spherical unwrap split at the back
    ang = np.arctan2(co[lv][:, 1], co[lv][:, 0]) / (2 * np.pi) + 0.5
    a.data.foreach_set("uv", np.stack([ang, co[lv][:, 2] * 0.25 + 0.5], 1).ravel())
    b.data.foreach_set("uv", (co[lv][:, [0, 2]] * 0.2 + 0.5).ravel())
    me.uv_layers.active = a
    col = me.color_attributes.new("paint", 'FLOAT_COLOR', 'POINT')
    c = np.ones((len(me.vertices), 4))
    c[:, :3] = np.clip(co * 0.25 + 0.5, 0, 1)
    col.data.foreach_set("color", c.ravel())
    wet = me.attributes.new("wetness", 'FLOAT', 'CORNER')
    wet.data.foreach_set("value", np.clip(co[lv][:, 2] * 0.25 + 0.5, 0, 1))
    me.materials.append(material(which))
    return o


def render(tag):
    s = bpy.context.scene
    s.render.filepath = os.path.join(OUT, tag + ".png")
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(s.render.filepath)
    px = np.array(img.pixels[:]).reshape(-1, 4)[:, :3] * 255
    bpy.data.images.remove(img)
    return px


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vgeo.register()
    try:
        run()
    finally:
        vgeo.unregister()
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


def run():
    check("library carries extra channels (version 3)", native.supports_extras())
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = 320, 240
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
    s.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    s.collection.objects.link(cam)
    s.camera = cam
    cam.location = (5, -5, 3)
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
    s.collection.objects.link(sun)
    sun.rotation_euler = (0.8, 0, 0.5)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "inputs.blend"))
    sample = None
    for w in INPUTS:
        src = source(w, 0.0)
        ref = render(f"orig_{w}")
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = src
        src.select_set(True)
        ok = bpy.ops.vgeo.virtualize(target_chunks=16) == {'FINISHED'}
        proxy = bpy.context.active_object
        px = render(f"vgeo_{w}")
        d = np.abs(px - ref).max(1)
        check(f"{w}: the virtualized render matches", ok and d.mean() < 1.5 and (d > 30).mean() < 0.005,
              f"mean {d.mean():.2f}, {100 * (d > 30).mean():.2f}% off")
        chunks = [o for o in bpy.data.objects if o.type == 'MESH' and o.parent == proxy and len(o.data.polygons)]
        if sample is None and chunks:
            sample = chunks[0].data
            check("chunks keep the UV map's own name, active", sample.uv_layers.active is not None
                  and sample.uv_layers.active.name == "Scan UV" and "Second" in sample.uv_layers)
            check("chunks keep the colour attribute, active", sample.color_attributes.active_color_name == "paint")
            check("chunks use the source's texture space", not sample.use_auto_texspace)
        if w == "uv2":                                    # the instancing path (level meshes) too
            ground = bpy.data.objects.new("ground", bpy.data.meshes.new("ground"))
            ground.data.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
            bpy.context.scene.collection.objects.link(ground)
            bpy.ops.object.select_all(action='DESELECT')
            ground.select_set(True)
            proxy.select_set(True)
            bpy.context.view_layer.objects.active = proxy
            sc = bpy.ops.vgeo.scatter(count=3, seed=1)
            levels = [o for o in bpy.data.objects if o.type == 'MESH' and ".L" in o.name and len(o.data.polygons)]
            check("instance level meshes carry the second UV map", sc == {'FINISHED'} and levels
                  and all("Second" in o.data.uv_layers for o in levels))
            for o in [ground] + [o for o in bpy.data.objects if o.get("vgeo_instancer") or "instancer" in o.name.lower()]:
                bpy.data.objects.remove(o)
        bpy.ops.object.select_all(action='DESELECT')
        proxy.select_set(True)
        bpy.context.view_layer.objects.active = proxy
        bpy.ops.vgeo.restore()
        for o in [o for o in bpy.data.objects if o.name.startswith("src_")]:
            bpy.data.objects.remove(o)


if __name__ == "__main__":
    main()
