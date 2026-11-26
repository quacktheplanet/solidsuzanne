"""Statue avenue: the benchmark scene from the Solid Suzanne demo film.

An avenue of carved Suzanne monoliths (1.43M triangles each, 1,704 copies) on a paved plain of
1.62M-triangle patches (1,188 copies) at dusk, a ringed planet on the horizon: 4.36 billion
triangles. All dense geometry is virtualized with the VGEO add-on and placed by two instancers.

Run (takes a few minutes; writes OUT/statue_avenue.blend and the .vgeo assets next to it):
  blender -b --factory-startup --python examples/statue_avenue/build_scene.py -- [OUT]
  python examples/statue_avenue/build_scene.py -- [OUT]            (bpy module)
Benchmark it with the film's cameras:
  blender -b OUT/statue_avenue.blend --python tools/instancing_bench.py -- \\
      --cameras examples/statue_avenue/cameras.json --error 1.5 --offscreen COARSEN,SHADOWS,CULL
The materials have a "vgeo_mode" value node (0 beauty; 1 and 2 read a "vgeo_viz" debug attribute
the add-on does not write yet, see docs/ROADMAP.md, Debug views)."""
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(D)), "addons"))
import assets  # noqa: E402
import vgeo  # noqa: E402
from vgeo import stream, instances  # noqa: E402
import layout  # noqa: E402

_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.abspath(_argv[0]) if _argv else os.path.join(os.getcwd(), "statue_avenue_out")
os.makedirs(OUT, exist_ok=True)

LOD_STOPS = [
    [0.97, 0.99, 1.00], [0.45, 0.93, 1.00], [0.20, 0.75, 1.00], [0.12, 0.50, 1.00], [0.25, 0.32, 1.00],
    [0.48, 0.26, 1.00], [0.70, 0.22, 0.95], [0.90, 0.20, 0.80], [1.00, 0.22, 0.55], [1.00, 0.26, 0.30],
    [1.00, 0.40, 0.15], [1.00, 0.56, 0.10], [1.00, 0.72, 0.12], [1.00, 0.86, 0.30],
]
FOG_COLOR = (0.55, 0.42, 0.40, 1.0)


def srgb_to_lin(c):
    return [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c]


def make_material(name, base_a, base_b, rough, glow_strength, noise_scale):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    n, L = nt.nodes, nt.links
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")

    def node(t, **kw):
        x = n.new(t)
        for k, v in kw.items():
            setattr(x, k, v)
        return x

    def math_(op, a, b=None):
        x = node("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                x.inputs[i].default_value = v
            else:
                L.new(v, x.inputs[i])
        return x.outputs[0]

    mode = node("ShaderNodeValue", name="vgeo_mode", label="vgeo_mode")
    mode.outputs[0].default_value = 0.0

    # ---- beauty
    tc = node("ShaderNodeTexCoord")
    nz = node("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = noise_scale
    nz.inputs["Detail"].default_value = 6
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = node("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35; ramp.color_ramp.elements[0].color = (*base_a, 1)
    ramp.color_ramp.elements[1].position = 0.7; ramp.color_ramp.elements[1].color = (*base_b, 1)
    L.new(nz.outputs["Fac"], ramp.inputs[0])
    uv = node("ShaderNodeUVMap")
    sep = node("ShaderNodeSeparateXYZ"); L.new(uv.outputs["UV"], sep.inputs[0])
    glow = node("ShaderNodeMapRange"); glow.inputs["From Min"].default_value = 0.15; glow.inputs["From Max"].default_value = 0.75
    glow.interpolation_type = 'SMOOTHSTEP'
    L.new(sep.outputs["X"], glow.inputs["Value"])
    bsdf = node("ShaderNodeBsdfPrincipled")
    L.new(ramp.outputs[0], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Emission Color"].default_value = (0.05, 0.55, 1.0, 1)
    L.new(math_('MULTIPLY', glow.outputs[0], glow_strength), bsdf.inputs["Emission Strength"])
    # distance fog (cheap, both engines)
    cam = node("ShaderNodeCameraData")
    fogk = math_('SUBTRACT', 1.0, math_('EXPONENT', math_('MULTIPLY', cam.outputs["View Distance"], -0.0021)))
    fogk = math_('MULTIPLY', fogk, 0.92)
    fog = node("ShaderNodeEmission"); fog.inputs["Color"].default_value = FOG_COLOR
    fog.inputs["Strength"].default_value = 1.0
    beauty = node("ShaderNodeMixShader")
    L.new(fogk, beauty.inputs[0]); L.new(bsdf.outputs[0], beauty.inputs[1]); L.new(fog.outputs[0], beauty.inputs[2])

    # ---- breakdown: ordered detail-level ramp, meshlet shading, optional facets
    a_cl = node("ShaderNodeAttribute", attribute_type='GEOMETRY', attribute_name="vgeo_viz")
    lod_r = node("ShaderNodeValToRGB")
    cr = lod_r.color_ramp
    cr.interpolation = 'CONSTANT'
    cr.elements[0].position = 0.0; cr.elements[0].color = (*srgb_to_lin(LOD_STOPS[0]), 1)
    cr.elements[1].position = 13 / 16.0; cr.elements[1].color = (*srgb_to_lin(LOD_STOPS[13]), 1)
    for i in range(1, 13):
        e = cr.elements.new(i / 16.0)
        e.color = (*srgb_to_lin(LOD_STOPS[i]), 1)
    L.new(math_('ADD', a_cl.outputs['Alpha'], 0.5 / 16.0), lod_r.inputs[0])
    # meshlet lightness from the cluster's hashed colour
    sc = node("ShaderNodeSeparateColor"); L.new(a_cl.outputs["Color"], sc.inputs[0])
    hsum = math_('ADD', math_('ADD', sc.outputs[0], math_('MULTIPLY', sc.outputs[1], 3.17)), math_('MULTIPLY', sc.outputs[2], 7.31))
    hval = math_('FRACT', math_('MULTIPLY', hsum, 12.9))
    shade = math_('ADD', math_('MULTIPLY', hval, 0.55), 0.45)
    is_mesh = math_('COMPARE', mode.outputs[0], 1.0); is_mesh.node.inputs[2].default_value = 0.1
    shade_sel = math_('ADD', math_('MULTIPLY', shade, is_mesh), math_('SUBTRACT', 1.0, is_mesh))
    dcol = node("ShaderNodeMix", data_type='RGBA', blend_type='MULTIPLY')
    dcol.inputs[0].default_value = 1.0
    L.new(lod_r.outputs[0], dcol.inputs[6])
    comb = node("ShaderNodeCombineColor")
    for i in range(3):
        L.new(shade_sel, comb.inputs[i])
    L.new(comb.outputs[0], dcol.inputs[7])
    geo = node("ShaderNodeNewGeometry")
    is_facet = math_('COMPARE', mode.outputs[0], 2.0); is_facet.node.inputs[2].default_value = 0.1
    nmix = node("ShaderNodeMix", data_type='VECTOR')
    L.new(is_facet, nmix.inputs[0]); L.new(geo.outputs["Normal"], nmix.inputs[4]); L.new(geo.outputs["True Normal"], nmix.inputs[5])
    dbsdf = node("ShaderNodeBsdfPrincipled")
    L.new(dcol.outputs[2], dbsdf.inputs["Base Color"])
    dbsdf.inputs["Roughness"].default_value = 0.75
    L.new(dcol.outputs[2], dbsdf.inputs["Emission Color"])
    dbsdf.inputs["Emission Strength"].default_value = 0.12
    L.new(nmix.outputs[1], dbsdf.inputs["Normal"])
    dfog = node("ShaderNodeMixShader")
    L.new(math_('MULTIPLY', fogk, 0.35), dfog.inputs[0]); L.new(dbsdf.outputs[0], dfog.inputs[1]); L.new(fog.outputs[0], dfog.inputs[2])

    final = node("ShaderNodeMixShader")
    L.new(math_('GREATER_THAN', mode.outputs[0], 0.5), final.inputs[0])
    L.new(beauty.outputs[0], final.inputs[1]); L.new(dfog.outputs[0], final.inputs[2])
    L.new(final.outputs[0], out.inputs["Surface"])
    return m


def set_mode(v):
    for m in bpy.data.materials:
        if m.use_nodes and m.node_tree and "vgeo_mode" in m.node_tree.nodes:
            m.node_tree.nodes["vgeo_mode"].outputs[0].default_value = float(v)


def make_world():
    w = bpy.data.worlds.new("Dusk")
    w.use_nodes = True
    nt = w.node_tree
    n, L = nt.nodes, nt.links
    bg = n["Background"]
    tc = n.new("ShaderNodeTexCoord")
    sepz = n.new("ShaderNodeSeparateXYZ"); L.new(tc.outputs["Generated"], sepz.inputs[0])
    ramp = n.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0; cr.elements[0].color = (0.32, 0.20, 0.18, 1)
    cr.elements[1].position = 0.55; cr.elements[1].color = (0.020, 0.035, 0.09, 1)
    e = cr.elements.new(0.04); e.color = (0.95, 0.52, 0.30, 1)
    e = cr.elements.new(0.16); e.color = (0.30, 0.22, 0.30, 1)
    L.new(sepz.outputs["Z"], ramp.inputs[0])
    # sun glow
    dotn = n.new("ShaderNodeVectorMath"); dotn.operation = 'DOT_PRODUCT'
    L.new(tc.outputs["Generated"], dotn.inputs[0]); dotn.inputs[1].default_value = layout.SUN_DIR
    pw = n.new("ShaderNodeMath"); pw.operation = 'POWER'; L.new(dotn.outputs["Value"], pw.inputs[0]); pw.inputs[1].default_value = 40.0
    pw2 = n.new("ShaderNodeMath"); pw2.operation = 'POWER'; L.new(dotn.outputs["Value"], pw2.inputs[0]); pw2.inputs[1].default_value = 900.0
    g = n.new("ShaderNodeMath"); g.operation = 'MULTIPLY_ADD'; L.new(pw.outputs[0], g.inputs[0]); g.inputs[1].default_value = 1.6
    L.new(pw2.outputs[0], g.inputs[2])
    sc = n.new("ShaderNodeMath"); sc.operation = 'MULTIPLY'; L.new(g.outputs[0], sc.inputs[0]); sc.inputs[1].default_value = 6.0
    addc = n.new("ShaderNodeMix"); addc.data_type = 'RGBA'; addc.blend_type = 'ADD'
    L.new(sc.outputs[0], addc.inputs[0]); L.new(ramp.outputs[0], addc.inputs[6]); addc.inputs[7].default_value = (1.0, 0.62, 0.35, 1)
    L.new(addc.outputs[2], bg.inputs[0])
    bg.inputs[1].default_value = 1.0
    return w


def make_planet():
    # gas giant: banded sphere far away on the avenue's horizon, lit by the sun
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=layout.PLANET_R, location=layout.PLANET_POS)
    pl = bpy.context.object; pl.name = "Planet"
    bpy.ops.object.shade_smooth()
    pl.rotation_euler = (0.35, 0.0, 0.5)
    m = bpy.data.materials.new("Planet"); m.use_nodes = True
    nt = m.node_tree; n, L = nt.nodes, nt.links
    b = n["Principled BSDF"]
    tc = n.new("ShaderNodeTexCoord")
    wave = n.new("ShaderNodeTexWave"); wave.wave_type = 'BANDS'; wave.bands_direction = 'Z'
    wave.inputs["Scale"].default_value = 0.0012; wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 3.0
    L.new(tc.outputs["Object"], wave.inputs["Vector"])
    r = n.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].color = (0.55, 0.36, 0.22, 1); r.color_ramp.elements[1].color = (0.92, 0.80, 0.62, 1)
    e = r.color_ramp.elements.new(0.5); e.color = (0.75, 0.50, 0.32, 1)
    L.new(wave.outputs["Fac"], r.inputs[0]); L.new(r.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 1.0
    pl.data.materials.append(m)
    # rings
    bpy.ops.mesh.primitive_circle_add(vertices=256, radius=layout.PLANET_R * 2.35, fill_type='NGON', location=layout.PLANET_POS)
    ring = bpy.context.object; ring.name = "Rings"
    ring.rotation_euler = (math.radians(13), math.radians(-9), 0.0)
    rm = bpy.data.materials.new("Rings"); rm.use_nodes = True
    rnt = rm.node_tree; rn, rL = rnt.nodes, rnt.links
    rb = rn["Principled BSDF"]
    rtc = rn.new("ShaderNodeTexCoord")
    ln = rn.new("ShaderNodeVectorMath"); ln.operation = 'LENGTH'; rL.new(rtc.outputs["Object"], ln.inputs[0])
    nr = rn.new("ShaderNodeMath"); nr.operation = 'DIVIDE'; rL.new(ln.outputs["Value"], nr.inputs[0]); nr.inputs[1].default_value = layout.PLANET_R * 2.35
    rr = rn.new("ShaderNodeValToRGB")
    cr = rr.color_ramp; cr.interpolation = 'LINEAR'
    cr.elements[0].position = 0.0; cr.elements[0].color = (0, 0, 0, 0)
    cr.elements[1].position = 1.0; cr.elements[1].color = (0, 0, 0, 0)
    for pos, a in ((0.58, 0.0), (0.60, 0.55), (0.70, 0.8), (0.72, 0.25), (0.78, 0.7), (0.86, 0.6), (0.88, 0.05), (0.92, 0.35), (0.97, 0.0)):
        e = cr.elements.new(pos); e.color = (0.9, 0.8, 0.66, a)
    rL.new(nr.outputs[0], rr.inputs[0])
    rL.new(rr.outputs["Color"], rb.inputs["Base Color"])
    rL.new(rr.outputs["Alpha"], rb.inputs["Alpha"])
    rb.inputs["Roughness"].default_value = 1.0
    ring.data.materials.append(rm)
    return pl, ring


def make_instancer(source, name, pos, rot, scale):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(pos))
    me.vertices.foreach_set("co", np.asarray(pos, np.float32).ravel())
    me.attributes.new(instances.ROT_ATTR, 'FLOAT_VECTOR', 'POINT').data.foreach_set("vector", np.asarray(rot, np.float32).ravel())
    me.attributes.new(instances.SCALE_ATTR, 'FLOAT', 'POINT').data.foreach_set("value", np.asarray(scale, np.float32))
    me.attributes.new(instances.LEVEL_ATTR, 'INT', 'POINT')
    me.update()
    inst = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(inst)
    inst.vgeo_inst.uid = stream.new_uid()
    inst.vgeo_inst.source = source
    inst.vgeo_inst.pixel_error = source.vgeo.pixel_error
    inst.vgeo_inst.render_pixel_error = source.vgeo.render_pixel_error
    instances.build_levels(inst, source)
    return inst


def virtualize(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(o == ob)
    bpy.context.view_layer.objects.active = ob
    t = time.perf_counter()
    bpy.ops.vgeo.virtualize()
    p = bpy.context.active_object
    print(f"virtualized {ob.name} in {time.perf_counter() - t:.1f}s", flush=True)
    return p


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vgeo.register()
    scene = bpy.context.scene
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "statue_avenue.blend"))
    t0 = time.perf_counter()
    mono = assets.make_monolith()
    floor = assets.make_floor_patch(size=layout.PATCH, n=layout.PATCH_N)
    print(f"assets built in {time.perf_counter() - t0:.1f}s", flush=True)
    mono.data.materials.append(make_material("Basalt", (0.045, 0.043, 0.042), (0.11, 0.10, 0.095), 0.5, 2.2, 0.9))
    floor.data.materials.append(make_material("Paving", (0.12, 0.10, 0.085), (0.24, 0.20, 0.16), 0.75, 0.5, 0.25))
    mono_tris = sum(len(p.vertices) - 2 for p in mono.data.polygons)
    floor_tris = sum(len(p.vertices) - 2 for p in floor.data.polygons)
    mp = virtualize(mono)
    fp = virtualize(floor)
    # the dense originals are no longer needed: the proxies stream from the .vgeo files on disk
    for src in (mono, floor):
        me = src.data
        if src.name in bpy.data.objects:
            bpy.data.objects.remove(src)
        if me.users == 0:
            bpy.data.meshes.remove(me)
    for p in (mp, fp):
        p.hide_render = True
        p.hide_set(True)
    st = layout.statue_placements()
    fl = layout.floor_placements()
    make_instancer(mp, "Statues", *st)
    make_instancer(fp, "Paving", *fl)
    # an underlay hides any hairline gap between paving patches
    bpy.ops.mesh.primitive_plane_add(size=4000, location=(0, 300, -0.07))
    und = bpy.context.object; und.name = "Underlay"
    um = bpy.data.materials.new("Underlay"); um.use_nodes = True
    um.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.025, 0.02, 1)
    und.data.materials.append(um)
    make_planet()
    scene.world = make_world()
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.2
    sun.data.color = (1.0, 0.78, 0.55)
    sun.data.angle = math.radians(1.5)
    sun.rotation_euler = Vector(layout.SUN_DIR).to_track_quat('Z', 'Y').to_euler()
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.clip_start = 0.1; cam.data.clip_end = 20000
    scene.collection.objects.link(cam); scene.camera = cam
    eye = bpy.data.objects.new("Eye", bpy.data.cameras.new("Eye"))
    eye.data.clip_start = 0.1; eye.data.clip_end = 20000
    scene.collection.objects.link(eye)
    scene["scene_tris"] = float(len(st[0]) * mono_tris + len(fl[0]) * floor_tris)
    scene["statues"] = len(st[0]); scene["patches"] = len(fl[0])
    scene["mono_tris"] = mono_tris; scene["floor_tris"] = floor_tris
    r = scene.render
    r.resolution_x, r.resolution_y = 1280, 720
    r.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except TypeError:
        pass
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.max_bounces = 4
    r.use_persistent_data = True
    scene.eevee.taa_render_samples = 8
    bpy.ops.wm.save_mainfile()
    print("scene triangles", f"{int(scene['scene_tris']):,}", "statues", len(st[0]), "patches", len(fl[0]), flush=True)


if __name__ == "__main__":
    build()
