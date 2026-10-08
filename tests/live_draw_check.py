"""Live Draw in a real window (needs the UI, not -b): it draws, looks like EEVEE (with and without sun shadows),
is fast, renders stay EEVEE.

    blender [scene.blend] --python tests/live_draw_check.py -- [--out DIR]

Without a scene it builds one: a 260k-triangle sphere with an image base colour and an image
normal map (both generated) floating over a virtualized ground, 60 instanced copies of a virtualized rock
scattered on the ground, two ordinary (not virtualized) meshes: a slab half in the sphere's shadow and a
pillar casting onto the ground, a sun and a plain world. Overlays are hidden (Material Preview then gives
draw handlers no depth; Live Draw writes the ordinary meshes' own). With a scene (e.g. a virtualized Poly Haven scan) it uses
its VGEO proxies, its suns and the scene camera.

1. Live Draw draws the assets in Material Preview (scene lights and world), triangles counted
2. with the sun's shadows off, the same view drawn by EEVEE from the Blender-mesh path (Live Draw off) looks
   the same: mean difference under 12/255 over the assets' pixels. Why not tighter: EEVEE adds what Live
   Draw doesn't (screen-space effects, its own specular and world lighting model, AgX done exactly instead
   of approximated); a missing texture, a flipped normal map or wrong colour management lands far above it
   (a flat grey asset differs by 40-70/255)
3. with the sun's shadows on, Live Draw against EEVEE again:
   - shadows appear: on the generated scene, the ground under the sphere's shadow (worked out from the sun
     direction) is much darker than without shadows; the ordinary slab in it darkens like in EEVEE, and
     the ordinary pillar's shadow on the virtualized ground is as dark as EEVEE's;
   - they land where EEVEE's do: the pixels that darken when shadows go on, Live Draw against EEVEE,
     overlap (intersection over union);
   - it still looks like EEVEE overall;
   - the same with overlays shown (Blender's depth instead of Live Draw's own);
   - the Shadows scene toggle and the light's own Shadow toggle each switch them off
4. drawing stays fast: the draw handler's own time (shadow map updates included) under 4 ms while
   orbiting, frames while orbiting under 33 ms median, with and without shadows (both reported), and a
   still view renders no shadow map
5. a final render (F12 path) still goes through EEVEE with real meshes: the assets are in the render, and
   on the generated scene the sphere's shadow is in it too
"""
import json
import math
import os
import sys
import tempfile
import time

import bpy
import numpy as np
from mathutils import Euler, Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))
import vgeo  # noqa: E402
from vgeo import instances, livedraw, shadows, stream  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else tempfile.mkdtemp(prefix="vgeo_live_")
os.makedirs(OUT, exist_ok=True)
bpy.context.preferences.view.show_splash = False
try:
    vgeo.register()
except ValueError:
    pass
results, report = [], {"blender": bpy.app.version_string}
SPHERE_CENTER, GROUND_Z = Vector((0.0, 0.0, 0.0)), -2.6
generated = {"on": False}


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def finish():
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed", flush=True)
    report["passed"], report["failed"] = len(results) - len(failed), failed
    with open(os.path.join(OUT, "live_draw_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    os._exit(1 if failed else 0)


def virtualize(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    win, area = area3d()
    with bpy.context.temp_override(window=win, area=area, active_object=ob, object=ob,
                                   selected_objects=[ob]):
        rc = bpy.ops.vgeo.virtualize()
    check(f"virtualize {ob.name}", rc == {'FINISHED'})
    return next(o for o in bpy.context.scene.objects if o.vgeo.uid and o.vgeo.source == ob)


def plain_material(name, rgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    return m


def build_scene():
    """A displaced sphere with generated base colour and normal map images over a ground with rocks."""
    import bmesh
    generated["on"] = True
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    W = 256
    yy, xx = np.mgrid[0:W, 0:W] / W
    base = np.ones((W, W, 4), np.float32)
    check_ = ((np.floor(xx * 8) + np.floor(yy * 8)) % 2)[..., None]
    base[..., :3] = np.where(check_ > 0, [0.75, 0.35, 0.15], [0.2, 0.45, 0.7])
    img = bpy.data.images.new("live_base", W, W)
    img.pixels.foreach_set(base.ravel())
    img.pack()
    h = np.sin(xx * 40) * np.sin(yy * 40)
    nx, ny = np.gradient(h)
    nrm = np.stack([0.5 - nx * 2, 0.5 - ny * 2, np.ones_like(h), np.ones_like(h)], -1).astype(np.float32)
    nimg = bpy.data.images.new("live_normal", W, W, float_buffer=True)
    nimg.colorspace_settings.name = "Non-Color"
    nimg.pixels.foreach_set(np.clip(nrm, 0, 1).ravel())
    nimg.pack()
    me = bpy.data.meshes.new("live_sphere")
    bm = bmesh.new()
    bm.loops.layers.uv.new("UVMap")          # calc_uvs fills an existing layer only
    bmesh.ops.create_uvsphere(bm, u_segments=512, v_segments=256, radius=2.0, calc_uvs=True)
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ob = bpy.data.objects.new("live_sphere", me)
    bpy.context.scene.collection.objects.link(ob)
    m = bpy.data.materials.new("live_mat")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    tn = nt.nodes.new("ShaderNodeTexImage")
    tn.image = nimg
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.6
    ob.data.materials.append(m)
    # the ground: a gently rolling 18 m grid (~130k triangles)
    gme = bpy.data.meshes.new("live_ground")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=256, y_segments=256, size=9.0)
    for v in bm.verts:
        v.co.z = 0.08 * math.sin(v.co.x * 1.3) * math.cos(v.co.y * 1.1)
    bm.to_mesh(gme)
    bm.free()
    ground = bpy.data.objects.new("live_ground", gme)
    ground.location.z = GROUND_Z
    ground.data.materials.append(plain_material("live_ground_mat", (0.55, 0.52, 0.48), 0.8))
    bpy.context.scene.collection.objects.link(ground)
    # a lumpy rock (~80k triangles), scattered 60 times over the ground
    rme = bpy.data.meshes.new("live_rock")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=7, radius=0.35)
    for v in bm.verts:
        n = v.co.normalized()
        v.co *= 1.0 + 0.18 * math.sin(n.x * 7.0) * math.sin(n.y * 6.0 + 1.0) * math.cos(n.z * 5.0)
        v.co.z *= 0.7
    bm.to_mesh(rme)
    bm.free()
    rme.polygons.foreach_set("use_smooth", [True] * len(rme.polygons))
    rock = bpy.data.objects.new("live_rock", rme)
    rock.location = (3.0, -3.2, GROUND_Z + 0.2)
    rock.data.materials.append(plain_material("live_rock_mat", (0.35, 0.33, 0.3), 0.7))
    bpy.context.scene.collection.objects.link(rock)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
    sun.data.energy = 3.0
    sun.data.use_shadow = False
    sun.rotation_euler = (0.7, 0.2, 0.6)
    bpy.context.scene.collection.objects.link(sun)
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.3, 0.35, 0.45, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    bpy.context.scene.world = w
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (-7.0, -8.5, 4.0)
    cam.rotation_euler = (Vector((-0.9, 0.6, -1.6)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    # an ordinary mesh (not virtualized), half in the sphere's shadow: Live Draw darkens it too
    bpy.context.view_layer.update()
    sme = bpy.data.meshes.new("live_slab")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(sme)
    bm.free()
    slab = bpy.data.objects.new("live_slab", sme)
    slab.location = slab_center()
    slab.scale = (SLAB, SLAB, 0.3)
    slab.data.materials.append(plain_material("live_slab_mat", (0.45, 0.55, 0.6), 0.6))
    bpy.context.scene.collection.objects.link(slab)
    # and an ordinary pillar in the open, casting onto the virtualized ground
    pme = bpy.data.meshes.new("live_pillar")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(pme)
    bm.free()
    pillar = bpy.data.objects.new("live_pillar", pme)
    base, _tip = pillar_points()
    pillar.location = base + Vector((0, 0, 1.4))
    pillar.scale = (0.4, 0.4, 3.0)
    pillar.data.materials.append(plain_material("live_pillar_mat", (0.6, 0.5, 0.4), 0.6))
    bpy.context.scene.collection.objects.link(pillar)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "live_scene.blend"))
    virtualize(ob)
    rp = virtualize(rock)
    inst = instances.create_instancer(bpy.context, rp, ground, 60, seed=3, scale_range=(0.6, 1.4))
    virtualize(ground)
    check("scatter 60 copies of the rock", inst is not None and len(inst.data.vertices) == 60)


def area3d():
    win = bpy.context.window_manager.windows[0]
    return win, next(a for a in win.screen.areas if a.type == 'VIEW_3D')


def shot(name):
    win, area = area3d()
    path = os.path.join(OUT, name + ".png")
    with bpy.context.temp_override(window=win, area=area):
        bpy.ops.screen.screenshot_area(filepath=path)
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], 4)[..., :3] * 255
    bpy.data.images.remove(img)
    return px


def save_side(a, b, name):
    side = np.concatenate([a, b], axis=1)
    out = bpy.data.images.new(name, side.shape[1], side.shape[0])
    rgba = np.ones((side.shape[0], side.shape[1], 4), np.float32)
    rgba[..., :3] = side / 255.0
    out.pixels.foreach_set(rgba.ravel())
    out.filepath_raw = os.path.join(OUT, name + ".png")
    out.file_format = 'PNG'
    out.save()
    bpy.data.images.remove(out)


def lum(px):
    return px @ np.array([0.2126, 0.7152, 0.0722])


def asset_mask(*imgs):
    ref = imgs[0]
    bg = np.median(np.concatenate([ref[:8].reshape(-1, 3), ref[-8:].reshape(-1, 3)]), axis=0)
    m = np.zeros(ref.shape[:2], bool)
    for im in imgs:
        m |= np.abs(im - bg).max(2) > 12
    return m


def compare(live, eevee):
    mask = asset_mask(eevee, live)
    d = np.abs(live - eevee).max(2)[mask]
    mean = float(d.mean()) if d.size else 999.0
    return mask, mean, (float(np.percentile(d, 90)) if d.size else 0.0)


def to_px(co):
    """(row, col) of a world point in an area screenshot (rows from the bottom, like the image)."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    win, area = area3d()
    region = next(r for r in area.regions if r.type == 'WINDOW')
    p = location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(co))
    if p is None:
        return None
    return int(region.y - area.y + p.y), int(region.x - area.x + p.x)


def patch(px, rc, r=6):
    y, x = rc
    return float(lum(px[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)).mean())


def shadow_point():
    """Where the sphere's centre falls on the ground along the sun: the middle of its shadow."""
    sun = next(o for o in bpy.context.scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN')
    d = (sun.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    t = (SPHERE_CENTER.z - GROUND_Z) / d.z
    return SPHERE_CENTER - d * t + Vector((0, 0, 0.05))


def lit_points():
    """Ground beside the sphere on the sun's side: no shadow falls there."""
    sun = next(o for o in bpy.context.scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN')
    d = (sun.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    h = Vector((d.x, d.y, 0)).normalized()
    side = Vector((-h.y, h.x, 0))
    return [SPHERE_CENTER + h * 3.0 + side * s_ + Vector((0, 0, GROUND_Z + 0.05)) for s_ in (-1.5, 0.0, 1.5)]


SLAB = 1.8


def slab_center():
    """Beside the shadow's middle, so the slab's inner half is shadowed."""
    sun = next(o for o in bpy.context.scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN')
    d = (sun.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    h = Vector((d.x, d.y, 0)).normalized()
    side = Vector((-h.y, h.x, 0))
    p = shadow_point() + side * 1.9
    return Vector((p.x, p.y, GROUND_Z + 0.15))


def slab_shadow_point():
    """On the slab's top, inside the sphere's shadow."""
    c = slab_center()
    toward = (shadow_point() - c)
    toward.z = 0
    p = c + toward.normalized() * (SLAB * 0.35)
    return Vector((p.x, p.y, GROUND_Z + 0.31))


def pillar_points():
    """(base of the pillar, a ground point in its shadow), in the open on the sun's side of the sphere."""
    sun = next(o for o in bpy.context.scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN')
    d = (sun.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    h = Vector((d.x, d.y, 0)).normalized()
    side = Vector((-h.y, h.x, 0))
    base = SPHERE_CENTER + h * 3.0 - side * 3.0
    base.z = GROUND_Z
    reach = 1.5 * Vector((d.x, d.y, 0)).length / d.z      # where the pillar's middle falls
    tip = base - h * reach
    tip.z = GROUND_Z + 0.08
    return base, tip


def suns(scene):
    return [o for o in scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN']


frames = []
_last = [time.perf_counter()]


def _count():
    now = time.perf_counter()
    frames.append((now - _last[0]) * 1000)
    _last[0] = now


st = {"k": 0, "t": 0.0}


def settled_live():
    """Live Draw has drawn, no cut or shadow level is still building."""
    if not livedraw.is_active() or livedraw.drawn_triangles() == 0:
        return False
    if any(c.pending is not None for c in livedraw._cuts.values()):
        return False
    if any(ls.queue or ls.job is not None for ls in livedraw._levels.values()):
        return False
    return not livedraw._frame.get("busy", True)


def settled_eevee(scene):
    for p in stream.proxies(scene):
        if p.vgeo.slot_of:
            continue
        rt = stream._runtimes.get(p.vgeo.uid)
        if rt is None or rt.updates == 0 or not rt.valid.all():
            return False
    return True


def wait(cond, k_retry, then, limit=90):
    if not cond() and time.perf_counter() - st["t"] < limit:
        st["k"] = k_retry
        return 0.25
    return then


def orbit_stats(label):
    f = np.array(frames[5:]) if len(frames) > 8 else np.array([999.0])
    dm = livedraw._state["frame_ms"][-100:]
    sm = livedraw._state.get("shadow_ms", [])[-100:]
    out = {"frame_ms_median": float(np.median(f)), "frame_ms_p95": float(np.percentile(f, 95)),
           "draw_handler_ms": float(np.median(dm)) if dm else 999.0,
           "draw_handler_ms_p95": float(np.percentile(dm, 95)) if dm else 999.0,
           "shadow_update_ms": float(np.median(sm)) if sm else 0.0,
           "shadow_update_ms_p95": float(np.percentile(sm, 95)) if sm else 0.0,
           "shadow_renders": shadows.stats["renders"] - st.get("renders0", 0),
           "frames": len(f)}
    rm = shadows.stats["render_ms"][-max(1, out["shadow_renders"]):] if out["shadow_renders"] else []
    out["shadow_render_ms"] = float(np.median(rm)) if rm else 0.0
    report[label] = out
    return out


ORBIT = 125


def tick():
    win, area = area3d()
    space = area.spaces.active
    r3 = space.region_3d
    k = st["k"]
    st["k"] += 1
    scene = bpy.context.scene
    if k == 0:
        if not stream.proxies(scene):
            build_scene()
        space.shading.type = 'MATERIAL'
        space.shading.use_scene_lights = True
        space.shading.use_scene_world = True
        # overlays hidden: Material Preview then gives draw handlers no depth, and Live Draw writes the
        # ordinary meshes' own (step 8 checks the overlays-on path, with Blender's depth, gives the same)
        space.overlay.show_overlays = False
        for prop in ("show_floor", "show_axis_x", "show_axis_y", "show_axis_z", "show_cursor", "show_text",
                     "show_stats", "show_extras", "show_object_origins", "show_outline_selected",
                     "show_relationship_lines", "show_bones", "show_motion_paths", "show_annotation",
                     "show_look_dev"):
            if hasattr(space.overlay, prop):
                setattr(space.overlay, prop, False)
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        r3.view_perspective = 'CAMERA'
        st["suns"] = suns(scene)
        st["shadow_was"] = [s.data.use_shadow for s in st["suns"]]
        for s in st["suns"]:
            s.data.use_shadow = False
        scene.vgeo_live_draw = True
        scene.vgeo_live_shadows = True
        st["t"] = time.perf_counter()
        return 0.5
    if k == 1:   # let the cuts land
        return wait(settled_live, 1, 1.0)
    if k == 2:
        st["live"] = shot("live_draw")
        tris = livedraw.drawn_triangles()
        report["live_triangles"] = tris
        check("Live Draw draws the assets", livedraw.is_active() and tris > 0, f"{tris:,} triangles")
        check("no shadow map while the sun's Shadow is off", not livedraw._state.get("shadowed"))
        scene.vgeo_live_draw = False          # the Blender-mesh path, drawn by EEVEE
        st["t"] = time.perf_counter()
        return 1.0
    if k == 3:   # wait for the EEVEE path to land and settle
        return wait(lambda: settled_eevee(scene), 3, 3.0)
    if k == 4:
        eevee = shot("eevee")
        mask, mean, p90 = compare(st["live"], eevee)
        report.update(asset_pixels=int(mask.sum()), mean_diff=mean, p90=p90)
        save_side(st["live"], eevee, "live_vs_eevee")
        check("looks like EEVEE (mean difference over the assets)", mask.sum() > 1000 and mean < 12.0,
              f"{mean:.1f}/255 over {int(mask.sum()):,} px; side by side: live_vs_eevee.png")
        st["eevee"] = eevee
        if not st["suns"]:
            check("the scene has a sun", False)
            finish()
        for s in st["suns"]:
            s.data.use_shadow = True
        st["t"] = time.perf_counter()
        return 3.0                             # EEVEE samples accumulate, shadows included
    if k == 5:
        st["eevee_sh"] = shot("eevee_shadows")
        scene.vgeo_live_draw = True
        st["t"] = time.perf_counter()
        return 1.0
    if k == 6:
        return wait(lambda: settled_live() and livedraw._state.get("shadowed"), 6, 1.0)
    if k == 7:
        live_sh = shot("live_draw_shadows")
        st["live_sh"] = live_sh
        live, eevee, eevee_sh = st["live"], st["eevee"], st["eevee_sh"]
        check("Live Draw renders a shadow map", livedraw._state.get("shadowed") and shadows.stats["renders"] > 0,
              f"{shadows.stats['renders']} cascade renders, {shadows.stats['cascades']} cascade(s), "
              f"{shadows.stats['tris']:,} triangles in the last")
        save_side(live_sh, eevee_sh, "live_vs_eevee_shadows")
        mask, mean, p90 = compare(live_sh, eevee_sh)
        report.update(shadow_mean_diff=mean, shadow_p90=p90)
        check("with shadows, looks like EEVEE with shadows (mean difference over the assets)",
              mask.sum() > 1000 and mean < 12.0,
              f"{mean:.1f}/255 over {int(mask.sum()):,} px; side by side: live_vs_eevee_shadows.png")
        # where shadows darken the image, in Live Draw and in EEVEE
        dl = (lum(live_sh) < 0.75 * lum(live) - 4) & mask
        de = (lum(eevee_sh) < 0.75 * lum(eevee) - 4) & mask
        inter, union = int((dl & de).sum()), int((dl | de).sum())
        iou = inter / union if union else 0.0
        report.update(shadow_pixels_live=int(dl.sum()), shadow_pixels_eevee=int(de.sum()), shadow_iou=iou)
        check("shadows land where EEVEE's do (overlap of the darkened pixels)", union > 500 and iou > 0.6,
              f"IoU {iou:.2f}: Live Draw {int(dl.sum()):,} px, EEVEE {int(de.sum()):,} px")
        save_side(np.where(dl[..., None], [255, 60, 60], live_sh * 0.5),
                  np.where(de[..., None], [60, 255, 60], eevee_sh * 0.5), "shadow_masks")
        report["shadow_levels"] = list(shadows.stats["levels"])
        if generated["on"]:
            rc = to_px(shadow_point())
            off, on, ev = patch(live, rc), patch(live_sh, rc), patch(eevee_sh, rc)
            lit = max(patch(live_sh, to_px(p)) for p in lit_points())
            report.update(shadow_spot_lum=dict(live_off=off, live_on=on, eevee_on=ev, live_lit=lit))
            check("the ground under the sphere's shadow is darker", on < 0.8 * off and on < 0.8 * lit,
                  f"luminance {off:.0f} -> {on:.0f} with shadows (EEVEE {ev:.0f}; lit ground beside it "
                  f"{lit:.0f}) at pixel {rc}")
            rc = to_px(slab_shadow_point())
            off, on, ev = patch(live, rc, 4), patch(live_sh, rc, 4), patch(eevee_sh, rc, 4)
            ev_off = patch(eevee, rc, 4)
            report.update(slab_spot_lum=dict(live_off=off, live_on=on, eevee_off=ev_off, eevee_on=ev),
                          receiver_triangles=livedraw._state.get("received", 0))
            rc2 = to_px(pillar_points()[1])
            p_off, p_on, p_ev, p_ev_off = (patch(im, rc2, 3) for im in (live, live_sh, eevee_sh, eevee))
            report.update(pillar_spot_lum=dict(live_off=p_off, live_on=p_on, eevee_off=p_ev_off, eevee_on=p_ev))
            check("an ordinary mesh casts onto virtualized ones, as dark as in EEVEE",
                  p_on < 0.8 * p_off and abs(p_on - p_ev) < 0.15 * p_ev_off,
                  f"luminance {p_off:.0f} -> {p_on:.0f} with shadows (EEVEE {p_ev_off:.0f} -> {p_ev:.0f}) "
                  f"at pixel {rc2}")
            check("an ordinary mesh receives the shadow too, as dark as in EEVEE",
                  on < 0.8 * off and abs(on - ev) < 0.15 * ev_off,
                  f"luminance {off:.0f} -> {on:.0f} with shadows (EEVEE {ev_off:.0f} -> {ev:.0f}) at pixel {rc}")
        space.overlay.show_overlays = True     # Blender's own depth this time
        return 1.0
    if k == 8:
        img = shot("live_draw_shadows_overlays")
        space.overlay.show_overlays = False
        if generated["on"]:
            pts = [shadow_point(), slab_shadow_point(), pillar_points()[1]]
            a = [patch(st["live_sh"], to_px(p), 3) for p in pts]
            b = [patch(img, to_px(p), 3) for p in pts]
            worst = max(abs(x - y) for x, y in zip(a, b))
            check("the same shadows with overlays shown (Blender's depth) as hidden (Live Draw's own)",
                  worst < 6.0, "luminance " + ", ".join(f"{x:.0f}/{y:.0f}" for x, y in zip(a, b)))
        scene.vgeo_live_shadows = False        # the scene toggle switches them off
        return 1.0
    if k == 9:
        img = shot("live_draw_shadows_toggled_off")
        d = float(np.abs(img - st["live"]).max(2)[asset_mask(img)].mean())
        check("the Shadows toggle switches them off", d < 1.5 and not livedraw._state.get("shadowed"),
              f"{d:.2f}/255 from the unshadowed image")
        scene.vgeo_live_shadows = True
        for s in st["suns"]:                    # ... and so does the light's own Shadow toggle
            s.data.use_shadow = False
        return 1.0
    if k == 10:
        img = shot("live_draw_light_shadow_off")
        d = float(np.abs(img - st["live"]).max(2)[asset_mask(img)].mean())
        check("the light's Shadow option switches them off", d < 1.5 and not livedraw._state.get("shadowed"),
              f"{d:.2f}/255 from the unshadowed image")
        for s in st["suns"]:
            s.data.use_shadow = True
        # orbit in Solid with shadows, then without
        space.shading.type = 'SOLID'
        r3.view_perspective = 'PERSP'
        frames.clear()
        st["h"] = bpy.types.SpaceView3D.draw_handler_add(_count, (), 'WINDOW', 'POST_PIXEL')
        st["renders0"] = shadows.stats["renders"]
        st["t"] = time.perf_counter()
        st["orbit0"] = k + 1
        return 0.5
    o = k - st["orbit0"]
    if o == 3:     # how Solid shading looks with shadows (no check: workbench lighting is not the sun's)
        shot("live_draw_solid_shadows")
    if o < ORBIT:  # orbit, shadows on
        r3.view_rotation.rotate(Euler((0, 0, 0.01)))
        area.tag_redraw()
        return 0.0
    if o == ORBIT:
        on = orbit_stats("orbit_shadows")
        check("draw handler stays cheap with shadows", on["draw_handler_ms"] < 4.0,
              f"{on['draw_handler_ms']:.2f} ms median (p95 {on['draw_handler_ms_p95']:.2f}), shadow update "
              f"{on['shadow_update_ms']:.2f} ms median, {on['shadow_renders']} cascade renders in "
              f"{on['frames']} frames at {on['shadow_render_ms']:.2f} ms")
        check("frames while orbiting with shadows", on["frame_ms_median"] < 33.0,
              f"median {on['frame_ms_median']:.1f} ms, p95 {on['frame_ms_p95']:.1f} ms")
        scene.vgeo_live_shadows = False
        frames.clear()
        st["renders0"] = shadows.stats["renders"]
        return 0.5
    if o < 2 * ORBIT + 1:  # orbit back, shadows off
        r3.view_rotation.rotate(Euler((0, 0, -0.01)))
        area.tag_redraw()
        return 0.0
    if o == 2 * ORBIT + 1:
        off = orbit_stats("orbit_no_shadows")
        on = report["orbit_shadows"]
        check("draw handler stays cheap", off["draw_handler_ms"] < 4.0, f"{off['draw_handler_ms']:.2f} ms")
        check("frames while orbiting", off["frame_ms_median"] < 33.0,
              f"median {off['frame_ms_median']:.1f} ms, p95 {off['frame_ms_p95']:.1f} ms")
        report["shadow_cost"] = dict(handler_ms=on["draw_handler_ms"] - off["draw_handler_ms"],
                                     frame_ms=on["frame_ms_median"] - off["frame_ms_median"])
        print(f"shadow cost while orbiting: draw handler {on['draw_handler_ms']:.2f} vs "
              f"{off['draw_handler_ms']:.2f} ms, frame {on['frame_ms_median']:.1f} vs "
              f"{off['frame_ms_median']:.1f} ms median", flush=True)
        scene.vgeo_live_shadows = True
        bpy.types.SpaceView3D.draw_handler_remove(st["h"], 'WINDOW')
        return 1.0
    if o < 2 * ORBIT + 5:       # shadows back on: the first redraws bring the map up to date
        area.tag_redraw()
        st["renders0"] = shadows.stats["renders"]
        return 0.1
    if o < 2 * ORBIT + 15:      # then a still view renders no shadow map at all
        area.tag_redraw()
        return 0.05
    if o == 2 * ORBIT + 15:
        n = shadows.stats["renders"] - st["renders0"]
        check("a still view re-renders no shadow map", n == 0, f"{n} cascade renders over 10 redraws")
        # final render: EEVEE with real meshes (render_pre fills the chunks), whatever Live Draw does
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 480, 270, 100
        scene.render.filepath = os.path.join(OUT, "final_render.png")
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(scene.render.filepath)
        px = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], 4)[..., :3]
        bpy.data.images.remove(img)
        corner = np.median(px[:6, :6].reshape(-1, 3), axis=0)
        covered = float((np.abs(px - corner).max(2) > 0.05).mean())
        filled = 0
        for p in stream.proxies(scene):
            if p.vgeo.collection:
                filled += sum(len(o.data.polygons) for o in p.vgeo.collection.objects)
        report.update(render_coverage=covered)
        check("final render is EEVEE with the real meshes", covered > 0.05 and filled > 0,
              f"{100 * covered:.0f}% of the frame is the assets, {filled:,} triangles in the chunk meshes")
        if generated["on"]:
            from bpy_extras.object_utils import world_to_camera_view
            sp = world_to_camera_view(scene, scene.camera, shadow_point())
            def at(co):
                q = world_to_camera_view(scene, scene.camera, co)
                y, x = int(q.y * px.shape[0]), int(q.x * px.shape[1])
                y, x = min(max(y, 3), px.shape[0] - 4), min(max(x, 3), px.shape[1] - 4)
                return float(np.median(lum(px[y - 3:y + 4, x - 3:x + 4].reshape(-1, 3) * 255)))
            near = at(shadow_point())
            lit = max(at(p) for p in lit_points())
            check("the final render has the sphere's shadow (EEVEE's own)", near < 0.8 * lit,
                  f"luminance {near:.0f} in the shadow, {lit:.0f} beside it")
        for s, was in zip(st["suns"], st["shadow_was"]):
            s.data.use_shadow = was
        finish()
    return 0.0


def safe():
    try:
        return tick()
    except Exception:
        import traceback
        traceback.print_exc()
        check("no Python errors", False)
        finish()


bpy.app.timers.register(safe, first_interval=1.0)
