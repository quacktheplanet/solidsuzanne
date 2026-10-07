"""Live Draw in a real window (needs the UI, not -b): it draws, looks like EEVEE, is fast, renders stay EEVEE.

    blender [scene.blend] --python tests/live_draw_check.py -- [--out DIR]

Without a scene it builds one: a 1.3M-triangle displaced sphere with an image base colour and an image
normal map (both generated), a sun (no shadows, so both sides compare like for like) and a plain world.
With a scene (e.g. a virtualized Poly Haven scan) it uses the first VGEO proxy and the scene camera.

1. Live Draw draws the proxy in Material Preview (scene lights and world), triangles counted
2. the same view drawn by EEVEE from the Blender-mesh path (Live Draw off) looks the same: mean
   difference under 12/255 over the asset's pixels. Why not tighter: EEVEE adds what Live Draw doesn't
   (screen-space effects, its own specular and world lighting model, AgX done exactly instead of
   approximated); a missing texture, a flipped normal map or wrong colour management lands far above it
   (a flat grey asset differs by 40-70/255)
3. drawing stays fast: the draw handler's own time under 4 ms, frames while orbiting under 33 ms median
4. a final render (F12 path) still goes through EEVEE with real meshes: the asset is in the render
"""
import json
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
from vgeo import livedraw, stream  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else tempfile.mkdtemp(prefix="vgeo_live_")
os.makedirs(OUT, exist_ok=True)
bpy.context.preferences.view.show_splash = False
try:
    vgeo.register()
except ValueError:
    pass
results, report = [], {}


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def finish():
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed", flush=True)
    with open(os.path.join(OUT, "live_draw_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    os._exit(1 if failed else 0)


def build_scene():
    """A displaced sphere with generated base colour and normal map images."""
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    W = 256
    yy, xx = np.mgrid[0:W, 0:W] / W
    base = np.ones((W, W, 4), np.float32)
    check_ = ((np.floor(xx * 8) + np.floor(yy * 8)) % 2)[..., None]
    base[..., :3] = np.where(check_ > 0, [0.75, 0.35, 0.15], [0.2, 0.45, 0.7])
    img = bpy.data.images.new("live_base", W, W)
    img.pixels.foreach_set(base.ravel())
    h = np.sin(xx * 40) * np.sin(yy * 40)
    nx, ny = np.gradient(h)
    nrm = np.stack([0.5 - nx * 2, 0.5 - ny * 2, np.ones_like(h), np.ones_like(h)], -1).astype(np.float32)
    nimg = bpy.data.images.new("live_normal", W, W, float_buffer=True)
    nimg.colorspace_settings.name = "Non-Color"
    nimg.pixels.foreach_set(np.clip(nrm, 0, 1).ravel())
    import bmesh
    me = bpy.data.meshes.new("live_sphere")
    bm = bmesh.new()
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
    cam.location = (5.5, -5.5, 3.0)
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "live_scene.blend"))
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    win, area = area3d()
    with bpy.context.temp_override(window=win, area=area, active_object=ob, object=ob):
        rc = bpy.ops.vgeo.virtualize()
    check("virtualize the textured asset", rc == {'FINISHED'})


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


frames = []
_last = [time.perf_counter()]


def _count():
    now = time.perf_counter()
    frames.append((now - _last[0]) * 1000)
    _last[0] = now


st = {"k": 0, "t": 0.0}


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
        space.overlay.show_overlays = False
        r3.view_perspective = 'CAMERA'
        scene.vgeo_live_draw = True
        st["t"] = time.perf_counter()
        return 0.5
    if k == 1:   # let the cut land
        if (not livedraw.is_active() or livedraw.drawn_triangles() == 0) and time.perf_counter() - st["t"] < 60:
            st["k"] = 1
            return 0.25
        if any(c.pending is not None for c in livedraw._cuts.values()) and time.perf_counter() - st["t"] < 60:
            st["k"] = 1
            return 0.25
        return 1.0
    if k == 2:
        st["live"] = shot("live_draw")
        tris = livedraw.drawn_triangles()
        report["live_triangles"] = tris
        check("Live Draw draws the asset", livedraw.is_active() and tris > 0, f"{tris:,} triangles")
        scene.vgeo_live_draw = False          # the Blender-mesh path, drawn by EEVEE
        st["t"] = time.perf_counter()
        return 1.0
    if k == 3:   # wait for the EEVEE path to land and settle
        p = stream.proxies(scene)[0]
        rt = stream._runtimes.get(p.vgeo.uid)
        if (rt is None or rt.updates == 0 or not rt.valid.all()) and time.perf_counter() - st["t"] < 60:
            st["k"] = 3
            return 0.25
        return 3.0                             # EEVEE samples accumulate
    if k == 4:
        eevee = shot("eevee")
        live = st["live"]
        bg = np.median(np.concatenate([eevee[:8].reshape(-1, 3), eevee[-8:].reshape(-1, 3)]), axis=0)
        mask = (np.abs(eevee - bg).max(2) > 12) | (np.abs(live - bg).max(2) > 12)
        d = np.abs(live - eevee).max(2)[mask]
        mean = float(d.mean()) if d.size else 999.0
        report.update(asset_pixels=int(mask.sum()), mean_diff=mean, p90=float(np.percentile(d, 90)) if d.size else 0)
        side = np.concatenate([live, eevee], axis=1)
        out = bpy.data.images.new("side", side.shape[1], side.shape[0])
        rgba = np.ones((side.shape[0], side.shape[1], 4), np.float32)
        rgba[..., :3] = side / 255.0
        out.pixels.foreach_set(rgba.ravel())
        out.filepath_raw = os.path.join(OUT, "live_vs_eevee.png")
        out.file_format = 'PNG'
        out.save()
        check("looks like EEVEE (mean difference over the asset)", mask.sum() > 1000 and mean < 12.0,
              f"{mean:.1f}/255 over {int(mask.sum()):,} px; side by side: live_vs_eevee.png")
        scene.vgeo_live_draw = True
        space.shading.type = 'SOLID'
        r3.view_perspective = 'PERSP'
        frames.clear()
        st["h"] = bpy.types.SpaceView3D.draw_handler_add(_count, (), 'WINDOW', 'POST_PIXEL')
        st["t"] = time.perf_counter()
        return 0.5
    if k < 130:  # orbit
        r3.view_rotation.rotate(Euler((0, 0, 0.01)))
        area.tag_redraw()
        return 0.0
    if k == 130:
        f = np.array(frames[5:]) if len(frames) > 8 else np.array([999.0])
        dm = livedraw._state["frame_ms"][-60:]
        draw_ms = float(np.median(dm)) if dm else 999.0
        report.update(frame_ms_median=float(np.median(f)), frame_ms_p95=float(np.percentile(f, 95)),
                      draw_handler_ms=draw_ms)
        check("draw handler stays cheap", draw_ms < 4.0, f"{draw_ms:.2f} ms")
        check("frames while orbiting", np.median(f) < 33.0,
              f"median {np.median(f):.1f} ms, p95 {np.percentile(f, 95):.1f} ms")
        # final render: EEVEE with real meshes (render_pre fills the chunks), whatever Live Draw does
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 480, 270, 100
        scene.render.filepath = os.path.join(OUT, "final_render.png")
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(scene.render.filepath)
        px = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], 4)[..., :3]
        corner = np.median(px[:6, :6].reshape(-1, 3), axis=0)
        covered = float((np.abs(px - corner).max(2) > 0.05).mean())
        p = stream.proxies(scene)[0]
        filled = sum(len(o.data.polygons) for o in p.vgeo.collection.objects) if p.vgeo.collection else 0
        report.update(render_coverage=covered)
        check("final render is EEVEE with the real meshes", covered > 0.05 and filled > 0,
              f"{100 * covered:.0f}% of the frame is the asset, {filled:,} triangles in the chunk meshes")
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
