"""Live Draw sun shadows on a real scene (needs the UI, not -b): looks against EEVEE, and what they cost.

    blender scene.blend --python tests/live_shadow_bench.py -- --out DIR [--seconds 8] [--speed 8] [--sync]

Turns Shadow on for the scene's suns, then:
1. the camera view in Material Preview, Live Draw with shadows and EEVEE with shadows, side by side
   (shadows_vs_eevee.png), plus Live Draw without shadows (live_no_shadows.png)
2. flies the view forward from the camera (as viewport_bench.py) in Solid, with shadows on and then off,
   and records frame times, the draw handler's own time, the shadow map update's share of it, and how
   many cascades were rendered at what cost (with --sync the CPU waits for the GPU after each cascade, so
   that cost is the real render time; frame times are then pessimistic)
Writes shadow_bench.json.
"""
import json
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))
import vgeo  # noqa: E402
from vgeo import livedraw, shadows, stream  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(REPO, "build-demo")
SECONDS = float(argv[argv.index("--seconds") + 1]) if "--seconds" in argv else 8.0
SPEED = float(argv[argv.index("--speed") + 1]) if "--speed" in argv else 8.0
SYNC = "--sync" in argv
os.makedirs(OUT, exist_ok=True)
bpy.context.preferences.view.show_splash = False
try:
    vgeo.register()
except ValueError:
    pass
report = {"blender": bpy.app.version_string, "scene": bpy.data.filepath}
st = {"k": 0, "t": 0.0, "frames": []}


def view3d():
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    return win, area, area.spaces.active


def shot(name):
    win, area, _space = view3d()
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


def settled_live():
    if not livedraw.is_active() or livedraw.drawn_triangles() == 0:
        return False
    if any(c.pending is not None for c in livedraw._cuts.values()):
        return False
    if any(ls.queue or ls.job is not None for ls in livedraw._levels.values()):
        return False
    return not livedraw._frame.get("busy", True)


def settled_eevee():
    for p in stream.proxies():
        if p.vgeo.slot_of:
            continue
        rt = stream._runtimes.get(p.vgeo.uid)
        if rt is None or rt.updates == 0 or not rt.valid.all():
            return False
    return True


def set_view(space, t):
    cam = bpy.context.scene.camera
    m = cam.matrix_world.copy()
    fwd = -(m.to_3x3() @ Vector((0, 0, 1)))
    fwd.z = 0
    fwd.normalize()
    yaw = Matrix.Rotation(0.25 * t, 4, 'Z')
    pos = m.translation + fwd * SPEED * t
    world = Matrix.Translation(pos) @ yaw @ m.to_quaternion().to_matrix().to_4x4()
    space.region_3d.view_perspective = 'PERSP'
    space.region_3d.view_matrix = world.inverted()


def _count():
    st["frames"].append(time.perf_counter())


PHASES = [("SOLID", True), ("SOLID", False), ("MATERIAL", True), ("MATERIAL", False), ("SOLID", True)]


def record(label):
    if len(st["frames"]) < 3 or not livedraw._state["frame_ms"]:
        raise RuntimeError(f"{label}: Live Draw did not draw (is there a VGEO proxy or instancer?)")
    f = np.diff(np.array(st["frames"])) * 1000.0
    n = len(st["frames"])
    dm = livedraw._state["frame_ms"][-max(1, n):]
    sm = livedraw._state.get("shadow_ms", [])[-max(1, n):]
    renders = shadows.stats["renders"] - st["renders0"]
    rm = shadows.stats["render_ms"][-renders:] if renders else []
    rep = {"frames": int(len(f)), "frame_ms_median": float(np.median(f)), "frame_ms_p95": float(np.percentile(f, 95)),
           "frame_ms_max": float(f.max()), "draw_handler_ms_median": float(np.median(dm)),
           "draw_handler_ms_p95": float(np.percentile(dm, 95)),
           "shadow_update_ms_median": float(np.median(sm)) if sm else 0.0,
           "shadow_update_ms_p95": float(np.percentile(sm, 95)) if sm else 0.0,
           "cascade_renders": renders, "cascade_render_ms_median": float(np.median(rm)) if rm else 0.0,
           "cascade_render_ms_max": float(max(rm)) if rm else 0.0,
           "cascades": shadows.stats["cascades"], "triangles_drawn": livedraw.drawn_triangles(),
           "shadow_triangles_last": shadows.stats["tris"]}
    report[label] = rep
    print("SHADOW_BENCH", label, json.dumps(rep), flush=True)


def tick():
    win, area, space = view3d()
    scene = bpy.context.scene
    k = st["k"]
    st["k"] += 1
    if k == 0:
        space.shading.type = 'MATERIAL'
        space.shading.use_scene_lights = True
        space.shading.use_scene_world = True
        space.overlay.show_overlays = False     # (Live Draw writes the ordinary meshes' depth itself)
        space.show_region_ui = False
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        space.region_3d.view_perspective = 'CAMERA'
        st["suns"] = [o for o in scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN']
        for s in st["suns"]:
            s.data.use_shadow = True
        report["suns"] = [s.name for s in st["suns"]]
        scene.vgeo_live_draw = True
        scene.vgeo_live_shadows = True
        st["t"] = time.perf_counter()
        return 1.0
    if k == 1:
        if not (settled_live() and livedraw._state.get("shadowed")) and time.perf_counter() - st["t"] < 120:
            st["k"] = 1
            return 0.5
        return 2.0
    if k == 2:
        st["live"] = shot("live_shadows")
        report["live_triangles"] = livedraw.drawn_triangles()
        scene.vgeo_live_shadows = False
        return 1.0
    if k == 3:
        shot("live_no_shadows")
        scene.vgeo_live_shadows = True
        scene.vgeo_live_draw = False
        st["t"] = time.perf_counter()
        return 2.0
    if k == 4:
        if not settled_eevee() and time.perf_counter() - st["t"] < 180:
            st["k"] = 4
            return 0.5
        return 20.0                 # EEVEE: shaders, shadows and samples settle
    if k == 5:
        eevee = shot("eevee_shadows")
        save_side(st["live"], eevee, "shadows_vs_eevee")
        live = st["live"]
        bg = np.median(np.concatenate([eevee[:8].reshape(-1, 3), eevee[-8:].reshape(-1, 3)]), axis=0)
        mask = (np.abs(eevee - bg).max(2) > 12) | (np.abs(live - bg).max(2) > 12)
        report["mean_diff_vs_eevee"] = float(np.abs(live - eevee).max(2)[mask].mean())
        scene.vgeo_live_draw = True
        shadows.sync_timing = SYNC
        report["sync"] = SYNC
        st["h"] = bpy.types.SpaceView3D.draw_handler_add(_count, (), 'WINDOW', 'POST_PIXEL')
        st["phase"] = 0
        st["t"] = time.perf_counter()
        return 3.0
    # fly phases
    shade, on = PHASES[st["phase"]]
    if "fly0" not in st:
        space.shading.type = shade
        scene.vgeo_live_shadows = on
        set_view(space, 0.0)
        st["fly0"] = time.perf_counter() + 2.0      # settle at the start pose
        return 0.05
    t = time.perf_counter() - st["fly0"]
    if t < 0:
        set_view(space, 0.0)
        area.tag_redraw()
        st["frames"].clear()
        st["renders0"] = shadows.stats["renders"]
        return 0.05
    if t < SECONDS:
        set_view(space, t)
        area.tag_redraw()
        return 1.0 / 120.0
    label = f"{shade.lower()}_{'shadows' if on else 'no_shadows'}"
    record(label + ("_again" if label in report else ""))
    del st["fly0"]
    st["phase"] += 1
    if st["phase"] >= len(PHASES):
        with open(os.path.join(OUT, "shadow_bench.json"), "w") as f:
            json.dump(report, f, indent=1)
        sys.stdout.flush()
        os._exit(0)
    return 0.0


def safe():
    try:
        return tick()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)


bpy.app.timers.register(safe, first_interval=2.0)
