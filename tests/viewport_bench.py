"""Interactive viewport benchmark (needs the Blender UI, not -b).

    blender path/to/scene.blend --python tests/viewport_bench.py -- --out DIR [--seconds 10]
        [--motion fly|orbit|path] [--cameras cams.json] [--offscreen COARSEN|SHADOWS|CULL] [--slots N]
        [--phases solid_frozen,solid_stream,eevee_frozen,eevee_stream]

--cameras (implies --motion path) flies through the cameras of a cams.json ({"pos", "target", "fov"})
over --seconds, the way tools/instancing_bench.py places them. --offscreen and --slots set every
instancer's Off-screen mode and Streamed Copies first.

Flies the 3D viewport from the scene camera forward while the VGEO live loop
streams, once in Solid and once in Material Preview (EEVEE). Records real
frame times (draw handler timestamps) and stream updates, saves screenshots
and a JSON report, then quits.
"""

import json
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))
import vgeo  # noqa: E402
from vgeo import stream  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(REPO, "build-demo")
SECONDS = float(argv[argv.index("--seconds") + 1]) if "--seconds" in argv else 10.0
SPEED = float(argv[argv.index("--speed") + 1]) if "--speed" in argv else 25.0
os.makedirs(OUT, exist_ok=True)

CAMS = None
if "--cameras" in argv:
    with open(argv[argv.index("--cameras") + 1]) as _f:
        CAMS = json.load(_f)

state = {"frames": [], "phase": None, "t0": 0.0, "report": {}, "updates0": 0, "tick_s": 0.0, "ticks": 0}
# (shading, label, freeze): frozen phases draw the same geometry without streaming = baseline
PHASES = [("SOLID", "solid_frozen", True), ("SOLID", "solid_stream", False),
          ("MATERIAL", "eevee_frozen", True), ("MATERIAL", "eevee_stream", False)]
if "--phases" in argv:
    _want = argv[argv.index("--phases") + 1].split(",")
    PHASES = [p for p in PHASES if p[1] in _want]


_dg = {"n": 0, "ids": {}}


def _on_dg(scene, depsgraph):
    _dg["n"] += 1
    for u in depsgraph.updates:
        key = f"{type(u.id).__name__}:{u.id.name[:24]}{' geo' if u.is_updated_geometry else ''}" \
              f"{' xf' if u.is_updated_transform else ''}{' sh' if u.is_updated_shading else ''}"
        _dg["ids"][key] = _dg["ids"].get(key, 0) + 1


def timed_tick():
    t0 = time.perf_counter()
    r = stream._tick()
    state["tick_s"] += time.perf_counter() - t0
    state["ticks"] += 1
    return r


def set_freeze(on):
    from vgeo import instances
    for o in stream.proxies():
        if not o.vgeo.slot_of:   # streamed copies follow their instancer
            o.vgeo.freeze = on
    for o in instances.instancers():
        o.vgeo_inst.freeze = on


def instancer_stats():
    from vgeo import instances
    return {o.name: {"shown_triangles": o.vgeo_inst.shown_triangles, "offscreen": getattr(o.vgeo_inst, "offscreen", "COARSEN"),
                     "copies": len(o.data.vertices)} for o in instances.instancers()}


def view3d():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                region = next(r for r in area.regions if r.type == 'WINDOW')
                return win, area, region, area.spaces.active
    return None


def _draw():
    state["frames"].append(time.perf_counter())
    state.setdefault("swaps_at", []).append(stream.swap_count)
    for o in stream.proxies():
        rt = stream._runtimes.get(o.vgeo.uid)
        if rt is not None and getattr(rt, "swap_dg_ms", None) is not None:
            state.setdefault("swap_dg", []).append(round(rt.swap_dg_ms))
            rt.swap_dg_ms = None


MOTION = argv[argv.index("--motion") + 1] if "--motion" in argv else ("path" if CAMS else "fly")


def path_matrix(t):
    """Camera-to-world matrix and horizontal FOV along the cameras of --cameras, t in [0, 1]."""
    n = len(CAMS)
    x = min(max(t, 0.0), 1.0) * (n - 1)
    k = min(int(x), n - 2) if n > 1 else 0
    f = x - k if n > 1 else 0.0
    a, b = CAMS[k], CAMS[min(k + 1, n - 1)]
    pos = Vector(a["pos"]).lerp(Vector(b["pos"]), f)
    qa = (Vector(a["target"]) - Vector(a["pos"])).to_track_quat('-Z', 'Y')
    qb = (Vector(b["target"]) - Vector(b["pos"])).to_track_quat('-Z', 'Y')
    fov = a.get("fov", 50) * (1 - f) + b.get("fov", 50) * f
    return Matrix.Translation(pos) @ qa.slerp(qb, f).to_matrix().to_4x4(), fov


def set_view(space, t):
    """fly: scene camera moving forward and yawing. orbit: circling a point ahead of the camera."""
    if MOTION == "path":
        import math
        m, fov = path_matrix(t / SECONDS)
        rv3d = space.region_3d
        rv3d.view_perspective = 'PERSP'
        rv3d.view_matrix = m.inverted()
        # the viewport frames lenses on a 72 mm sensor (36 mm with zoom 2), fitted to the wider side
        space.lens = 36.0 / math.tan(math.radians(fov) * 0.5)
        return
    cam = bpy.context.scene.camera
    m = cam.matrix_world.copy()
    if MOTION == "orbit":
        fwd = -(m.to_3x3() @ Vector((0, 0, 1)))
        pivot = m.translation + fwd * 600.0
        rot = Matrix.Translation(pivot) @ Matrix.Rotation(0.12 * t, 4, 'Z') @ Matrix.Translation(-pivot)
        space.region_3d.view_perspective = 'PERSP'
        space.region_3d.view_matrix = (rot @ m).inverted()
        return
    fwd = -(m.to_3x3() @ Vector((0, 0, 1)))
    fwd.z = 0
    fwd.normalize()
    yaw = Matrix.Rotation(0.25 * t, 4, 'Z')
    pos = m.translation + fwd * SPEED * t
    world = Matrix.Translation(pos) @ yaw @ m.to_quaternion().to_matrix().to_4x4()
    rv3d = space.region_3d
    rv3d.view_perspective = 'PERSP'
    rv3d.view_matrix = world.inverted()


def proxy_stats():
    out = []
    for o in stream.proxies():
        rt = stream._runtimes.get(o.vgeo.uid)
        if rt and rt.asset:
            out.append({"object": o.name, "triangles": rt.triangles, "updates": rt.updates,
                        "last_build_ms": round(rt.last_ms, 1), "last_latency_ms": round(rt.latency_ms, 1),
                        "last_chunks": rt.last_rebuilt})
    return out


def screenshot(name):
    win, area, region, space = view3d()
    path = os.path.join(OUT, name)
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=path)
    return path


def tick():
    now = time.perf_counter()
    win, area, region, space = view3d()
    if state["phase"] is None:
        # settle: let the first cut stream in before measuring
        if now - state["t0"] < 6.0:
            set_view(space, 0.0)
            return 0.05
        state["phase"] = 0
        begin_phase(space)
        return 0.0
    shade, label, frozen = PHASES[state["phase"]]
    t = now - state["t0"]
    if t < SECONDS:
        set_view(space, t)
        area.tag_redraw()
        return 1.0 / 120.0
    # phase done: stop moving, let the stream catch up, then record
    if "settle" not in state:
        state["settle"] = now
        return 0.1
    rtbusy = any(stream._runtimes[o.vgeo.uid].target is not None for o in stream.proxies()
                 if o.vgeo.uid in stream._runtimes)
    if rtbusy and not frozen and now - state["settle"] < 10:
        return 0.1
    frames = state["frames"]
    sw = state.get("swaps_at", [])[-len(frames):]
    swap_dts = [1000 * (frames[i] - frames[i - 1]) for i in range(1, len(frames))
                if i < len(sw) and sw[i] != sw[i - 1]]
    other = [1000 * (frames[i] - frames[i - 1]) for i in range(1, len(frames))
             if i < len(sw) and sw[i] == sw[i - 1]]
    state["swap_frames"] = {"swap_frame_ms": [round(x) for x in swap_dts][:12],
                            "other_frame_ms_max": round(max(other)) if other else 0}
    dts = sorted(b - a for a, b in zip(frames, frames[1:]))
    stats = proxy_stats()
    top = sorted(_dg["ids"].items(), key=lambda kv: -kv[1])[:8]
    rep = {
        "dg_updates": _dg["n"],
        "dg_top": top,
        "stream_tick_ms_total": round(state["tick_s"] * 1000, 1),
        "stream_ticks": state["ticks"],
        "frames": len(frames),
        "fps_avg": round((len(frames) - 1) / (frames[-1] - frames[0]), 1) if len(frames) > 1 else 0,
        "frame_ms_median": round(1000 * dts[len(dts) // 2], 1) if dts else 0,
        "frame_ms_p95": round(1000 * dts[int(len(dts) * 0.95)], 1) if dts else 0,
        "frame_ms_max": round(1000 * dts[-1], 1) if dts else 0,
        "stream_updates": sum(p["updates"] for p in stats) - state["updates0"],
        "catch_up_s": round(now - state["settle"], 2),
        **state["swap_frames"],
        "swap_depsgraph_ms": state.get("swap_dg", [])[:12],
        "swap_tris": [getattr(stream._runtimes.get(o.vgeo.uid), "swap_tris", 0) for o in stream.proxies()],
        "proxies": stats,
        "instancers": instancer_stats(),
        "region": [region.width, region.height],
    }
    state["report"][label] = rep
    print("VIEWPORT", label, json.dumps(rep), flush=True)
    screenshot(f"viewport_{label}.png")
    del state["settle"]
    state["phase"] += 1
    if state["phase"] >= len(PHASES):
        with open(os.path.join(OUT, "viewport_report.json"), "w") as f:
            json.dump(state["report"], f, indent=2)
        bpy.ops.wm.quit_blender()
        return None
    begin_phase(space)
    return 0.0


def begin_phase(space):
    shade, label, frozen = PHASES[state["phase"]]
    set_freeze(frozen)
    space.shading.type = shade
    state["t0"] = time.perf_counter()
    state["frames"].clear()
    state["swaps_at"] = []
    state["swap_dg"] = []
    _dg["n"] = 0
    _dg["ids"] = {}
    state["tick_s"] = 0.0
    state["ticks"] = 0
    state["updates0"] = sum(p["updates"] for p in proxy_stats())


def start():
    vgeo.register()
    from vgeo import instances
    for o in instances.instancers():
        if "--offscreen" in argv:
            if hasattr(o.vgeo_inst, "offscreen"):
                o.vgeo_inst.offscreen = argv[argv.index("--offscreen") + 1]
        if "--slots" in argv:
            o.vgeo_inst.stream_slots = int(argv[argv.index("--slots") + 1])
    stream.DIAG_SWAP = "--diag" in argv
    # time the live loop: swap its timer for a timed wrapper
    if bpy.app.timers.is_registered(stream._tick):
        bpy.app.timers.unregister(stream._tick)
    bpy.app.timers.register(timed_tick, first_interval=0.1, persistent=True)
    win, area, region, space = view3d()
    space.overlay.show_overlays = False
    space.clip_start = 0.1
    space.clip_end = 10000
    bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_PIXEL')
    bpy.app.handlers.depsgraph_update_post.append(_on_dg)
    state["t0"] = time.perf_counter()
    bpy.app.timers.register(tick, first_interval=1.0)
    return None


bpy.app.timers.register(start, first_interval=0.5)
