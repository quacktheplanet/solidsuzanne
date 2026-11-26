"""Memory over a long render: renders N frames in one session along a camera path and logs memory per frame.

Per frame it logs the process's resident and private memory, Blender's datablock counts (meshes,
objects, collections, meshes with no users), and the native runtime's own memory (mapped and heap
bytes of every open .vgeo handle), so growth can be pinned on the add-on, on Blender or on the
render engine.

Usage:
  blender -b scene.blend --python tools/render_memory_bench.py -- --cameras cams.json
        [--frames 100] [--engine EEVEE|CYCLES|WORKBENCH] [--res 1280x720] [--samples 4]
        [--mode stills|animation] [--freeze] [--no-addon] [--gpu] [--json out.json]

cams.json is a list of {"pos", "target", "fov"} (optionally "frame"); the camera is keyed through them,
evenly over the frames. --mode stills renders one still per frame (frame_set + render), the way the
demo film was made; --mode animation renders the frames as one animation render. --freeze freezes
every instancer and proxy after the first frame (the add-on stops changing geometry: whatever still
grows is Blender's or the engine's). --no-addon renders without the add-on registered at all.
"""

import ctypes
import json
import math
import os
import sys
import time

import bpy
from mathutils import Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def flag(name):
    return name in argv


# ---------------------------------------------------------------- process memory

def process_memory():
    """(resident, private) bytes of this process."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return int(pmc.WorkingSetSize), int(pmc.PrivateUsage)
    rss = priv = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("RssAnon:"):
                    priv = int(line.split()[1]) * 1024
    except OSError:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return rss, priv


def native_memory():
    """(mapped, heap) bytes over every open .vgeo handle the add-on holds."""
    mapped = heap = 0
    try:
        from vgeo import stream, instances
    except ImportError:
        return 0, 0
    seen = set()
    handles = [rt.asset for rt in stream._runtimes.values() if getattr(rt, "asset", None) is not None]
    handles += [a for a in getattr(instances, "_open_assets", {}).values()]
    for a in handles:
        if id(a) in seen:
            continue
        seen.add(id(a))
        try:
            m, h = a.memory()
            mapped += m
            heap += h
        except Exception:
            pass
    return mapped, heap


def sample(frame, t_render):
    rss, priv = process_memory()
    mapped, heap = native_memory()
    meshes = bpy.data.meshes
    return {
        "frame": frame,
        "render_s": round(t_render, 2),
        "rss_mb": round(rss / 2**20, 1),
        "private_mb": round(priv / 2**20, 1),
        "meshes": len(meshes),
        "orphan_meshes": sum(1 for m in meshes if m.users == 0),
        "mesh_tris_m": round(sum(len(m.loop_triangles) for m in meshes if m.users) / 1e6, 2)
        if flag("--count-tris") else None,
        "objects": len(bpy.data.objects),
        "collections": len(bpy.data.collections),
        "native_mapped_mb": round(mapped / 2**20, 1),
        "native_heap_mb": round(heap / 2**20, 1),
    }


# ---------------------------------------------------------------- camera path

def camera_path(scene, cams, frames):
    cam = bpy.data.objects.new("VGEO Memory Bench Cam", bpy.data.cameras.new("VGEO Memory Bench Cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens_unit = 'FOV'
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.rotation_mode = 'QUATERNION'
    n = len(cams)
    for k, spec in enumerate(cams):
        f = scene.frame_start + round(k * (frames - 1) / max(1, n - 1))
        cam.location = Vector(spec["pos"])
        cam.rotation_quaternion = (Vector(spec["target"]) - Vector(spec["pos"])).to_track_quat('-Z', 'Y')
        cam.data.angle = math.radians(spec.get("fov", 50))
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        cam.data.keyframe_insert("lens", frame=f)
    return cam


def freeze_all():
    """Stop the add-on changing anything: freeze its objects and remove its render handlers (final
    renders cut regardless of Freeze)."""
    from vgeo import stream
    h = bpy.app.handlers
    for lst, fn in ((h.render_pre, stream._on_render_pre), (h.frame_change_post, stream._on_frame_change_post)):
        while fn in lst:
            lst.remove(fn)
    for ob in bpy.data.objects:
        if getattr(ob, "vgeo_inst", None) is not None and ob.vgeo_inst.uid:
            ob.vgeo_inst.freeze = True
        if getattr(ob, "vgeo", None) is not None and ob.vgeo.uid:
            ob.vgeo.freeze = True


def main():
    use_addon = not flag("--no-addon")
    if use_addon:
        import vgeo
        if not hasattr(bpy.types.Object, "vgeo"):
            vgeo.register()
    scene = bpy.context.scene
    frames = int(arg("--frames", 100))
    engine = arg("--engine", "EEVEE").upper()
    scene.render.engine = {"EEVEE": "BLENDER_EEVEE", "CYCLES": "CYCLES", "WORKBENCH": "BLENDER_WORKBENCH"}.get(
        engine, engine)
    w, h = (int(x) for x in arg("--res", "1280x720").split("x"))
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    samples = int(arg("--samples", 4))
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = False
        if flag("--gpu"):
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for backend in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = backend
                    prefs.get_devices()
                    if any(d.type == backend for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type == backend
                        scene.cycles.device = "GPU"
                        break
                except TypeError:
                    continue
    elif engine == "EEVEE":
        scene.eevee.taa_render_samples = samples
    cams = json.load(open(arg("--cameras")))
    scene.frame_start, scene.frame_end = 1, frames
    camera_path(scene, cams, frames)
    out_dir = arg("--outdir", bpy.app.tempdir)
    scene.render.filepath = os.path.join(out_dir, "vgeo_mem_")
    scene.render.image_settings.file_format = 'PNG'
    log = []
    t0 = [time.perf_counter()]
    mode = arg("--mode", "stills")
    print(f"memory bench: {frames} frames, {engine} {w}x{h}, {samples} samples, mode {mode}, "
          f"add-on {'on' if use_addon else 'off'}{', frozen after frame 1' if flag('--freeze') else ''}")

    def record(frame):
        now = time.perf_counter()
        s = sample(frame, now - t0[0])
        t0[0] = now
        log.append(s)
        print("MEM " + json.dumps(s), flush=True)

    if mode == "animation":
        def post(sc, _dg=None):
            record(sc.frame_current)
            if flag("--freeze") and sc.frame_current == sc.frame_start and use_addon:
                freeze_all()
        bpy.app.handlers.render_post.append(post)
        bpy.ops.render.render(animation=True)
        bpy.app.handlers.render_post.remove(post)
    else:
        for f in range(1, frames + 1):
            scene.frame_set(f)
            bpy.ops.render.render(write_still=flag("--write"))
            record(f)
            if f == 1 and flag("--freeze") and use_addon:
                freeze_all()

    first, last = log[min(1, len(log) - 1)], log[-1]   # frame 1 warms caches: growth from frame 2
    growth = (last["rss_mb"] - first["rss_mb"]) / max(1e-9, first["rss_mb"]) * 100
    pgrowth = (last["private_mb"] - first["private_mb"]) / max(1e-9, first["private_mb"]) * 100
    # steady state: the camera path visits new places (new pages of the mapped assets, new cuts) early
    # on; a leak shows as a slope over the second half
    half = log[len(log) // 2:]
    slope = 0.0
    if len(half) > 2:
        import numpy as np
        f = np.array([x["frame"] for x in half], float)
        slope = float(np.polyfit(f, np.array([x["private_mb"] for x in half], float), 1)[0])
    mid = half[0]
    summary = {"frames": frames, "engine": engine, "mode": mode, "freeze": flag("--freeze"),
               "addon": use_addon, "rss_first_mb": first["rss_mb"], "rss_last_mb": last["rss_mb"],
               "rss_peak_mb": max(s["rss_mb"] for s in log), "rss_growth_pct": round(growth, 2),
               "private_first_mb": first["private_mb"], "private_last_mb": last["private_mb"],
               "private_growth_pct": round(pgrowth, 2),
               "second_half_private_growth_pct": round((last["private_mb"] - mid["private_mb"])
                                                       / max(1e-9, mid["private_mb"]) * 100, 2),
               "second_half_private_mb_per_frame": round(slope, 2),
               "mean_render_s": round(sum(s["render_s"] for s in log[1:]) / max(1, len(log) - 1), 2)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    if arg("--json"):
        with open(arg("--json"), "w") as f:
            json.dump({"summary": summary, "frames": log}, f, indent=1)


main()
