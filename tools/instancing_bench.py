"""Instancing benchmark: triangles drawn per view, against what a meshlet cut on every copy would draw.

For every camera and every instancer it reports:
  shown      what the add-on draws: whole-copy levels plus streamed copies (the real update path)
  cut        a meshlet cut on every copy, copies outside the view coarsened (the add-on's own rule)
  cut+cull   a meshlet cut on every copy, copies outside the view skipped (what the browser library does)
plus how much of `shown` comes from copies outside the view, and how long the update took.

Usage:
  blender -b scene.blend --python tools/instancing_bench.py -- [--cameras cams.json] [--error 1.0]
        [--res 1280x720] [--offscreen COARSEN,SHADOWS,CULL] [--slots 4,8,16] [--no-ideal] [--json out.json]
  blender -b --factory-startup --python tools/instancing_bench.py -- --synthetic [--json out.json]

cams.json is a list of {"pos": [x, y, z], "target": [x, y, z], "fov": horizontal degrees}. Without it the
scene camera is used. --synthetic builds a test field (a dense rock scattered over a plane) in a temp folder.
--offscreen and --slots run the add-on once per combination; the ideal numbers are computed once per camera.
With the bpy module instead of a Blender binary: python tools/instancing_bench.py [scene.blend] -- ...
"""

import json
import math
import os
import sys
import tempfile
import time

import bpy
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))

import vgeo  # noqa: E402
from vgeo import instances, native, stream  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def flag(name):
    return name in argv


# ---------------------------------------------------------------- scene setup

def synthetic_scene():
    """A dense displaced rock scattered over a plane, saved to a temp folder (assets need a saved file)."""
    out = tempfile.mkdtemp(prefix="vgeo_bench_")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vgeo.register()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=7, radius=1.0)
    rock = bpy.context.object
    tex = bpy.data.textures.new("rock", "CLOUDS")
    tex.noise_scale = 0.35
    disp = rock.modifiers.new("disp", "DISPLACE")
    disp.texture, disp.strength = tex, 0.3
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "bench.blend"))
    bpy.ops.vgeo.virtualize()
    proxy = bpy.context.active_object
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 80, -1))
    ground = bpy.context.active_object
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ground.select_set(True)
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.vgeo.scatter(count=1500, seed=7, scale_min=0.6, scale_max=1.6, align=False)
    bpy.context.active_object.vgeo_inst.render_pixel_error = 1.0
    r = bpy.context.scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = 1280, 720, 100
    proxy.hide_render = True
    proxy.hide_set(True)
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.rotation_euler = (math.radians(55), 0, math.radians(30))
    bpy.context.scene.collection.objects.link(sun)
    bpy.ops.wm.save_mainfile()
    return [{"pos": [0, -25, 1.0], "target": [0, 60, 0], "fov": 60},      # ground level, down the field
            {"pos": [2.5, 8, 0.6], "target": [0, 14, 0], "fov": 50},       # close-up inside the field
            {"pos": [-60, -40, 45], "target": [0, 80, 0], "fov": 55}]      # overview from above


def bench_camera(spec, scene):
    cam = bpy.data.objects.get("VGEO Bench Cam")
    if cam is None:
        cam = bpy.data.objects.new("VGEO Bench Cam", bpy.data.cameras.new("VGEO Bench Cam"))
        scene.collection.objects.link(cam)
    cam.data.lens_unit = 'FOV'
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.angle = math.radians(spec.get("fov", 50))
    cam.data.clip_start = spec.get("clip_start", 0.1)
    cam.location = Vector(spec["pos"])
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = (Vector(spec["target"]) - Vector(spec["pos"])).to_track_quat('-Z', 'Y')
    scene.camera = cam
    bpy.context.view_layer.update()
    return stream.camera_view(scene, bpy.context.evaluated_depsgraph_get())


# ---------------------------------------------------------------- ideal cuts

def local_view(m, view, err, mode):
    v, window, height, persp, clip_start = view
    scale = max(abs(s) for s in m.to_scale()) or 1.0
    cam_local = m.inverted() @ v.inverted().translation
    planes = stream._frustum_planes(window @ v @ m)
    return native.make_view(cam_local, window[1][1], max(clip_start / scale, 1e-6), err / max(1, height),
                            planes=planes, mode=mode, offscreen_scale=8.0)


def ideal(inst, asset, view, err):
    """Triangles of a meshlet cut on every copy: (off-screen coarsened, off-screen skipped)."""
    coarsen = cull = 0
    for i in range(len(inst.data.vertices)):
        m = inst.matrix_world @ instances.placement_matrix(inst, i)
        asset.select([local_view(m, view, err, "COARSEN")])
        coarsen += int(asset.last.triangles)
        asset.select([local_view(m, view, err, "CULL")])
        cull += int(asset.last.triangles)
    return coarsen, cull


def offscreen_share(inst, view, err, mode):
    """Triangles the add-on draws for whole-copy levels of copies outside the view."""
    import numpy as np
    errors, tris = instances._level_tables(inst)
    co, sc = instances._points(inst)
    mw = np.array(inst.matrix_world, dtype=np.float64)
    world = co @ mw[:3, :3].T + mw[:3, 3]
    radius = float(inst.get("vgeo_radius", 1.0)) * sc * max(abs(x) for x in inst.matrix_world.to_scale())
    planes = np.array(stream._frustum_planes(view[1] @ view[0]), dtype=np.float64)
    inside = np.all(world @ planes[:, :3].T + planes[:, 3] >= -radius[:, None], axis=1)
    a = inst.data.attributes.get(instances.LEVEL_ATTR)
    lv = np.zeros(len(co), np.int32)
    if a is not None:
        a.data.foreach_get("value", lv)
    streamed = instances._streamed_flags(inst)
    off = ~inside & ~streamed & (lv >= 0)
    return int(tris[lv[off]].sum()), int((~inside).sum())


# ---------------------------------------------------------------- main

def main():
    synthetic = flag("--synthetic")
    if synthetic:
        cams = synthetic_scene()
    else:
        vgeo.register()
        pre = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
        blend = next((a for a in pre if a.endswith(".blend")), None)
        if blend and os.path.abspath(bpy.data.filepath or "") != os.path.abspath(blend):
            bpy.ops.wm.open_mainfile(filepath=blend)   # bpy-module runs: Blender did not open it
        cams_path = arg("--cameras")
        cams = json.load(open(cams_path)) if cams_path else None
    scene = bpy.context.scene
    res = arg("--res")
    if res:
        scene.render.resolution_x, scene.render.resolution_y = (int(x) for x in res.split("x"))
        scene.render.resolution_percentage = 100
    insts = [i for i in instances.instancers(scene) if not i.hide_render]
    if not insts:
        print("no instancers in this scene")
        return
    modes = [m.strip().upper() for m in arg("--offscreen", "").split(",") if m.strip()] or [None]
    slot_counts = [int(s) for s in arg("--slots", "").split(",") if s.strip()] or [None]
    assets = {i.name: native.Asset(stream.asset_path(i.vgeo_inst.source)) for i in insts}
    rows = []
    n_cams = len(cams) if cams else 1
    for ci in range(n_cams):
        view = bench_camera(cams[ci], scene) if cams else stream.camera_view(scene)
        for inst in insts:
            err = float(arg("--error", inst.vgeo_inst.render_pixel_error))
            inst.vgeo_inst.render_pixel_error = err   # streamed copies cut at their instancer's render error
            row = {"camera": ci, "instancer": inst.name, "copies": len(inst.data.vertices), "error_px": err,
                   "runs": []}
            if not flag("--no-ideal"):
                row["cut"], row["cut_cull"] = ideal(inst, assets[inst.name], view, err)
            if native.has_atoms() and not flag("--no-atoms"):
                # phase 2: a meshlet cut on every copy drawn as instanced atoms, per Off-screen mode
                row["atoms"] = {}
                tr = instances.copy_transforms(inst)
                for mode in [m for m in modes if m] or ["COARSEN"]:
                    t0 = time.perf_counter()
                    _pairs, st = instances.select_atoms(inst, assets[inst.name], [view], err, mode=mode,
                                                        transforms=tr)
                    row["atoms"][mode] = {"triangles": int(st.triangles), "pairs": int(st.pairs),
                                          "select_ms": round((time.perf_counter() - t0) * 1000, 2)}
            for mode in modes:
                for slots in slot_counts:
                    if slots is not None:
                        inst.vgeo_inst.stream_slots = slots
                    if mode is not None and hasattr(inst.vgeo_inst, "offscreen"):
                        inst.vgeo_inst.offscreen = mode
                    t0 = time.perf_counter()
                    if mode is None:
                        instances.update(inst, [view], err, render=True)
                    else:
                        instances.update(inst, [view], err, mode=mode, render=True)
                    ms = (time.perf_counter() - t0) * 1000
                    off_tris, off_copies = offscreen_share(inst, view, err, mode)
                    row["runs"].append({"offscreen": mode or "default", "slots": inst.vgeo_inst.stream_slots,
                                        "shown": int(inst.vgeo_inst.shown_triangles),
                                        "offscreen_tris": off_tris, "offscreen_copies": off_copies,
                                        "update_ms": round(ms, 1)})
            rows.append(row)
            print(json.dumps(row), flush=True)
    for a in assets.values():
        a.close()

    # summary: totals per camera, averaged over cameras
    print("\nper camera (all instancers):")
    combos = [(r["offscreen"], r["slots"]) for r in rows[0]["runs"]]
    head = "".join(f"  shown[{m},{s}]".rjust(22) for m, s in combos)
    amodes = list(rows[0].get("atoms", {}).keys())
    head += "".join(f"atoms[{m}]".rjust(18) for m in amodes)
    print("cam".ljust(5) + head + "cut".rjust(14) + "cut+cull".rjust(14))
    sums = {"cut": 0, "cut_cull": 0, **{c: 0 for c in combos}, **{("atoms", m): 0 for m in amodes}}
    for ci in range(n_cams):
        rs = [r for r in rows if r["camera"] == ci]
        line = str(ci).ljust(5)
        for k, c in enumerate(combos):
            v = sum(r["runs"][k]["shown"] for r in rs)
            sums[c] += v / n_cams
            line += f"{v:,}".rjust(22)
        for m in amodes:
            v = sum(r["atoms"][m]["triangles"] for r in rs)
            sums[("atoms", m)] += v / n_cams
            line += f"{v:,}".rjust(18)
        for key in ("cut", "cut_cull"):
            v = sum(r.get(key, 0) for r in rs)
            sums[key] += v / n_cams
            line += f"{v:,}".rjust(14)
        print(line)
    line = "mean".ljust(5) + "".join(f"{int(sums[c]):,}".rjust(22) for c in combos)
    line += "".join(f"{int(sums[('atoms', m)]):,}".rjust(18) for m in amodes)
    line += f"{int(sums['cut']):,}".rjust(14) + f"{int(sums['cut_cull']):,}".rjust(14)
    print(line)
    out = arg("--json")
    if out:
        with open(out, "w") as f:
            json.dump({"rows": rows, "mean": {f"{m},{s}": int(sums[(m, s)]) for m, s in combos} |
                       {f"atoms,{m}": int(sums[("atoms", m)]) for m in amodes} |
                       {"cut": int(sums["cut"]), "cut_cull": int(sums["cut_cull"])}}, f, indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
