"""'Before' numbers: what the add-on drew before phase 1 for the same frames, without rendering.

Run it with the add-on of the old commit (a git worktree of master at 239e3a9) on the same scene: each
frame places the timeline's camera, uses the film's error budget and Coarsen (the only off-screen
rule the old code had) and its default of 4 streamed copies, cuts every instancer the way a final
render would (streamed copies cut synchronously), and records the triangles drawn.
Writes WORK/before.json: {frame: drawn triangles}.

Usage: blender -b --factory-startup --python before_stats.py -- [f0 f1]
Env:   VGEO_ADDONS_BEFORE  addons folder of the old worktree     SCENE, WORK, ERR_SCALE (1.5), RES
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
sys.path.insert(0, os.environ["VGEO_ADDONS_BEFORE"])
import vgeo  # noqa: E402
from vgeo import instances  # noqa: E402
import timeline as T  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
f0, f1 = (int(args[0]), int(args[1])) if len(args) >= 2 else (0, T.FRAMES)
WORK = os.path.abspath(os.environ.get("WORK", "work"))
W, H = (int(x) for x in os.environ.get("RES", "1280x720").split("x"))
ERR_SCALE = float(os.environ.get("ERR_SCALE", "1.5"))
OLD_SLOTS = 4   # the old default number of streamed copies

vgeo.register()
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(os.environ["SCENE"]))
scene = bpy.context.scene
eye = bpy.data.objects["Eye"]
eye.rotation_mode = 'QUATERNION'
eye.data.lens_unit, eye.data.sensor_fit = 'FOV', 'HORIZONTAL'
insts = list(instances.instancers(scene))
for i in insts:
    i.vgeo_inst.stream_slots = OLD_SLOTS


def view_of(st):
    q = (Vector(st["target"]) - Vector(st["pos"])).to_track_quat('-Z', 'Y')
    eye.location, eye.rotation_quaternion = Vector(st["pos"]), q
    eye.data.angle = math.radians(st["fov"])
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()   # the new lens must reach the evaluated camera
    mw = Matrix.Translation(Vector(st["pos"])) @ q.to_matrix().to_4x4()
    proj = eye.evaluated_get(dg).calc_matrix_camera(dg, x=W, y=H, scale_x=1.0, scale_y=1.0)
    return (mw.inverted(), proj, H, True, eye.data.clip_start)


path = os.path.join(WORK, "before.json")
out = json.load(open(path)) if os.path.exists(path) else {}
for f in range(f0, f1):
    st = T.state((f + 0.5) / T.FPS)
    view = view_of(st.get("eye") or st)
    err = st["errorPx"] * ERR_SCALE
    for inst in insts:
        instances.update(inst, [view], err, render=True)
    out[str(f)] = int(sum(i.vgeo_inst.shown_triangles for i in insts))
    if f % 48 == 0:
        print(f"BEFORE frame {f}: {out[str(f)]:,}", flush=True)
        json.dump(out, open(path, "w"))
json.dump(out, open(path, "w"))
print("BEFORE DONE", f0, f1, flush=True)
