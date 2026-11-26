"""Renders frames of the statue avenue film with EEVEE, one still per frame, in one Blender session.
Pass 'a' renders the main view of every frame and writes NNNN.json (view, wipe, error, triangle
stats, overlay positions for the compositor); pass 'b' renders the incoming view of wipe frames.
Usage: python render_frames.py a|b f0 f1      (bpy module; or blender -b --python ... -- a f0 f1)
Env:   SCENE  the built statue_avenue.blend          WORK  output folder (frames go to WORK/frames)
       VGEO_ADDONS  add-on folder to import vgeo from (a scratch copy with demo_viz.patch applied)
       ERR_SCALE (1.5) film error = timeline error x this; TAA (4) EEVEE samples
Memory grows per rendered frame (see docs/ROADMAP.md), so render_chunks.sh runs 8 frames per process."""
import json
import math
import os
import sys
import time

import bpy
from bpy.app.handlers import persistent
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(D)))
sys.path.insert(0, D)
sys.path.insert(0, os.path.dirname(D))              # examples/statue_avenue: layout.py
sys.path.insert(0, os.environ.get("VGEO_ADDONS", os.path.join(REPO, "addons")))
import vgeo  # noqa: E402
from vgeo import stream, instances  # noqa: E402
import layout  # noqa: E402
import timeline as T  # noqa: E402

_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
PASS, f0, f1 = _args[0], int(_args[1]), int(_args[2])
WORK = os.path.abspath(os.environ.get("WORK", "work"))
OUT = os.path.join(WORK, "frames")
os.makedirs(OUT, exist_ok=True)
W, H = 1280, 720
ERR_SCALE = float(os.environ.get("ERR_SCALE", "1.5"))   # film error budget = timeline value x this
TAA = int(os.environ.get("TAA", "4"))

vgeo.register()
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(os.environ.get("SCENE", "statue_avenue_out/statue_avenue.blend")))
scene = bpy.context.scene
cam, eye = bpy.data.objects["Cam"], bpy.data.objects["Eye"]
insts = list(instances.instancers(scene))
MODES = {"beauty": 0, "meshlod": 1, "facetlod": 2}


def set_mode(v):
    """Material view: 0 beauty, 1 meshlets + LOD ramp, 2 facets + LOD ramp (vgeo_mode value node)."""
    for m in bpy.data.materials:
        if m.use_nodes and m.node_tree and "vgeo_mode" in m.node_tree.nodes:
            m.node_tree.nodes["vgeo_mode"].outputs[0].default_value = float(v)
SCENE_TRIS = int(scene["scene_tris"])
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = TAA
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.quality = 94
scene.render.use_overwrite = False
scene.render.use_placeholder = False
scene.render.filepath = os.path.join(OUT, "####_" + PASS)
for ob in (cam, eye):
    ob.rotation_mode = 'QUATERNION'
    ob.data.lens_unit = 'FOV'
    ob.data.sensor_fit = 'HORIZONTAL'
    ob.animation_data_clear()


def look(pos, target, roll=0.0):
    q = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y')
    if roll:
        q = q @ Quaternion((0, 0, 1), roll)
    return q


def place(ob, st):
    ob.location = Vector(st["pos"])
    ob.rotation_quaternion = look(st["pos"], st["target"], st.get("roll", 0.0))
    ob.data.angle = math.radians(st["fov"])


def view_of(st, dg):
    """A VGEO view tuple for a camera state (used to cut from the frozen eye)."""
    mw = Matrix.Translation(Vector(st["pos"])) @ look(st["pos"], st["target"]).to_matrix().to_4x4()
    eye.data.angle = math.radians(st["fov"])
    proj = eye.evaluated_get(dg).calc_matrix_camera(dg, x=W, y=H, scale_x=1.0, scale_y=1.0)
    return (mw.inverted(), proj, H, True, eye.data.clip_start)


def set_error(px):
    for p in stream.proxies(scene):
        p.vgeo.render_pixel_error = px
    for i in insts:
        i.vgeo_inst.render_pixel_error = px


def to_px(camobj_matrix_state, p):
    v = world_to_camera_view(scene, cam, Vector(p))
    return [v.x * W, (1 - v.y) * H, v.z > 0]


CUR = {}


def floor_no_shadow():
    """The paving never needs to cast shadows (it only shadows its own grooves): skip it in shadow maps."""
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.visible_shadow and ob.data and ob.data.materials and ob.data.materials[0] \
                and ob.data.materials[0].name in ("Paving", "Underlay"):
            ob.visible_shadow = False
    pv = bpy.data.objects.get("Paving")
    if pv is not None:
        pv.visible_shadow = False


@persistent
def pre(scn, dg=None):
    f = scn.frame_current
    st = T.state((f + 0.5) / T.FPS)
    CUR.clear(); CUR.update(st=st, t0=time.perf_counter())
    place(cam, st)
    scn.camera = cam
    err = st["errorPx"] * ERR_SCALE
    set_error(err)
    set_mode(MODES[st["viewB"] if PASS == "b" else st["view"]])
    if st.get("eye"):
        stream.FREEZE = False
        view = view_of(st["eye"], dg or bpy.context.evaluated_depsgraph_get())
        for inst in insts:
            instances.update(inst, [view], err, render=True)
        stream.FREEZE = True
    else:
        stream.FREEZE = False
    floor_no_shadow()


def stats():
    drawn = sum(i.vgeo_inst.shown_triangles for i in insts)
    streamed = [p for p in stream.proxies(scene) if p.vgeo.slot_of and not p.hide_render]
    rts = [stream._runtimes.get(p.vgeo.uid) for p in streamed]
    meshlets = sum(rt.clusters for rt in rts if rt is not None)
    return {"drawnTriangles": int(drawn), "streamedMeshlets": int(meshlets), "streamedCopies": len(streamed),
            "sceneTris": SCENE_TRIS, "statues": int(scene["statues"]), "patches": int(scene["patches"])}


def overlay(st):
    o = {"star": to_px(None, Vector(st["pos"]) + Vector(layout.SUN_DIR) * 10000.0)}
    if st.get("eye"):
        e = st["eye"]
        q = look(e["pos"], e["target"])
        org = Vector(e["pos"])
        half_w = math.tan(math.radians(e["fov"]) / 2)
        half_h = half_w * H / W
        dirs = [q @ Vector((sx * half_w, sy * half_h, -1.0)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        far = [org + d.normalized() * 260.0 for d in dirs]
        near = [org + d.normalized() * 3.0 for d in dirs]
        segs = []
        for i in range(4):
            for a, b in ((org, far[i]), (far[i], far[(i + 1) % 4]), (near[i], near[(i + 1) % 4])):
                A, B = to_px(None, a), to_px(None, b)
                if A[2] and B[2]:
                    segs.append([A[0], A[1], B[0], B[1]])
        o["frustum"] = segs
        o["eye2d"] = to_px(None, org)
    return o


@persistent
def post(scn, dg=None):
    st = CUR.get("st")
    f = scn.frame_current
    sec = time.perf_counter() - CUR.get("t0", time.perf_counter())
    if PASS == "a" and st is not None:
        rec = {"f": f, "t": (f + 0.5) / T.FPS, "view": st["view"], "viewB": st["viewB"], "wipe": st["wipe"],
               "errorPx": st["errorPx"] * ERR_SCALE, "frozen": bool(st.get("eye")), "sec": round(sec, 2)}
        rec.update(stats())
        rec.update(overlay(st))
        with open(os.path.join(OUT, f"{f:04d}.json"), "w") as fh:
            json.dump(rec, fh)
    print(f"frame {f} pass {PASS} {sec:.1f}s", flush=True)


bpy.app.handlers.frame_change_pre.append(pre)

if PASS == "look":
    for f in [int(x) for x in os.environ["FRAMES"].split(",")]:
        for which in ("a", "b"):
            PASS = which
            st = T.state((f + 0.5) / T.FPS)
            if which == "b" and not st["viewB"]:
                continue
            scene.frame_set(f)
            stream.FREEZE = bool(st.get("eye"))
            scene.render.filepath = os.path.join(OUT, f"{f:04d}_{which}.jpg")
            t = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            post(scene)
            stream.FREEZE = False
            print(f"LOOK {f}{which} {time.perf_counter() - t:.1f}s", flush=True)
    sys.exit(0)
# still renders, one per frame (memory stays bounded better than one long animation render)
t0 = time.perf_counter()
for f in range(f0, f1):
    st = T.state((f + 0.5) / T.FPS)
    if PASS == "b" and not st["viewB"]:
        continue
    path = os.path.join(OUT, f"{f:04d}_{PASS}.jpg")
    if os.path.exists(path) and (PASS == "b" or os.path.exists(os.path.join(OUT, f"{f:04d}.json"))):
        continue
    scene.frame_set(f)
    stream.FREEZE = bool(st.get("eye"))
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    post(scene)
    stream.FREEZE = False
print("CHUNK DONE", PASS, f0, f1, f"{time.perf_counter() - t0:.0f}s", flush=True)
