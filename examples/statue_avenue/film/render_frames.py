"""Renders frames of the statue avenue film with EEVEE, one still per frame, in one Blender session.

Pass 'a' renders the main view of every frame and writes NNNN.json (view, wipe, error, triangle
stats, overlay positions for the compositor); pass 'b' renders the incoming view of wipe frames and
writes NNNN_b.json with that view's stats.

Every frame cuts explicitly before it renders (and keeps that cut through the render handlers): the
camera's view, or the frozen "eye" view in the observer shots, at the view's Off-screen mode. That
way the streamed copies of the frame are known before rendering, and their chunks get an object
property `vgeo_streamed` = 1 that the materials read for the tint (a node added here, at render time).
In the Off-screen and payoff shots, pass 'a' also records what each Off-screen mode would draw.

Usage: blender -b --factory-startup --python render_frames.py -- a|b f0 f1
       (or the bpy module: python render_frames.py a f0 f1)
Env:   SCENE  the built statue_avenue.blend          WORK  output folder (frames go to WORK/frames)
       VGEO_ADDONS  add-on folder to import vgeo from (a scratch copy with demo_viz.patch applied)
       ERR_SCALE (1.5) film error = timeline error x this; TAA (8) EEVEE samples; RES (1280x720)
"""
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
W, H = (int(x) for x in os.environ.get("RES", "1280x720").split("x"))
ERR_SCALE = float(os.environ.get("ERR_SCALE", "1.5"))   # film error budget = timeline value x this
TAA = int(os.environ.get("TAA", "8"))
MODE_SHOTS = ("frozen", "payoff")                        # shots that record every Off-screen mode

vgeo.register()
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(os.environ.get("SCENE", "statue_avenue_out/statue_avenue.blend")))
scene = bpy.context.scene
cam, eye = bpy.data.objects["Cam"], bpy.data.objects["Eye"]
insts = list(instances.instancers(scene))
SCENE_TRIS = int(scene["scene_tris"])
# source triangles of each instancer's asset (the close-up caption reads the statue's)
ASSET_TRIS = {}
for _i in insts:
    _p = _i.vgeo_inst.source
    _n = int(_p.vgeo.source_triangles) if _p is not None else 0
    if not _n and _p is not None:
        _rt = stream.runtime_for(_p)
        _n = int(_rt.asset.info["source_triangles"]) if _rt.asset is not None else 0
    ASSET_TRIS[_i.name] = _n
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = TAA
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = W, H, 100
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.quality = 94
scene.render.use_overwrite = False
scene.render.use_placeholder = False
for ob in (cam, eye):
    ob.rotation_mode = 'QUATERNION'
    ob.data.lens_unit = 'FOV'
    ob.data.sensor_fit = 'HORIZONTAL'
    ob.animation_data_clear()


# ---------------------------------------------------------------- materials: view switch and tint

def add_tint():
    """Mix an amber glow into every VGEO material, weighted by the object property vgeo_streamed (set on
    the chunks of streamed copies) times the value node vgeo_tint (0/1 per frame)."""
    for m in bpy.data.materials:
        nt = m.node_tree if m.use_nodes else None
        if nt is None or "vgeo_mode" not in nt.nodes or "vgeo_tint" in nt.nodes:
            continue
        out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        link = out.inputs["Surface"].links[0]
        src = link.from_socket
        nt.links.remove(link)
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_type, attr.attribute_name = 'OBJECT', "vgeo_streamed"
        tint = nt.nodes.new("ShaderNodeValue")
        tint.name = tint.label = "vgeo_tint"
        tint.outputs[0].default_value = 0.0
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = 'MULTIPLY'
        nt.links.new(attr.outputs["Fac"], mul.inputs[0])
        nt.links.new(tint.outputs[0], mul.inputs[1])
        k = nt.nodes.new("ShaderNodeMath")
        k.operation = 'MULTIPLY'
        k.inputs[1].default_value = 0.13
        nt.links.new(mul.outputs[0], k.inputs[0])
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 0.46, 0.08, 1.0)
        em.inputs["Strength"].default_value = 1.6
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(k.outputs[0], mix.inputs[0])
        nt.links.new(src, mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])


def set_view_nodes(mode, tint):
    """Material view: 0 beauty, 1 meshlets + LOD ramp (vgeo_mode); tint on streamed copies (vgeo_tint)."""
    for m in bpy.data.materials:
        if m.use_nodes and m.node_tree and "vgeo_mode" in m.node_tree.nodes:
            m.node_tree.nodes["vgeo_mode"].outputs[0].default_value = float(mode)
            if "vgeo_tint" in m.node_tree.nodes:
                m.node_tree.nodes["vgeo_tint"].outputs[0].default_value = float(tint)


add_tint()


# ---------------------------------------------------------------- cameras and cuts

def look(pos, target, roll=0.0):
    q = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y')
    if roll:
        q = q @ Quaternion((0, 0, 1), roll)
    return q


def place(ob, st):
    ob.location = Vector(st["pos"])
    ob.rotation_quaternion = look(st["pos"], st["target"], st.get("roll", 0.0))
    ob.data.angle = math.radians(st["fov"])


def view_of(st):
    """A VGEO view tuple for a camera state (eye object, so the scene camera is left alone)."""
    eye.location = Vector(st["pos"])
    eye.rotation_quaternion = look(st["pos"], st["target"], st.get("roll", 0.0))
    eye.data.angle = math.radians(st["fov"])
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()   # a changed lens must reach the evaluated camera before calc_matrix_camera
    mw = Matrix.Translation(Vector(st["pos"])) @ eye.rotation_quaternion.to_matrix().to_4x4()
    proj = eye.evaluated_get(dg).calc_matrix_camera(dg, x=W, y=H, scale_x=1.0, scale_y=1.0)
    return (mw.inverted(), proj, H, True, eye.data.clip_start)


def set_error(px):
    for p in stream.proxies(scene):
        p.vgeo.render_pixel_error = px
    for i in insts:
        i.vgeo_inst.render_pixel_error = px


def cut(view, err, mode):
    """Cut every instancer (and its streamed copies) for this view and Off-screen mode. The streamed copies
    are released first, so a cut does not depend on the cuts before it (which copy keeps a streamed cut
    has hysteresis): every frame, mode and pass gives the same numbers, in any order or chunking."""
    for inst in insts:
        inst.vgeo_inst.offscreen = mode
        instances.assign_slots(inst, [])
        instances.update(inst, [view], err, mode=mode, render=True)
    return sum(i.vgeo_inst.shown_triangles for i in insts)


def active_slots():
    out = []
    for inst in insts:
        flags = instances._streamed_flags(inst)
        for ob in instances.slots(inst):
            i = ob.vgeo.slot_index
            if 0 <= i < len(flags) and flags[i]:
                out.append((inst, ob, i))
    return out


def mark_streamed(active):
    """vgeo_streamed = 1 on the chunks of streamed copies that show, 0 elsewhere (the tint reads it)."""
    on = {ob.name for _inst, ob, _i in active}
    for inst in insts:
        for ob in instances.slots(inst):
            col = ob.vgeo.collection
            v = 1.0 if ob.name in on else 0.0
            for c in (col.objects if col is not None else ()):
                if c.get("vgeo_streamed") != v:
                    c["vgeo_streamed"] = v


def floor_no_shadow():
    """The paving never needs to cast shadows (it only shadows its own grooves): skip it in shadow maps."""
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.visible_shadow and ob.data and ob.data.materials and ob.data.materials[0] \
                and ob.data.materials[0].name in ("Paving", "Underlay"):
            ob.visible_shadow = False
    pv = bpy.data.objects.get("Paving")
    if pv is not None:
        pv.visible_shadow = False


# ---------------------------------------------------------------- per frame

CUR = {}


def to_px(p):
    v = world_to_camera_view(scene, cam, Vector(p))
    return [v.x * W, (1 - v.y) * H, v.z > 0]


def streamed_overlay(active, cam_pos):
    """Screen tags for the streamed copies: anchor above the copy, distance, triangles, and how the copy
    ranks among all copies of its instancer by distance to the camera (1 = nearest)."""
    import numpy as np
    tags = []
    for inst, ob, i in active:
        mw = ob.matrix_world
        pts = [mw @ Vector(c) for c in ob.bound_box]
        top = Vector((sum(p.x for p in pts) / 8, sum(p.y for p in pts) / 8, max(p.z for p in pts) + 0.6))
        center = sum(pts, Vector()) / 8
        rt = stream._runtimes.get(ob.vgeo.uid)
        co, _scale = instances._points(inst)
        mw_i = np.array(inst.matrix_world, dtype=np.float64)
        world = co @ mw_i[:3, :3].T + mw_i[:3, 3]
        d = np.linalg.norm(world - np.array(cam_pos), axis=1)
        rank = int((d < d[i]).sum()) + 1
        tags.append({"xy": to_px(top), "dist": round((center - Vector(cam_pos)).length, 1),
                     "tris": int(rt.triangles) if rt is not None else 0, "rank": rank, "instancer": inst.name})
    return tags


def frame_stats(active):
    drawn = sum(i.vgeo_inst.shown_triangles for i in insts)
    rts = [stream._runtimes.get(ob.vgeo.uid) for _inst, ob, _i in active]
    meshlets = sum(rt.clusters for rt in rts if rt is not None)
    streamed_tris = sum(rt.triangles for rt in rts if rt is not None)
    return {"drawnTriangles": int(drawn), "streamedMeshlets": int(meshlets), "streamedCopies": len(active),
            "streamedTriangles": int(streamed_tris), "sceneTris": SCENE_TRIS,
            "statues": int(scene["statues"]), "patches": int(scene["patches"]), "assetTris": ASSET_TRIS}


def frustum_overlay(e):
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
            A, B = to_px(a), to_px(b)
            if A[2] and B[2]:
                segs.append([A[0], A[1], B[0], B[1]])
    return {"frustum": segs, "eye2d": to_px(org)}


@persistent
def pre(scn, dg=None):
    f = scn.frame_current
    st = T.state((f + 0.5) / T.FPS)
    vname = st["viewB"] if PASS == "b" else st["view"]
    mat_mode, off_mode, tint = T.VIEWS[vname]
    CUR.clear()
    CUR.update(st=st, view=vname, t0=time.perf_counter())
    place(cam, st)
    scn.camera = cam
    err = st["errorPx"] * ERR_SCALE
    set_error(err)
    set_view_nodes(mat_mode, tint)
    stream.FREEZE = False
    view = view_of(st.get("eye") or st)
    modes = {}
    if PASS == "a" and st["shot"] in MODE_SHOTS:
        for m in ("COARSEN", "SHADOWS", "CULL"):
            modes[m] = cut(view, err, m)
    cut(view, err, off_mode)
    stream.FREEZE = True            # the render handlers keep this cut
    active = active_slots()
    mark_streamed(active)
    CUR.update(modes=modes, active=active, err=err, off_mode=off_mode)
    floor_no_shadow()


@persistent
def post(scn, dg=None):
    st = CUR.get("st")
    f = scn.frame_current
    sec = time.perf_counter() - CUR.get("t0", time.perf_counter())
    if st is None:
        return
    active = CUR["active"]
    rec = {"f": f, "t": (f + 0.5) / T.FPS, "shot": st["shot"], "view": CUR["view"], "viewB": st["viewB"],
           "wipe": st["wipe"], "errorPx": CUR["err"], "offscreen": CUR["off_mode"], "frozen": bool(st.get("eye")),
           "sec": round(sec, 2)}
    rec.update(frame_stats(active))
    if CUR["modes"]:
        rec["modes"] = CUR["modes"]
    rec["star"] = to_px(Vector(st["pos"]) + Vector(layout.SUN_DIR) * 10000.0)
    rec["tags"] = streamed_overlay(active, st["pos"])
    if st.get("eye"):
        rec.update(frustum_overlay(st["eye"]))
    name = f"{f:04d}.json" if PASS == "a" else f"{f:04d}_b.json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(rec, fh)
    print(f"frame {f} pass {PASS} {CUR['view']} {sec:.1f}s drawn {rec['drawnTriangles']:,}", flush=True)


bpy.app.handlers.frame_change_pre.append(pre)

# still renders, one per frame
t0 = time.perf_counter()
for f in range(f0, f1):
    st = T.state((f + 0.5) / T.FPS)
    if PASS == "b" and not st["viewB"]:
        continue
    path = os.path.join(OUT, f"{f:04d}_{PASS}.jpg")
    js = os.path.join(OUT, f"{f:04d}.json" if PASS == "a" else f"{f:04d}_b.json")
    if os.path.exists(path) and os.path.exists(js):
        continue
    scene.frame_set(f)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    post(scene)
    stream.FREEZE = False
print("CHUNK DONE", PASS, f0, f1, f"{time.perf_counter() - t0:.0f}s", flush=True)
