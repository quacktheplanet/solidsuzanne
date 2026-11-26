"""GUI smoke test (needs the Blender UI, not -b): what a user does by hand, scripted.

    blender scene.blend --python tests/gui_check.py -- --out DIR [--cameras cams.json]

--cameras: the viewport looks from the first camera of a cams.json ({"pos", "target", "fov"}),
otherwise through the scene camera.

Waits for the live loop to settle, then: duplicates an instancer, undoes and redoes it, saves, renders
(F12 path, render handlers), turns a debug view and the stats overlay on and off, and after each
step checks that every visible instancer shows the cut its view calls for (a fresh selection for the
current view) and that nothing raised. Screenshots go to DIR. Prints GUICHECK lines and quits.
"""

import json
import os
import sys
import time
import traceback

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(REPO, "build-demo", "gui_check")
os.makedirs(OUT, exist_ok=True)

import vgeo  # noqa: E402
from vgeo import instances, stream  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("GUICHECK PASS " if ok else "GUICHECK FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def ctx():
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return dict(window=win, screen=win.screen, area=area, region=region), area


def view_cut_ok():
    """Every visible atom instancer shows the cut the live loop should show for the current view: a
    fresh viewport update (paced, so with the level fallback, but forced to run now) changes nothing,
    neither the reported triangles nor what the depsgraph draws."""
    views = stream.viewport_views()
    bad = []
    for inst in instances.instancers(bpy.context.scene):
        if not inst.vgeo_inst.use_atoms or not inst.visible_get():
            continue
        shown, drawn = inst.vgeo_inst.shown_triangles, drawn_triangles(inst)
        instances._atom_pace.pop(inst.vgeo_inst.uid, None)
        instances.update(inst, views, inst.vgeo_inst.pixel_error, paced=True)
        now, drawn2 = inst.vgeo_inst.shown_triangles, drawn_triangles(inst)
        if now != shown or drawn2 != drawn or drawn != shown:
            bad.append((inst.name, shown, now, drawn, drawn2))
    return not bad, bad


def drawn_triangles(inst):
    """Triangles the depsgraph instances for an instancer: its level meshes and its atom clouds."""
    dg = bpy.context.evaluated_depsgraph_get()
    t = 0
    for i in dg.object_instances:
        if i.is_instance and i.parent is not None:
            p = i.parent.original
            if p == inst or (p.parent == inst and instances.SLICE_PROP in p):
                t += len(i.object.data.polygons)
    return t


def per_instancer():
    """{instancer: (shown triangles, drawn instances from its clouds, drawn instances from its levels)}"""
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for inst in instances.instancers(bpy.context.scene):
        c = l = 0
        for i in dg.object_instances:
            if i.is_instance and i.parent is not None:
                p = i.parent.original
                if p == inst:
                    l += 1
                elif p.parent == inst and instances.SLICE_PROP in p:
                    c += 1
        out[inst.name] = (inst.vgeo_inst.shown_triangles, c, l)
    return out


def drawn_instances():
    dg = bpy.context.evaluated_depsgraph_get()
    return sum(1 for i in dg.object_instances if i.is_instance)


steps = []
state = {"i": 0, "t": 0.0, "dup": None}


def step(fn, wait=1.5):
    steps.append((fn, wait))
    return fn


@step
def settle():
    ok, bad = view_cut_ok()
    check("settled: shown cut is the view's cut", ok, str(bad))
    state["n0"] = drawn_instances()
    state["t0"] = per_instancer()
    check("something is drawn", state["n0"] > 0, str(state["n0"]))
    with bpy.context.temp_override(**ctx()[0]):
        bpy.ops.screen.screenshot_area(filepath=os.path.join(OUT, "01_settled.png"))


@step
def duplicate():
    over, _ = ctx()
    inst = next(i for i in instances.instancers(bpy.context.scene) if i.vgeo_inst.use_atoms)
    with bpy.context.temp_override(**over):
        bpy.ops.ed.undo_push(message="before duplicate")   # scripted operators push no undo step
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        inst.select_set(True)
        bpy.context.view_layer.objects.active = inst
        bpy.ops.object.duplicate_move(TRANSFORM_OT_translate={"value": (0, 0, 40)})
        bpy.ops.ed.undo_push(message="Duplicate")
    state["dup"] = bpy.context.active_object.name
    state["orig"] = inst.name


@step
def after_duplicate():
    dup = bpy.data.objects.get(state["dup"])
    orig = bpy.data.objects.get(state["orig"])
    ok = dup is not None and orig is not None and dup.vgeo_inst.uid != orig.vgeo_inst.uid
    kids = [c for c in dup.children if instances.SLICE_PROP in c] if dup else []
    check("duplicate (Shift+D) gets its own ID and clouds", ok and kids, f"{len(kids)} clouds")
    ok, bad = view_cut_ok()
    check("with the copy the shown cuts are the view's cuts", ok, str(bad))


@step
def undo():
    over, _ = ctx()
    with bpy.context.temp_override(**over):
        bpy.ops.ed.undo()


@step
def after_undo():
    check("undo removes the copy", bpy.data.objects.get(state["dup"]) is None)
    ok, bad = view_cut_ok()
    check("after undo the shown cut is the view's cut", ok, str(bad))
    now = per_instancer()
    check("drawn as before the duplicate", now == state["t0"], f"{now} vs {state['t0']}")


@step
def redo():
    over, _ = ctx()
    with bpy.context.temp_override(**over):
        bpy.ops.ed.redo()


@step
def after_redo():
    dup = bpy.data.objects.get(state["dup"])
    check("redo brings the copy back", dup is not None)
    ok, bad = view_cut_ok()
    check("after redo the shown cut is the view's cut", ok, str(bad))


@step
def save():
    path = os.path.join(OUT, "gui_check_saved.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True, relative_remap=False)
    libs = [o for o in bpy.data.objects if instances.LIB_PROP in o]
    check("save keeps the session's atom libraries", libs and all(len(o.data.polygons) for o in libs))
    check("the saved file is small", os.path.getsize(path) < 400 * 2**20, f"{os.path.getsize(path) / 2**20:.0f} MB")


@step
def render():
    scene = bpy.context.scene
    if "--cameras" in argv:   # render from the same viewpoint as the viewport
        import math
        from mathutils import Vector
        spec = json.load(open(argv[argv.index("--cameras") + 1]))[0]
        cam = bpy.data.objects.new("GUI Check Cam", bpy.data.cameras.new("GUI Check Cam"))
        scene.collection.objects.link(cam)
        cam.data.lens_unit, cam.data.sensor_fit = 'FOV', 'HORIZONTAL'
        cam.data.angle = math.radians(spec.get("fov", 50))
        cam.location = Vector(spec["pos"])
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = (Vector(spec["target"]) - Vector(spec["pos"])).to_track_quat('-Z', 'Y')
        scene.camera = cam
    scene.render.resolution_percentage = 25
    scene.render.filepath = os.path.join(OUT, "02_render.png")
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    check("F12 render runs", os.path.exists(scene.render.filepath), f"{time.perf_counter() - t0:.1f} s")


@step
def debug_on():
    bpy.context.scene.vgeo_debug = 'MESHLETS'
    bpy.context.scene.vgeo_stats = True


@step
def debug_shot():
    with bpy.context.temp_override(**ctx()[0]):
        bpy.ops.screen.screenshot_area(filepath=os.path.join(OUT, "03_meshlets.png"))
    libs = [o for o in bpy.data.objects if instances.LIB_PROP in o]
    check("debug view colors the atom libraries", libs and all(o.data.color_attributes.get("vgeo_viz") for o in libs))


@step
def debug_off():
    bpy.context.scene.vgeo_debug = 'NONE'
    bpy.context.scene.vgeo_stats = False


@step
def finish():
    with bpy.context.temp_override(**ctx()[0]):
        bpy.ops.screen.screenshot_area(filepath=os.path.join(OUT, "04_end.png"))
    ok, bad = view_cut_ok()
    check("at the end the shown cut is the view's cut", ok, str(bad))


def driver():
    try:
        if state["i"] >= len(steps):
            n_fail = sum(1 for _n, ok in results if not ok)
            print(f"GUICHECK DONE {len(results) - n_fail}/{len(results)} passed", flush=True)
            with open(os.path.join(OUT, "gui_check.json"), "w") as f:
                json.dump(results, f, indent=1)
            bpy.ops.wm.quit_blender()
            return None
        fn, wait = steps[state["i"]]
        state["i"] += 1
        fn()
        return wait
    except Exception:
        traceback.print_exc()
        check(f"step {steps[state['i'] - 1][0].__name__} raised", False)
        return 0.5


def start():
    if not hasattr(bpy.types.Object, "vgeo"):
        vgeo.register()
    _over, area = ctx()
    sp = area.spaces.active
    if "--cameras" in argv:   # a fixed view for the whole run
        import math
        from mathutils import Matrix, Vector
        cam = json.load(open(argv[argv.index("--cameras") + 1]))[0]
        q = (Vector(cam["target"]) - Vector(cam["pos"])).to_track_quat('-Z', 'Y')
        sp.region_3d.view_perspective = 'PERSP'
        sp.region_3d.view_matrix = (Matrix.Translation(Vector(cam["pos"])) @ q.to_matrix().to_4x4()).inverted()
        sp.lens = 36.0 / math.tan(math.radians(cam.get("fov", 50)) * 0.5)
        sp.clip_start, sp.clip_end = 0.1, 10000
    else:
        sp.region_3d.view_perspective = 'CAMERA'
    bpy.app.timers.register(driver, first_interval=6.0)   # let the first cut stream in
    return None


bpy.app.timers.register(start, first_interval=0.5)
