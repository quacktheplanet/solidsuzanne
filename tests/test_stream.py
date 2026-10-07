"""Headless tests for the VGEO streaming add-on.

Run:  blender -b --factory-startup --python tests/test_stream.py [-- --out DIR]
"""

import os
import sys
import tempfile
import time

import bpy
import numpy as np
from mathutils import Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else tempfile.mkdtemp(prefix="vgeo_test_")
os.makedirs(OUT, exist_ok=True)

import vgeo  # noqa: E402
from vgeo import native, stream  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def dense_rock(subdiv=8):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=1.0)
    ob = bpy.context.object
    bpy.ops.object.shade_smooth()
    tex = bpy.data.textures.new("rock", "CLOUDS")
    tex.noise_scale = 0.35
    tex.noise_depth = 4
    m = ob.modifiers.new("disp", "DISPLACE")
    m.texture = tex
    m.strength = 0.3
    mat_a = bpy.data.materials.new("Stone")
    mat_a.diffuse_color = (0.55, 0.5, 0.45, 1)
    mat_b = bpy.data.materials.new("Moss")
    mat_b.diffuse_color = (0.2, 0.4, 0.15, 1)
    for m, c, r in ((mat_a, (0.55, 0.5, 0.45, 1), 0.8), (mat_b, (0.2, 0.4, 0.15, 1), 0.9)):
        m.use_nodes = True
        bsdf = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        bsdf.inputs["Base Color"].default_value = c
        bsdf.inputs["Roughness"].default_value = r
    ob.data.materials.append(mat_a)
    ob.data.materials.append(mat_b)
    # upper half gets the second material, so there is a material border to keep
    zs = np.empty(len(ob.data.polygons) * 3, np.float32)
    ob.data.polygons.foreach_get("center", zs)
    mi = (zs.reshape(-1, 3)[:, 2] > 0.3).astype(np.int32)
    ob.data.polygons.foreach_set("material_index", mi)
    ob.data.update()
    return ob


def cut_geometry(proxy):
    """All chunk geometry of a proxy as (positions, triangles)."""
    pos_all, tri_all, base = [], [], 0
    for ob in stream.fronts(proxy):
        me = ob.data
        nv, nl = len(me.vertices), len(me.loops)
        if nl == 0:
            continue
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        cv = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", cv)
        pos_all.append(co.reshape(-1, 3))
        tri_all.append(cv.reshape(-1, 3) + base)
        base += nv
    return np.concatenate(pos_all), np.concatenate(tri_all)


def open_edges(pos, tris):
    """Edges used by exactly one triangle, after welding identical positions."""
    _, weld = np.unique(np.round(pos, 6), axis=0, return_inverse=True)
    t = weld.ravel()[tris]
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum())


def camera(loc, target=(0, 0, 0), lens=50):
    cam = bpy.data.objects.get("Cam")
    if cam is None:
        cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
        bpy.context.scene.collection.objects.link(cam)
    cam.data.lens = lens
    cam.data.clip_start = 0.01
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()
    return cam


def main():
    print("Blender", bpy.app.version_string, "| out:", OUT)
    reset()
    vgeo.register()
    try:
        run()
    finally:
        vgeo.unregister()
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


def run():
    check("native library loads", native.available(), native.library_path())

    src = dense_rock(8)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 960, 540
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "stream_test.blend"))

    t0 = time.perf_counter()
    rc = bpy.ops.vgeo.virtualize()
    dt = time.perf_counter() - t0
    proxy = bpy.context.active_object
    check("virtualize finished", rc == {'FINISHED'} and proxy is not src, f"{dt:.1f}s")
    check("proxy stores relative path", proxy.vgeo.path.startswith("//vgeo/"), proxy.vgeo.path)
    check("vgeo file written", os.path.exists(stream.asset_path(proxy)))
    check("source hidden, kept", src.hide_render and src.hide_get() and proxy.vgeo.source == src)
    check("materials carried", [m.name for m in proxy.data.materials] == ["Stone", "Moss"])
    import struct
    blob = open(stream.asset_path(proxy), "rb").read()
    k = blob.find(b"MATP")
    params = struct.unpack_from("<8f", blob, k + 4) if k > 0 else ()
    check("material colors embedded for web viewers",
          k > 0 and abs(params[0] - 0.55) < 1e-5 and abs(params[3] - 0.8) < 1e-5 and abs(params[5] - 0.4) < 1e-5,
          str([round(x, 3) for x in params]))
    with open(os.path.join(OUT, "last_asset.txt"), "w") as f:
        f.write(stream.asset_path(proxy))
    rt = stream.runtime_for(proxy)
    src_tris = rt.asset.info["source_triangles"]
    check("source triangle count", src_tris == 327680, str(src_tris))
    check("DAG has many levels", rt.asset.info["lod_levels"] >= 8, str(rt.asset.info["lod_levels"]))

    # chunks are scene objects parented to the proxy (no Geometry Nodes instancing: EEVEE would
    # treat the whole surface as changed whenever one chunk changed)
    col = proxy.vgeo.collection
    chunks = list(col.objects)
    check("chunk pairs are parented to the proxy", len(chunks) == 2 * rt.asset.chunk_count
          and all(o.parent == proxy for o in chunks), f"{len(chunks)} chunk objects")
    check("chunk collection linked next to the proxy",
          all(col.name in c.children for c in proxy.users_collection))
    check("proxy has no instancing modifier", not any(m.type == 'NODES' for m in proxy.modifiers))
    check("proxy mesh is only its bounds", len(proxy.data.vertices) == 8 and len(proxy.data.polygons) == 0)
    dg = bpy.context.evaluated_depsgraph_get()
    drawn = sum(1 for i in dg.object_instances if i.object.original in set(chunks))
    check("chunks are drawn as objects", drawn == 2 * rt.asset.chunk_count, str(drawn))
    check("direct mesh writes verified", stream._fast_write is True)

    # cuts from the render camera at increasing distance
    counts = []
    for d in (1.6, 3, 8, 40):
        camera((0, -d, 0.2))
        view = stream.camera_view(scene)
        stream.apply_cut(proxy, [view], 1.0)
        counts.append(rt.triangles)
    check("cut shrinks with distance", all(a > b for a, b in zip(counts, counts[1:])), str(counts))
    check("far cut is small", counts[-1] < src_tris * 0.05, str(counts[-1]))

    # a cliff-sized rock seen from its surface mixes many LOD levels in one cut: must be crack-free
    proxy.scale = (30, 30, 30)
    camera((0, -33, 3), target=(25, 10, 0), lens=24)
    lods = set()
    proxy.vgeo.lod_colors = True
    stream.apply_cut(proxy, [stream.camera_view(scene)], 1.0, force=True)
    for ob in stream.fronts(proxy):
        a = ob.data.color_attributes.get("vgeo_lod")
        if a is not None and len(a.data):
            c = np.empty(len(a.data) * 4, np.float32)
            a.data.foreach_get("color", c)
            lods.update(map(tuple, np.round(c.reshape(-1, 4)[:, :3], 2)))
    # (a 327k-triangle test mesh has few, large groups; the large-scale demo shows deep mixes)
    check("grazing view mixes LOD levels", len(lods) >= 2, f"{len(lods)} levels visible")
    pos, tris = cut_geometry(proxy)
    oe = open_edges(pos, tris)
    check("mixed cut is watertight", oe == 0, f"{oe} open edges, {len(tris)} tris")
    # off-screen handling at cliff scale
    view = stream.camera_view(scene)
    stream.apply_cut(proxy, [view], 1.0, "FULL")
    full = rt.triangles
    stream.apply_cut(proxy, [view], 1.0, "COARSEN", 8.0)
    coarse = rt.triangles
    pos, tris = cut_geometry(proxy)
    oe = open_edges(pos, tris)
    check("coarsened off-screen is watertight", oe == 0, f"{oe} open edges")
    check("coarsening saves triangles", coarse < full, f"{full:,} -> {coarse:,}")
    stream.apply_cut(proxy, [view], 1.0, "CULL")
    check("culling saves more", rt.triangles < coarse, f"{rt.triangles:,}")
    stream.apply_cut(proxy, [view], 1.0, "FULL")

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = 'VERTEX'
    scene.render.filepath = os.path.join(OUT, "lod_cliff.png")
    bpy.ops.render.render(write_still=True)
    stream.apply_cut(proxy, [stream.camera_view(scene)], 1.0)
    proxy.scale = (1, 1, 1)
    proxy.vgeo.lod_colors = False

    # full detail and coarsest levels are also closed
    stream.apply_level(proxy, 0)
    pos, tris = cut_geometry(proxy)
    check("level 0 equals source", len(tris) == src_tris, str(len(tris)))
    check("level 0 watertight", open_edges(pos, tris) == 0)

    # incremental: a tiny camera move rebuilds few chunks
    camera((0, -3, 0.2))
    stream.apply_cut(proxy, [stream.camera_view(scene)], 1.0)
    camera((0.02, -3, 0.2))
    stream.apply_cut(proxy, [stream.camera_view(scene)], 1.0)
    check("small move rebuilds few chunks", rt.last_rebuilt <= rt.asset.chunk_count // 2,
          f"{rt.last_rebuilt}/{rt.asset.chunk_count}")
    stream.apply_cut(proxy, [stream.camera_view(scene)], 1.0)
    check("unchanged view is a no-op", rt.last_rebuilt == 0 or rt.key is not None)

    # incremental (live loop) path: nothing changes on screen until the swap, then it matches a direct cut
    camera((0, -1.8, 0.3))
    view = stream.camera_view(scene)
    before = {o.name: o.data.name for o in stream.fronts(proxy)}
    tris_before = sum(len(o.data.polygons) for o in stream.fronts(proxy))
    steps, updates0, unchanged = 0, rt.updates, True
    while rt.updates == updates0:  # until the flip
        stream.stream_step(proxy, [view], 1.0, "FULL", 8.0, budget=0.0005)
        steps += 1
        if rt.updates == updates0:
            unchanged &= sum(len(o.data.polygons) for o in stream.fronts(proxy)) == tris_before
    check("incremental keeps old cut until swap", unchanged, f"{steps} steps")
    while stream.stream_step(proxy, [view], 1.0, "FULL", 8.0, budget=0.0005):
        steps += 1
    inc = sum(len(o.data.polygons) for o in stream.fronts(proxy))
    check("incremental ran in several steps", steps >= 2, f"{steps} steps, {rt.last_rebuilt} chunks")
    check("incremental result matches stats", inc == rt.triangles, f"{inc} vs {rt.triangles}")
    stream.apply_cut(proxy, [view], 1.0, "FULL", force=True)
    check("incremental equals direct cut", inc == rt.triangles)
    pos, tris = cut_geometry(proxy)
    check("incremental cut watertight", open_edges(pos, tris) == 0)
    def backs():
        return [o for o in proxy.vgeo.collection.objects if o.scale[0] < 0.5]
    check("direct cut empties hidden backs", all(len(o.data.vertices) == 0 for o in backs()))
    check("one back per chunk", len(backs()) == rt.asset.chunk_count, str(len(backs())))
    # streaming never creates or deletes datablocks
    n_meshes, n_objects = len(bpy.data.meshes), len(bpy.data.objects)
    for d in (2.4, 1.8, 2.4):
        camera((0, -d, 0.3))
        while stream.stream_step(proxy, [stream.camera_view(scene)], 1.0, "FULL", 8.0, budget=0.002):
            pass
    check("streaming creates no datablocks", (len(bpy.data.meshes), len(bpy.data.objects)) == (n_meshes, n_objects),
          f"{n_meshes}/{n_objects} -> {len(bpy.data.meshes)}/{len(bpy.data.objects)}")
    check("idle loop empties hidden backs", all(len(o.data.vertices) == 0 for o in backs()))
    pos, tris = cut_geometry(proxy)
    check("flipped cut watertight", open_edges(pos, tris) == 0, f"{len(tris)} tris")
    # BATCH strategy (EEVEE viewports): unlinked spares, landed in one step
    def run_batch(d):
        camera((0, -d, 0.3))
        v = stream.camera_view(scene)
        u0, shown0, same = rt.updates, sum(len(o.data.polygons) for o in stream.fronts(proxy)), True
        while rt.updates == u0:
            stream.stream_step(proxy, [v], 1.0, "FULL", 8.0, budget=0.0005, strategy='BATCH')
            if rt.updates == u0:
                same &= sum(len(o.data.polygons) for o in stream.fronts(proxy)) == shown0
        while stream.stream_step(proxy, [v], 1.0, "FULL", 8.0, budget=0.0005, strategy='BATCH'):
            pass
        return same
    same = run_batch(2.1)
    check("batch keeps old cut until it lands", same)
    pos, tris = cut_geometry(proxy)
    check("batch cut watertight", open_edges(pos, tris) == 0, f"{len(tris)} tris")
    check("batch cut matches stats", len(tris) == rt.triangles)
    run_batch(2.6)
    run_batch(2.1)
    counts = (len(bpy.data.meshes), len(bpy.data.objects))
    run_batch(2.6)
    run_batch(2.1)
    check("batch reuses its spares", (len(bpy.data.meshes), len(bpy.data.objects)) == counts,
          f"{counts} -> {(len(bpy.data.meshes), len(bpy.data.objects))}")
    spares = [m for m in bpy.data.meshes if m.name.startswith("vgeo.") and m.users == 0]
    check("at most one spare per chunk", len(spares) <= 2 * rt.asset.chunk_count, str(len(spares)))
    check("spares are empty when idle", all(len(m.vertices) == 0 for m in spares), f"{len(spares)} spares")

    # an interrupted update is discarded cleanly
    camera((0, -6, 0.3))
    shown = sum(len(o.data.polygons) for o in stream.fronts(proxy))
    stream.stream_step(proxy, [stream.camera_view(scene)], 1.0, "FULL", 8.0, budget=0.0)
    rt.invalidate()
    check("interrupted update leaves the shown cut alone",
          not rt.pending and sum(len(o.data.polygons) for o in stream.fronts(proxy)) == shown)
    # saving drops hidden geometry
    for o in backs()[:3]:
        stream.fill_mesh(o.data, rt.asset.extract(0), (), False) if rt.asset.extract(0) else None
    bpy.ops.wm.save_mainfile()
    check("save empties hidden backs", all(len(o.data.vertices) == 0 for o in backs()))

    # chunks follow the proxy: visibility, render visibility, selection, collections
    vl = bpy.context.view_layer
    lc = stream._layer_collection(vl.layer_collection, proxy.vgeo.collection)
    proxy.hide_set(True)
    stream.sync_visibility(proxy, vl)
    hidden = lc.hide_viewport and not any(o.visible_get() for o in stream.fronts(proxy))
    proxy.hide_set(False)
    stream.sync_visibility(proxy, vl)
    check("hiding the proxy hides its chunks", hidden and all(o.visible_get() for o in stream.fronts(proxy)))
    proxy.hide_render = True
    stream.update_for_render(scene)
    check("proxy hide_render reaches the chunks", proxy.vgeo.collection.hide_render)
    proxy.hide_render = False
    stream.sync_visibility(proxy, vl)
    for o in vl.objects:
        o.select_set(False)
    front0 = stream.fronts(proxy)[0]
    front0.select_set(True)
    vl.objects.active = front0
    stream.redirect_selection(vl)
    check("clicking a chunk selects the proxy", proxy.select_get() and not front0.select_get()
          and vl.objects.active == proxy)
    moved = bpy.data.collections.new("Moved")
    scene.collection.children.link(moved)
    moved.objects.link(proxy)
    scene.collection.objects.unlink(proxy)
    stream.link_chunks(proxy)
    check("chunk collection follows the proxy to another collection",
          proxy.vgeo.collection.name in moved.children
          and proxy.vgeo.collection.name not in scene.collection.children)
    scene.collection.objects.link(proxy)
    moved.objects.unlink(proxy)
    stream.link_chunks(proxy)
    bpy.data.collections.remove(moved)
    # files from before: chunks instanced by a Geometry Nodes modifier, collection not linked
    scene.collection.children.unlink(proxy.vgeo.collection)
    mod = proxy.modifiers.new("VGEO Stream", 'NODES')
    mod.node_group = bpy.data.node_groups.new(stream.NODE_GROUP, "GeometryNodeTree")
    stream.link_chunks(proxy)
    check("old instancing files are switched over", not proxy.modifiers
          and proxy.vgeo.collection.name in scene.collection.children)

    # material border survives simplification
    mats = set()
    for ob in stream.fronts(proxy):
        me = ob.data
        if "material_index" in me.attributes and len(me.polygons):
            m = np.empty(len(me.polygons), np.int32)
            me.attributes["material_index"].data.foreach_get("value", m)
            mats.update(np.unique(m).tolist())
    check("both materials present in cut", mats == {0, 1}, str(mats))

    # renders: EEVEE + Cycles through the render_pre handler
    proxy.vgeo.lod_colors = False
    camera((0, -3.2, 0.9), lens=50)
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3
    sun.rotation_euler = (0.9, 0.2, 0.6)
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("W")
    world.color = (0.05, 0.06, 0.08)
    scene.world = world
    for eng, fname in (("BLENDER_EEVEE", "eevee.png"), ("CYCLES", "cycles.png")):
        scene.render.engine = eng
        if eng == "CYCLES":
            scene.cycles.samples = 16
        scene.render.filepath = os.path.join(OUT, fname)
        rt.key = None
        before = rt.updates
        bpy.ops.render.render(write_still=True)
        check(f"{eng} render wrote image", os.path.exists(scene.render.filepath))
        check(f"{eng} render_pre updated the cut", rt.updates > before, f"{rt.triangles:,} tris")
    check("render flag cleared", stream._rendering is False)

    # LOD color debug still (workbench, color attribute)
    proxy.vgeo.lod_colors = True
    camera((1.25, -0.2, 0.1), target=(-1, 2, 0), lens=24)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = 'VERTEX'
    scene.display.shading.light = 'STUDIO'
    scene.render.filepath = os.path.join(OUT, "lod_colors.png")
    bpy.ops.render.render(write_still=True)
    check("LOD color render", os.path.exists(scene.render.filepath))
    proxy.vgeo.lod_colors = False

    # export for web: compact asset + viewer + page, never overwriting a page it did not write
    from vgeo import webexport
    web_dir = os.path.join(OUT, "web_export")
    os.makedirs(web_dir, exist_ok=True)
    with open(os.path.join(web_dir, "index.html"), "w") as f:
        f.write("<p>hand written</p>")
    res = webexport.export(proxy, web_dir)
    check("web export writes asset, viewer, decoder",
          all(os.path.exists(os.path.join(web_dir, f)) for f in res["files"]) and res["asset"].endswith(".vgeow"),
          ", ".join(res["files"]))
    check("web export keeps a foreign index.html",
          open(os.path.join(web_dir, "index.html")).read() == "<p>hand written</p>"
          and res["files"][-1].endswith(".html") and res["files"][-1] != "index.html", res["files"][-1])
    check("web asset smaller than .vgeo", res["bytes"] < os.path.getsize(stream.asset_path(proxy)),
          f"{res['bytes'] / 2**20:.1f} MB")
    head = open(os.path.join(web_dir, res["asset"]), "rb").read(160)
    import struct as _st
    version, pages, roots = _st.unpack_from("<I", head, 8)[0], _st.unpack_from("<I", head, 100)[0],         _st.unpack_from("<I", head, 124)[0]
    check("web export is paged (range-request streaming)", version == 2 and pages > 1 and 1 <= roots <= pages,
          f"{pages} pages, {roots} root")
    with open(os.path.join(web_dir, "last_web.txt"), "w") as f:
        f.write(res["files"][-1])

    # save / reload: chunk meshes persist, runtime reopens
    bpy.ops.wm.save_mainfile()
    tris_saved = sum(len(o.data.polygons) for o in stream.fronts(proxy))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(OUT, "stream_test.blend"))
    proxy = next(o for o in bpy.data.objects if o.vgeo.uid)
    check("chunks survive save/load", sum(len(o.data.polygons) for o in stream.fronts(proxy)) == tris_saved)
    check("runtimes reset on load", not stream._runtimes)
    camera((0, -8, 0.2))
    rt = stream.apply_cut(proxy, [stream.camera_view(bpy.context.scene)], 1.0)
    check("runtime reopens after load", rt.asset is not None and rt.triangles > 0, str(rt.triangles))

    # ---- instancing: many copies, each at its own level
    from vgeo import instances
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 30, -1))
    ground = bpy.context.active_object
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ground.select_set(True)
    proxy.select_set(True)
    bpy.context.view_layer.objects.active = proxy
    rc = bpy.ops.vgeo.scatter(count=400, seed=3, scale_min=0.6, scale_max=1.2)
    inst = bpy.context.active_object
    check("scatter creates an instancer", rc == {'FINISHED'} and inst.vgeo_inst.uid and len(inst.data.vertices) == 400)
    errs = list(inst["vgeo_level_errors"])
    lvl_tris = list(inst["vgeo_level_tris"])
    check("a level mesh per error level, full detail to coarsest",
          len(inst.vgeo_inst.levels.objects) == len(errs) >= 3 and lvl_tris[0] == rt.asset.info["source_triangles"],
          f"{len(errs)} levels, {lvl_tris[0]:,} -> {lvl_tris[-1]:,} tris")
    check("level errors grow, triangles shrink", all(a <= b for a, b in zip(errs, errs[1:]))
          and all(a >= b for a, b in zip(lvl_tris, lvl_tris[1:])) and errs[0] == 0.0)
    camera((0, -12, 1.5), target=(0, 30, -1), lens=35)
    instances.update(inst, [stream.camera_view(scene)], 1.0)
    dg = bpy.context.evaluated_depsgraph_get()
    n_inst = sum(1 for i in dg.object_instances if i.is_instance and i.parent and i.parent.original == inst)
    check("every placement is a real instance", n_inst == 400, str(n_inst))
    co = np.empty(1200, np.float32)
    inst.data.vertices.foreach_get("co", co)
    lv = np.empty(400, np.int32)
    inst.data.attributes["vgeo_level"].data.foreach_get("value", lv)
    dist = np.linalg.norm(co.reshape(-1, 3) - np.array([0, -12, 1.5]), axis=1)
    order = np.argsort(dist)
    near, far = lv[order[:40]].mean(), lv[order[-40:]].mean()
    check("near copies finer than far copies", near < far, f"mean level near {near:.1f}, far {far:.1f}")
    full = 400 * lvl_tris[0]
    check("instances show a fraction of full detail", inst.vgeo_inst.shown_triangles < full * 0.25,
          f"{inst.vgeo_inst.shown_triangles:,} of {full:,}")
    check("unchanged view writes nothing", instances.update(inst, [stream.camera_view(scene)], 1.0) is False)

    # streamed copies: the copy right in front of the camera wants level 0 (the whole asset at full
    # detail); a view-dependent cut of its own is lighter, and takes over only once it has landed
    inst.vgeo_inst.stream_slots = 2
    pts = np.empty(1200, np.float32)
    inst.data.vertices.foreach_get("co", pts)
    pts = pts.reshape(-1, 3)
    target = int(np.argmin(np.linalg.norm(pts - np.array([0, 10, -1]), axis=1)))
    p_world = inst.matrix_world @ Vector(pts[target])
    r_target = float(inst["vgeo_radius"]) * float(inst.data.attributes["vgeo_scale"].data[target].value)
    eye = p_world + Vector((0.0, -1.7, 0.6)) * r_target
    camera(tuple(eye), target=tuple(p_world), lens=24)
    views = [stream.camera_view(scene)]
    instances.update(inst, views, 1.0)
    sl = instances.slots(inst)
    assigned = sorted(o.vgeo.slot_index for o in sl if o.vgeo.slot_index >= 0)
    flags = instances._streamed_flags(inst)
    check("streamed copies assigned to the nearest full-detail copies",
          len(sl) == 2 and target in assigned and all(o.hide_select for o in sl), str(assigned))
    check("instance kept until the streamed cut lands", not flags.any())
    for _ in range(400):
        busy = False
        for o in sl:
            if o.vgeo.slot_index >= 0:
                busy |= stream.stream_step(o, views, 1.0, "COARSEN", 8.0, budget=0.01)
        instances.sync_slots(inst)
        if not busy and instances._streamed_flags(inst).sum() == len(assigned):
            break
    flags = instances._streamed_flags(inst)
    check("landed copies switch to their streamed cut", sorted(np.nonzero(flags)[0].tolist()) == assigned)
    near = next(o for o in sl if o.vgeo.slot_index == target)
    rt_near = stream._runtimes[near.vgeo.uid]
    check("streamed copy is lighter than full detail", 0 < rt_near.triangles < lvl_tris[0] * 0.8,
          f"{rt_near.triangles:,} vs level 0 {lvl_tris[0]:,}")
    mw = near.matrix_world
    check("streamed copy sits on its placement",
          (mw.translation - p_world).length < 1e-4 and abs(mw.to_scale()[0] - float(
              inst.data.attributes["vgeo_scale"].data[target].value)) < 1e-4)
    dg = bpy.context.evaluated_depsgraph_get()
    n_inst = sum(1 for i in dg.object_instances if i.is_instance and i.parent and i.parent.original == inst)
    check("streamed placements leave the instancing", n_inst == 400 - len(assigned), str(n_inst))
    check("shown triangles count the streamed cuts", inst.vgeo_inst.shown_triangles > 0)
    # walking away hands them back
    camera((0, -150, 60), target=(0, 30, -1), lens=35)
    instances.update(inst, [stream.camera_view(scene)], 1.0)
    left = [c.name for o in sl if o.vgeo.collection for c in o.vgeo.collection.objects if len(c.data.vertices)]
    check("far view returns copies to instances", not instances._streamed_flags(inst).any()
          and all(o.vgeo.slot_index < 0 for o in sl) and not left,
          f"flags {int(instances._streamed_flags(inst).sum())}, slots {[o.vgeo.slot_index for o in sl]}, "
          f"non-empty {left[:3]}")
    # final renders make the streamed cut on the spot
    camera(tuple(eye), target=tuple(p_world), lens=24)
    stream.update_for_render(scene)
    check("render cuts streamed copies at once", instances._streamed_flags(inst)[target]
          and stream._runtimes[near.vgeo.uid].triangles > 0)
    camera((0, -12, 1.5), target=(0, 30, -1), lens=35)
    instances.update(inst, [stream.camera_view(scene)], 1.0)
    world = bpy.data.worlds.get("W") or bpy.data.worlds.new("W")
    scene.world = world
    if not any(o.type == 'LIGHT' for o in scene.objects):
        sun = bpy.data.objects.new("Sun2", bpy.data.lights.new("Sun2", "SUN"))
        sun.rotation_euler = (0.9, 0.2, 0.6)
        scene.collection.objects.link(sun)
    for eng, fname in (("BLENDER_EEVEE", "instances_eevee.png"), ("CYCLES", "instances_cycles.png")):
        scene.render.engine = eng
        if eng == "CYCLES":
            scene.cycles.samples = 8
        scene.render.filepath = os.path.join(OUT, fname)
        bpy.ops.render.render(write_still=True)
        check(f"{eng} renders the instancer", os.path.exists(scene.render.filepath))
    shown_render = inst.vgeo_inst.shown_triangles
    check("render cut uses the render error", shown_render > 0, f"{shown_render:,} tris at 0.5 px")
    bpy.ops.wm.save_mainfile()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(OUT, "stream_test.blend"))
    inst = next(o for o in bpy.data.objects if o.vgeo_inst.uid)
    proxy = next(o for o in bpy.data.objects if o.vgeo.uid)
    check("instancer survives save/load", len(inst.vgeo_inst.levels.objects) == len(errs)
          and instances.choose_levels(inst, [stream.camera_view(bpy.context.scene)], 1.0).max() >= 0)

    # corrupt / missing file is reported, not a crash
    proxy.vgeo.path = "//vgeo/missing.vgeo"
    rt = stream.runtime_for(proxy)
    check("missing file reported", rt.asset is None and "cannot open" in (rt.error or ""), rt.error or "")
    bad = os.path.join(OUT, "bad.vgeo")
    with open(bad, "wb") as f:
        f.write(b"VGEO2\0\0\0" + b"\xff" * 400)
    try:
        native.Asset(bad)
        check("corrupt file rejected", False)
    except RuntimeError as e:
        check("corrupt file rejected", True, str(e))

    # the file is memory-mapped: a handle allocates little of its own, however big the asset
    good = open(os.path.join(OUT, "last_asset.txt")).read()
    a = native.Asset(good)
    mapped, heap = a.memory()
    check("asset is memory-mapped", mapped == os.path.getsize(good) and heap < mapped * 0.25,
          f"mapped {mapped / 2**20:.1f} MB, own {heap / 2**20:.2f} MB")
    a.select_level(0)
    a.prefetch(np.arange(a.chunk_count))
    full = sum(d["tri_count"] for d in (a.extract(c) for c in range(a.chunk_count)) if d)
    check("prefetch + extract after mapping", full == a.info["source_triangles"] and not a.corrupt, f"{full:,}")
    a.close()
    # indices are validated per cluster on first use (a full scan would read the whole file):
    # a bad cluster is skipped and reported, the rest still extracts
    import struct
    blob = bytearray(open(good, "rb").read())
    off_indices = struct.unpack_from("<Q", blob, 80 + 4 * 8)[0]   # Header.off_indices
    struct.pack_into("<3I", blob, off_indices, 0xFFFFFFF0, 0xFFFFFFF1, 0xFFFFFFF2)
    bad_idx = os.path.join(OUT, "bad_index.vgeo")
    with open(bad_idx, "wb") as f:
        f.write(blob)
    a = native.Asset(bad_idx)
    a.select_level(0)
    got = sum(d["tri_count"] for d in (a.extract(c) for c in range(a.chunk_count)) if d)
    check("bad cluster skipped and reported", a.corrupt and 0 < got < full, f"{got:,} of {full:,}")
    a.close()

    # indexed build path: smooth, no UVs, material border split automatically
    from vgeo import build as vbuild
    rock2 = dense_rock(7)
    rock2.name = "Smooth"
    while rock2.data.uv_layers:
        rock2.data.uv_layers.remove(rock2.data.uv_layers[0])
    arrays = vbuild.mesh_arrays(rock2, bpy.context.evaluated_depsgraph_get())
    check("smooth mesh uses indexed layout", arrays["indices"] is not None,
          f"{len(arrays['positions'])} verts")
    bpy.context.view_layer.objects.active = rock2
    rock2.select_set(True)
    rc = bpy.ops.vgeo.virtualize()
    p2 = bpy.context.active_object
    check("indexed virtualize", rc == {'FINISHED'} and p2.vgeo.uid)
    camera((0, -2.2, 0.4))
    rt2 = stream.apply_cut(p2, [stream.camera_view(bpy.context.scene)], 2.0)
    pos, tris = cut_geometry(p2)
    check("indexed cut watertight", open_edges(pos, tris) == 0, f"{len(tris)} tris")
    mats = set()
    for ob in stream.fronts(p2):
        me = ob.data
        if "material_index" in me.attributes and len(me.polygons):
            m = np.empty(len(me.polygons), np.int32)
            me.attributes["material_index"].data.foreach_get("value", m)
            mats.update(np.unique(m).tolist())
    check("indexed keeps both materials", mats == {0, 1}, str(mats))
    bpy.context.view_layer.objects.active = p2
    bpy.ops.vgeo.restore()

    # geometry-nodes materials: the evaluated mesh's materials win over the object's slots
    gn_obj = bpy.data.objects.new("GNGrid", bpy.data.meshes.new("GNGrid"))
    bpy.context.scene.collection.objects.link(gn_obj)
    gn_obj.data.materials.append(bpy.data.materials.new("SlotMat"))
    ng = bpy.data.node_groups.new("gridmat", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    grid = ng.nodes.new("GeometryNodeMeshGrid")
    grid.inputs["Vertices X"].default_value = 64
    grid.inputs["Vertices Y"].default_value = 64
    setm = ng.nodes.new("GeometryNodeSetMaterial")
    gn_mat = bpy.data.materials.new("GNMat")
    setm.inputs["Material"].default_value = gn_mat
    outn = ng.nodes.new("NodeGroupOutput")
    ng.links.new(grid.outputs["Mesh"], setm.inputs["Geometry"])
    ng.links.new(setm.outputs["Geometry"], outn.inputs[0])
    gn_obj.modifiers.new("gn", "NODES").node_group = ng
    bpy.context.view_layer.objects.active = gn_obj
    gn_obj.select_set(True)
    rc = bpy.ops.vgeo.virtualize()
    gp = bpy.context.active_object
    slots = [m.name if m else None for m in gp.data.materials]
    shown = set()
    for ob in stream.fronts(gp):
        me = ob.data
        if len(me.polygons):
            mi = np.zeros(len(me.polygons), np.int32)
            if "material_index" in me.attributes:
                me.attributes["material_index"].data.foreach_get("value", mi)
            shown.update(slots[i] for i in np.unique(mi))
    # Blender evaluates Set Material as slots [None, GNMat] with faces on slot 1; VGEO must match
    check("GN Set Material carried to proxy", rc == {'FINISHED'} and shown == {"GNMat"}, f"{slots} -> {shown}")
    # deleting a proxy (not Restore) leaves its chunk collection behind: it must be unlinked
    gcol = gp.vgeo.collection
    bpy.data.objects.remove(gp)
    stream.unlink_orphans()
    linked = [c.name for c in bpy.data.collections if gcol.name in c.children]
    linked += [sc.name for sc in bpy.data.scenes if gcol.name in sc.collection.children]
    check("deleted proxy's chunks are unlinked", not linked, str(linked))

    # restore brings the source back
    proxy_uid = proxy.vgeo.uid
    bpy.context.view_layer.objects.active = proxy
    rc = bpy.ops.vgeo.restore()
    src = bpy.data.objects.get("Icosphere")
    check("restore returns source", rc == {'FINISHED'} and src is not None and not src.hide_render)
    check("restore removes its chunks (instancer levels stay)",
          not any(o.name.startswith(f"vgeo.{proxy_uid}.") for o in bpy.data.objects))


main()
