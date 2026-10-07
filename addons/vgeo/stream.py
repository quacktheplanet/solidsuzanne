"""Streams a view-dependent cut of a .vgeo asset into real Blender meshes.

A virtualized object is a normal mesh object (the proxy, whose own mesh only
holds its materials and eight loose corner points for framing). Its chunk
objects live in a collection linked next to it and are parented to it, so
they follow its transform. Each chunk holds the part of the current DAG cut
that falls in its region of space. EEVEE, Cycles and Workbench render the
chunks like any other mesh: full materials, lights, shadows, ray tracing.

Every chunk is a pair of objects: the front (scale 1) shows the current
cut, the back (scale 0) is where the next contents are built. When the view
changes, the native runtime picks a new cut; the backs of the chunks that
changed are refilled in place a few milliseconds per tick, and because they
are drawn (at zero size) Blender prepares their GPU buffers along the way.
When all are ready the pairs flip scale in one tick. The surface is never a
mix of two cuts, no datablocks are created or deleted (which would force
depsgraph relation rebuilds), and the flip itself costs a transform update.

The chunks used to be instanced by a Geometry Nodes modifier on the proxy.
Any chunk change then re-evaluated the proxy, and EEVEE treated the whole
instanced surface as changed (about 120 ms per frame on the terrain demo, so
EEVEE views had to land updates in one expensive frame). As separate objects
only the chunk that changed is updated. Files made with the modifier are
switched over when they are opened.
"""

import ctypes
import time
import uuid

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Vector

from . import native

NODE_GROUP = "VGEO Stream"
TICK = 0.1           # seconds between view checks when idle
TICK_BUDGET = 0.008  # seconds of mesh building per tick while an update is in flight
WARM_CACHES = True
CYCLES_SETTLE = 0.3  # seconds the view must be still before a Cycles viewport gets a new cut
DIAG_SWAP = False    # benchmarks: time the depsgraph evaluation right after a swap

_runtimes = {}       # uid -> Runtime
_rendering = False
swap_count = 0       # completed live swaps (diagnostics)

LOD_PALETTE = np.array([
    (0.90, 0.30, 0.25, 1), (0.95, 0.60, 0.20, 1), (0.95, 0.85, 0.25, 1), (0.55, 0.85, 0.30, 1),
    (0.25, 0.75, 0.55, 1), (0.25, 0.65, 0.90, 1), (0.40, 0.45, 0.95, 1), (0.65, 0.40, 0.90, 1),
    (0.90, 0.40, 0.75, 1), (0.70, 0.70, 0.70, 1), (0.45, 0.30, 0.20, 1), (0.20, 0.35, 0.30, 1),
], dtype=np.float32)


class Runtime:
    """Native asset handle plus what is currently written into the chunk meshes."""

    def __init__(self, uid, path):
        self.uid = uid
        self.path = path
        self.error = None
        self.asset = None
        try:
            self.asset = native.Asset(path)
        except Exception as e:  # missing file, corrupt file, no DLL
            self.error = str(e)
            return
        n = self.asset.chunk_count
        self.applied = np.zeros(n, dtype=np.uint64)
        self.valid = np.zeros(n, dtype=bool)
        self.key = None
        self.lod_colors = None
        self.triangles = 0
        self.clusters = 0
        self.last_ms = 0.0
        self.latency_ms = 0.0
        self.last_rebuilt = 0
        self.updates = 0
        # incremental update in flight (see stream_step)
        self.target = None
        self.todo = []
        self.pending = set()
        self.strategy = None
        self.to_clear = set()  # chunks whose (hidden) back still holds old geometry
        self.spare = {}        # chunk -> unlinked mesh used by BATCH builds
        self.spare_clear = set()
        self.chunk_cache = None

    def invalidate(self):
        if self.asset:
            self.valid[:] = False
            self.key = None
            self.chunk_cache = None
            _discard_pending(self)

    def close(self):
        if self.asset:
            self.asset.close()
            self.asset = None


# ---------------------------------------------------------------- helpers

def proxies(scene=None):
    """Virtualized objects, optionally limited to one scene."""
    objs = scene.objects if scene is not None else bpy.data.objects
    return [o for o in objs if o.type == 'MESH' and o.vgeo.uid and o.vgeo.path]


def asset_path(obj):
    return bpy.path.abspath(obj.vgeo.path, library=obj.library)


def runtime_for(obj):
    uid = obj.vgeo.uid
    path = asset_path(obj)
    rt = _runtimes.get(uid)
    if rt is None or rt.path != path:
        if rt:
            rt.close()
        rt = Runtime(uid, path)
        _runtimes[uid] = rt
    return rt


def invalidate_all(close=False):
    for rt in list(_runtimes.values()):
        if close:
            rt.close()
        else:
            rt.invalidate()
    if close:
        _runtimes.clear()


def chunk_name(uid, i):
    return f"vgeo.{uid}.{i:04d}"


def ensure_node_group():
    ng = bpy.data.node_groups.get(NODE_GROUP)
    if ng and ng.bl_idname == "GeometryNodeTree" and "Chunks" in ng.interface.items_tree:
        return ng
    ng = bpy.data.node_groups.new(NODE_GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Chunks", in_out="INPUT", socket_type="NodeSocketCollection")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gin.location = (-300, 0)
    info = ng.nodes.new("GeometryNodeCollectionInfo")
    info.transform_space = "ORIGINAL"
    info.inputs["Separate Children"].default_value = False
    info.inputs["Reset Children"].default_value = False
    out = ng.nodes.new("NodeGroupOutput")
    out.location = (300, 0)
    ng.links.new(gin.outputs["Chunks"], info.inputs["Collection"])
    ng.links.new(info.outputs["Instances"], out.inputs["Geometry"])
    return ng


def modifier_socket_id(ng):
    return ng.interface.items_tree["Chunks"].identifier


def chunk_objects(obj, rt):
    """(a, b) object pairs by chunk index, creating any that are missing.

    The full check (collection membership) runs once per runtime; later calls
    only confirm the cached objects still exist.
    """
    if rt.chunk_cache is not None:
        try:
            if all(a.name and b.name for a, b in rt.chunk_cache):  # ReferenceError if freed (undo)
                return rt.chunk_cache
        except ReferenceError:
            pass
        rt.chunk_cache = None
    col = obj.vgeo.collection
    if col is None:
        col = bpy.data.collections.new(f".vgeo {obj.vgeo.uid}")
        obj.vgeo.collection = col
    link_chunks(obj)
    members = set(col.objects.keys())

    def get(name, shown):
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != 'MESH':
            ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
            _set_shown(ob, shown)
        if ob.name not in members:
            col.objects.link(ob)
        if ob.parent != obj:
            ob.parent = obj
            ob.matrix_parent_inverse.identity()
        return ob

    out = []
    materials = tuple(obj.data.materials)
    for i in range(rt.asset.chunk_count):
        name = chunk_name(obj.vgeo.uid, i)
        a, b = get(name, True), get(name + "~", False)
        if (a.scale[0] >= 0.5) == (b.scale[0] >= 0.5):  # both shown or both hidden: repair
            _set_shown(a, True)
            _set_shown(b, False)
        # materials up front: assigning them later, mid-stream, forces relation rebuilds
        _sync_materials(a.data, materials)
        _sync_materials(b.data, materials)
        out.append((a, b))
    rt.chunk_cache = out
    rt.materials = materials
    return out


def front_back(pair):
    a, b = pair
    return (a, b) if a.scale[0] >= 0.5 else (b, a)


def _set_shown(ob, shown):
    """Front: full size, casts shadows. Back: zero size, no shadows (a changing caster would
    invalidate EEVEE's shadow maps every frame while it is being filled)."""
    ob.scale = (1.0, 1.0, 1.0) if shown else (0.0, 0.0, 0.0)
    ob.visible_shadow = shown


def fronts(obj):
    """The chunk objects currently shown (for inspection and tests)."""
    col = obj.vgeo.collection
    return [o for o in col.objects if o.scale[0] >= 0.5] if col else []


def link_chunks(obj):
    """Link the chunk collection next to the proxy (in every collection that holds it), drop the
    old Geometry Nodes instancing if the file still has it, and give the proxy corner points."""
    col = obj.vgeo.collection
    if col is None:
        return
    if obj.library is None:
        for m in [m for m in obj.modifiers if m.type == 'NODES' and m.node_group
                  and m.node_group.name.startswith(NODE_GROUP)]:
            obj.modifiers.remove(m)
    holders = [c for c in obj.users_collection if c.library is None and c != col]
    for c in holders:
        if col.name not in c.children:
            c.children.link(col)
    # the proxy was moved to other collections: follow it
    for parent in [c for c in bpy.data.collections if col.name in c.children and c not in holders]:
        parent.children.unlink(col)
    for scene in bpy.data.scenes:
        master = scene.collection
        if col.name in master.children and master not in holders:
            master.children.unlink(col)
    _proxy_bounds(obj)


def _proxy_bounds(obj):
    """Eight loose points at the asset's bounds: nothing is drawn or rendered, but Frame Selected
    and bounding boxes see the whole asset."""
    me = obj.data
    if me is None or me.library is not None or len(me.vertices) >= 8:
        return
    rt = _runtimes.get(obj.vgeo.uid)
    if rt is None or rt.asset is None:
        return
    lo, hi = rt.asset.info["aabb_min"], rt.asset.info["aabb_max"]
    co = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    me.clear_geometry()
    me.vertices.add(8)
    me.vertices.foreach_set("co", [c for p in co for c in p])
    me.update()


def _layer_collection(lc, col):
    if lc.collection == col:
        return lc
    for child in lc.children:
        found = _layer_collection(child, col)
        if found is not None:
            return found
    return None


def sync_visibility(obj, view_layer=None):
    """Chunks show and render only when the proxy does (the eye, the monitor and the camera)."""
    col = obj.vgeo.collection
    if col is None or col.library is not None:
        return
    if col.hide_render != obj.hide_render:
        col.hide_render = obj.hide_render
    if col.hide_viewport != obj.hide_viewport:
        col.hide_viewport = obj.hide_viewport
    vl = view_layer or bpy.context.view_layer
    if vl is None:
        return
    lc = _layer_collection(vl.layer_collection, col)
    if lc is not None:
        try:
            hidden = obj.hide_get(view_layer=vl)
        except RuntimeError:  # proxy not in this view layer
            return
        if lc.hide_viewport != hidden:
            lc.hide_viewport = hidden


def redirect_selection(view_layer=None):
    """Clicking the surface picks a chunk object: select its proxy instead."""
    vl = view_layer or bpy.context.view_layer
    if vl is None:
        return
    active = vl.objects.active
    for ob in list(vl.objects.selected):
        p = ob.parent
        if p is not None and p.vgeo.uid and ob.name.startswith(f"vgeo.{p.vgeo.uid}."):
            if p.vgeo.slot_of and p.parent is not None:
                p = p.parent  # a streamed copy's chunk: select the instancer
            ob.select_set(False)
            try:
                p.select_set(True)
            except RuntimeError:
                continue
            if active == ob:
                vl.objects.active = p


def unlink_orphans():
    """Chunk collections whose proxy was deleted would stay visible: unlink them."""
    owned = {o.vgeo.collection.name for o in bpy.data.objects
             if o.vgeo.uid and o.vgeo.collection is not None and o.users}
    for col in bpy.data.collections:
        if not col.name.startswith(".vgeo ") or col.name in owned or col.library is not None:
            continue
        for parent in [c for c in bpy.data.collections if col.name in c.children]:
            parent.children.unlink(col)
        for scene in bpy.data.scenes:
            if col.name in scene.collection.children:
                scene.collection.children.unlink(col)


# ---------------------------------------------------------------- views

def viewport_views():
    """(view_matrix, window_matrix, height_px, is_persp, clip_start) for every visible 3D view."""
    views = []
    wm = bpy.context.window_manager
    for win in wm.windows:
        screen = win.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            regions = [r for r in area.regions if r.type == 'WINDOW' and r.width > 1 and r.height > 1]
            rv3ds = list(space.region_quadviews) if space.region_quadviews else [space.region_3d]
            for region, rv3d in zip(regions, rv3ds):
                if rv3d is None:
                    continue
                views.append((rv3d.view_matrix.copy(), rv3d.window_matrix.copy(), region.height,
                              rv3d.is_perspective, space.clip_start))
    return views


def viewport_strategy():
    """How the live loop should land updates, given what the 3D views show.

    CYCLES: some view renders with Cycles (update once the view settles)
    STAGED: everything else. EEVEE used to need BATCH (see the module notes);
            with chunks as separate objects, staging is about 4x cheaper there.
    """
    for win in bpy.context.window_manager.windows:
        if win.screen is None:
            continue
        engine = win.scene.render.engine
        for area in win.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            if area.spaces.active.shading.type == 'RENDERED' and engine == 'CYCLES':
                return 'CYCLES'
    return 'STAGED'


def camera_view(scene, depsgraph=None):
    """The render camera as a view tuple, at final render resolution."""
    cam = scene.camera
    if cam is None:
        return None
    r = scene.render
    w = max(1, int(r.resolution_x * r.resolution_percentage / 100))
    h = max(1, int(r.resolution_y * r.resolution_percentage / 100))
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    cam_eval = cam.evaluated_get(dg)
    proj = cam_eval.calc_matrix_camera(dg, x=w, y=h, scale_x=r.pixel_aspect_x, scale_y=r.pixel_aspect_y)
    persp = cam.type == 'CAMERA' and cam.data.type != 'ORTHO'
    clip = cam.data.clip_start if cam.type == 'CAMERA' else 0.01
    return (cam_eval.matrix_world.inverted(), proj, h, persp, clip)


def _frustum_planes(clip):
    rows = [clip.row[i] for i in range(4)]
    planes = []
    for i in range(3):
        for s in (1, -1):
            p = Vector(rows[3]) + s * Vector(rows[i])
            n = p.xyz.length or 1.0
            planes.append((p.x / n, p.y / n, p.z / n, p.w / n))
    return planes


def local_views(obj, views, pixel_error, mode="FULL", offscreen_scale=8.0):
    mw = obj.matrix_world
    try:
        inv = mw.inverted()
    except ValueError:
        return []
    scale = max(abs(s) for s in mw.to_scale()) or 1.0
    out = []
    for view, window, height, persp, clip_start in views:
        cam_world = view.inverted().translation
        cam_local = inv @ cam_world
        threshold = pixel_error / max(1, height)
        planes = _frustum_planes(window @ view @ mw) if mode != "FULL" else None
        if persp:
            out.append(native.make_view(cam_local, window[1][1], max(clip_start / scale, 1e-6),
                                        threshold, planes=planes, mode=mode, offscreen_scale=offscreen_scale))
        else:
            ortho_h = 2.0 / window[1][1] if window[1][1] else 1.0
            out.append(native.make_view(cam_local, 1.0, 1e-6, threshold, ortho=True,
                                        ortho_height=ortho_h / scale, planes=planes, mode=mode,
                                        offscreen_scale=offscreen_scale))
    return out


def _key(obj, views, pixel_error, mode, offscreen_scale):
    parts = [round(pixel_error, 4), mode, round(offscreen_scale, 3), bool(obj.vgeo.lod_colors)]
    for m in [obj.matrix_world] + [v[0] for v in views] + [v[1] for v in views]:
        parts.extend(round(x, 5) for row in m for x in row)
    parts.extend(v[2] for v in views)
    return tuple(parts)


# ---------------------------------------------------------------- mesh writing

# Direct copies into Blender's attribute arrays are 5-10x faster than
# foreach_set, which takes a per-item path for topology arrays. The layout is
# verified once per session against foreach_get; any mismatch switches back
# to foreach_set for good.
_fast_write = None   # None = not verified yet


def _copy_into(attr_data, arr):
    ctypes.memmove(attr_data[0].as_pointer(), arr.ctypes.data, arr.nbytes)


def _write_fast(me, d):
    A = me.attributes
    _copy_into(A["position"].data, d["positions"])
    _copy_into(A[".edge_verts"].data, d["edge_verts"])
    _copy_into(A[".corner_vert"].data, d["corner_verts"])
    _copy_into(A[".corner_edge"].data, d["corner_edges"])
    ctypes.memmove(me.polygons[0].as_pointer(), d["loop_starts"].ctypes.data, d["loop_starts"].nbytes)


def _write_slow(me, d):
    me.vertices.foreach_set("co", d["positions"])
    me.edges.foreach_set("vertices", d["edge_verts"])
    me.loops.foreach_set("vertex_index", d["corner_verts"])
    me.loops.foreach_set("edge_index", d["corner_edges"])
    me.polygons.foreach_set("loop_start", d["loop_starts"])


def _fast_write_works():
    """Write a known two-triangle quad through the direct path and read it back.

    Uses fixed data on purpose: real chunks can legitimately contain things
    validate() would "fix" (simplification leaves rare back-to-back fins),
    which must not be mistaken for a broken memory layout.
    """
    d = {
        "positions": np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], np.float32),
        "edge_verts": np.array([0, 1, 1, 2, 2, 0, 2, 3, 3, 0], np.int32),
        "corner_verts": np.array([0, 1, 2, 0, 2, 3], np.int32),
        "corner_edges": np.array([0, 1, 2, 2, 3, 4], np.int32),
        "loop_starts": np.array([0, 3], np.int32),
    }
    me = bpy.data.meshes.new("vgeo.write-check")
    try:
        me.vertices.add(4)
        me.edges.add(5)
        me.loops.add(6)
        me.polygons.add(2)
        _write_fast(me, d)
        me.update()

        def get(seq, prop, n):
            out = np.empty(n, np.float32 if prop == "co" else np.int32)
            seq.foreach_get(prop, out)
            return out
        return (np.array_equal(get(me.vertices, "co", 12), d["positions"])
                and np.array_equal(get(me.edges, "vertices", 10), d["edge_verts"])
                and np.array_equal(get(me.loops, "vertex_index", 6), d["corner_verts"])
                and np.array_equal(get(me.loops, "edge_index", 6), d["corner_edges"])
                and np.array_equal(get(me.polygons, "loop_start", 2), d["loop_starts"])
                and np.array_equal(get(me.polygons, "loop_total", 2), np.array([3, 3], np.int32))
                and not me.validate(verbose=False))
    except Exception:
        return False
    finally:
        bpy.data.meshes.remove(me)


def _set_attr(me, name, kind, domain, prop, arr):
    a = me.attributes.new(name, kind, domain)
    if _fast_write:
        _copy_into(a.data, arr)
    else:
        a.data.foreach_set(prop, arr)
    return a


_EXTRA_PROPS = {"FLOAT2": "vector", "FLOAT_COLOR": "color", "FLOAT": "value", "FLOAT_VECTOR": "vector"}


def _write_extras(me, data, desc, uv_name):
    """The other UV maps and colour/float attributes of the source, the active ones marked as they were, and
    the source's texture space (each chunk is its own mesh: left automatic, Generated coordinates would
    be fitted to every chunk separately)."""
    ex = data.get("extras")
    if ex is not None:
        cv = data["corner_verts"]
        for ch in desc.get("channels", ()):
            kind = ch.get("type", "FLOAT")
            if kind not in _EXTRA_PROPS:
                continue
            o, n = int(ch["offset"]), int(ch["size"])
            vals = np.ascontiguousarray(ex[:, o:o + n][cv], dtype=np.float32).ravel()
            try:
                _set_attr(me, ch["name"], kind, 'CORNER', _EXTRA_PROPS[kind], vals)
            except (RuntimeError, TypeError, ValueError):
                pass                                  # a clashing name: skip, never break the stream
    if uv_name in me.uv_layers:
        me.uv_layers.active = me.uv_layers[uv_name]
    ru = desc.get("render_uv")
    if ru and ru in me.uv_layers:
        me.uv_layers[ru].active_render = True
    cols = me.color_attributes
    if desc.get("active_color") and desc["active_color"] in cols:
        cols.active_color = cols[desc["active_color"]]
    if desc.get("render_color") and desc["render_color"] in cols and hasattr(cols, "render_color_index"):
        cols.render_color_index = list(cols).index(cols[desc["render_color"]])
    ts = desc.get("texspace")
    if ts and hasattr(me, "texspace_location"):
        me.use_auto_texspace = False
        me.texspace_location = ts[0]
        me.texspace_size = ts[1]


def fill_mesh(me, data, materials, lod_colors):
    """Replace a mesh's geometry with an extracted chunk."""
    global _fast_write
    me.clear_geometry()
    if data is None:
        return
    nv, nt, ne = data["vertex_count"], data["tri_count"], data["edge_count"]
    data["loop_starts"] = np.arange(0, nt * 3, 3, dtype=np.int32)
    if _fast_write is None:
        _fast_write = _fast_write_works()
        if not _fast_write:
            print("VGEO: direct mesh writes unavailable, using foreach_set")
    me.vertices.add(nv)
    me.edges.add(ne)
    me.loops.add(nt * 3)
    me.polygons.add(nt)
    if _fast_write:
        _write_fast(me, data)
    else:
        _write_slow(me, data)

    _set_attr(me, "custom_normal", 'FLOAT_VECTOR', 'POINT', "vector", data["normals"])
    desc = data.get("desc") or {}
    uv_name = desc.get("uv_name") or "UVMap"
    if data["uvs"] is not None:
        corner_uv = np.ascontiguousarray(data["uvs"].reshape(-1, 2)[data["corner_verts"]]).ravel()
        _set_attr(me, uv_name, 'FLOAT2', 'CORNER', "vector", corner_uv)
    _write_extras(me, data, desc, uv_name)
    if len(materials) > 1:
        _set_attr(me, "material_index", 'INT', 'FACE', "value", data["face_materials"])
    if lod_colors:
        lod = np.repeat(data["face_lod"] % len(LOD_PALETTE), 3)
        col = _set_attr(me, "vgeo_lod", 'FLOAT_COLOR', 'CORNER', "color",
                        np.ascontiguousarray(LOD_PALETTE[lod]).ravel())
        me.color_attributes.active_color = col
    me.update()


def _sync_materials(me, materials):
    if tuple(me.materials) != materials:
        me.materials.clear()
        for m in materials:
            me.materials.append(m)


def _discard_pending(rt):
    """Drop a half-built update; its geometry sits in hidden backs or spares, cleared later."""
    (rt.spare_clear if rt.strategy == 'BATCH' else rt.to_clear).update(rt.pending)
    rt.pending.clear()
    rt.todo = []
    rt.target = None


def _spare_mesh(rt, c):
    """An unlinked mesh for BATCH builds: filling it costs Blender nothing per frame."""
    name = rt.spare.get(c)
    me = bpy.data.meshes.get(name) if name else None
    if me is None or me.users != 0:
        me = bpy.data.meshes.new(chunk_name(rt.uid, c) + "^")
        rt.spare[c] = me.name
    return me


def _clear_spares(rt, limit=None):
    for c in list(rt.spare_clear)[:limit]:
        rt.spare_clear.discard(c)
        me = bpy.data.meshes.get(rt.spare.get(c, ""))
        if me is not None and me.users == 0 and len(me.vertices):
            me.clear_geometry()


def clear_backs(obj, rt):
    """Empty every hidden back and spare now (before renders and saves)."""
    _discard_pending(rt)
    for pair in chunk_objects(obj, rt):
        _front, back = front_back(pair)
        if len(back.data.vertices):
            back.data.clear_geometry()
    rt.to_clear.clear()
    rt.spare_clear.update(rt.spare.keys())
    _clear_spares(rt)


def stream_step(obj, views, pixel_error, mode="COARSEN", offscreen_scale=8.0, budget=0.012,
                strategy='STAGED'):
    """Incremental update for the live loop; returns True while work remains.

    The current (valid) cut stays on screen while the changed chunks are
    rebuilt a few per tick; when all are ready they land together, so the
    surface never shows a mix of two cuts. The native selection is not touched
    until then, so extraction keeps reading the target cut.

    strategy STAGED (Solid/Workbench): build into the hidden backs, which are
      drawn at zero size so Blender prepares them as they fill; landing is a
      scale flip. Steady frames, no spike.
    strategy BATCH: build into unlinked spare meshes and land them in one frame
      by reassigning the fronts' mesh data. EEVEE needed it while chunks were
      instanced through Geometry Nodes; now it only serves callers that want
      the fewest geometry-changing frames (its landing frame is the dearest).
    """
    rt = runtime_for(obj)
    if rt.asset is None:
        return False
    if rt.target is not None and rt.strategy != strategy:
        _discard_pending(rt)  # the viewport changed mode mid-update: start over
    if rt.spare_clear:
        _clear_spares(rt, 64)  # unlinked meshes: clearing them is cheap
    if rt.target is None:
        key = _key(obj, views, pixel_error, mode, offscreen_scale)
        if key == rt.key and rt.valid.all():
            if rt.to_clear:  # idle: release old geometry from hidden backs, a few per tick
                chunks = chunk_objects(obj, rt)
                for _ in range(min(32, len(rt.to_clear))):
                    _front, back = front_back(chunks[rt.to_clear.pop()])
                    back.data.clear_geometry()
                return bool(rt.to_clear)
            return bool(rt.spare_clear)
        lv = local_views(obj, views, pixel_error, mode, offscreen_scale)
        if not lv:
            return False
        t0 = time.perf_counter()
        sigs = rt.asset.select(lv)
        lod_colors = bool(obj.vgeo.lod_colors)
        if rt.lod_colors != lod_colors:
            rt.valid[:] = False
            rt.lod_colors = lod_colors
        changed = np.nonzero(~rt.valid | (sigs != rt.applied))[0]
        rt.target_key = key
        rt.target_stats = (int(rt.asset.last.triangles), int(rt.asset.last.clusters))
        rt.target_t0 = t0
        if len(changed) == 0:
            rt.key = key
            rt.triangles, rt.clusters = rt.target_stats
            return False
        rt.target = sigs.copy()
        rt.todo = [int(c) for c in changed[::-1]]  # pop() takes them in order
        rt.asset.prefetch(changed)  # the file is mapped: start reading pages the build will touch
        rt.build_s = 0.0
        rt.strategy = strategy
        if strategy == 'BATCH' and len(rt.spare) < rt.asset.chunk_count:
            # create every spare now, with its materials, in one tick: creating datablocks and
            # assigning materials both force a depsgraph relations rebuild (and a full EEVEE
            # re-sync), so neither may trickle through the build
            materials = tuple(obj.data.materials)
            for c in range(rt.asset.chunk_count):
                _sync_materials(_spare_mesh(rt, c), materials)

    materials = tuple(obj.data.materials)
    if materials != getattr(rt, "materials", None):
        # the proxy's materials changed: resync every chunk mesh and spare in one go
        rt.chunk_cache = None
        for name in rt.spare.values():
            me = bpy.data.meshes.get(name)
            if me is not None:
                _sync_materials(me, materials)
    chunks = chunk_objects(obj, rt)
    # release old geometry from hidden backs even while building, so continuous motion never
    # leaves a second copy of the cut being drawn (at zero size, but through every render pass)
    if rt.to_clear:
        todo = set(rt.todo)
        for c in [c for c in rt.to_clear if c not in todo][:64]:
            rt.to_clear.discard(c)
            back = front_back(chunks[c])[1]
            if len(back.data.vertices):
                back.data.clear_geometry()
    t_start = time.perf_counter()
    built = 0
    # at least one chunk per tick, so an update can never stall
    while rt.todo and (built == 0 or time.perf_counter() - t_start < budget):
        built += 1
        c = rt.todo.pop()
        me = _spare_mesh(rt, c) if strategy == 'BATCH' else front_back(chunks[c])[1].data
        _sync_materials(me, materials)
        fill_mesh(me, rt.asset.extract(c), materials, rt.lod_colors)
        if WARM_CACHES and len(me.polygons):
            # compute triangulation + normals now (spread over ticks); the evaluated copy
            # shares these caches, so the flip frame does not have to
            me.loop_triangles[0]
            me.corner_normals[0]
        rt.pending.add(c)
        if strategy == 'BATCH':
            rt.spare_clear.discard(c)
        else:
            rt.to_clear.discard(c)
    rt.build_s += time.perf_counter() - t_start
    if rt.todo:
        return True

    # everything is built: land all changed chunks in one go
    global swap_count
    swap_count += 1
    rt.swap_tris = 0
    for c in rt.pending:
        front, back = front_back(chunks[c])
        if strategy == 'BATCH':
            spare = bpy.data.meshes.get(rt.spare.get(c, ""))
            if spare is None:  # lost to undo: start over
                _discard_pending(rt)
                rt.invalidate()
                return True
            old = front.data
            front.data = spare
            rt.spare[c] = old.name  # the replaced mesh becomes this chunk's next spare
            rt.spare_clear.add(c)
            rt.swap_tris += len(spare.polygons)
        else:
            _set_shown(back, True)
            _set_shown(front, False)
            rt.to_clear.add(c)  # the old front is now a hidden back
            rt.swap_tris += len(back.data.polygons)
        rt.applied[c] = rt.target[c]
        rt.valid[c] = True
    rt.last_rebuilt = len(rt.pending)
    if DIAG_SWAP:
        t_dg = time.perf_counter()
        bpy.context.view_layer.update()
        rt.swap_dg_ms = (time.perf_counter() - t_dg) * 1000.0
    rt.pending.clear()
    rt.target = None
    rt.key = rt.target_key
    rt.triangles, rt.clusters = rt.target_stats
    rt.last_ms = rt.build_s * 1000.0
    rt.latency_ms = (time.perf_counter() - rt.target_t0) * 1000.0
    rt.updates += 1
    return bool(rt.to_clear or rt.spare_clear)  # old geometry is released on following ticks


def apply_cut(obj, views, pixel_error, mode="FULL", offscreen_scale=8.0, force=False):
    """Select a cut for these (world-space) views and write changed chunks now.

    Used for renders and scripts. mode: FULL, COARSEN (off-screen detail
    reduced by offscreen_scale) or CULL. Returns the Runtime (with stats).
    """
    rt = runtime_for(obj)
    if rt.asset is None:
        return rt
    _discard_pending(rt)
    key = _key(obj, views, pixel_error, mode, offscreen_scale)
    if not force and key == rt.key and rt.valid.all():
        return rt
    t0 = time.perf_counter()
    lv = local_views(obj, views, pixel_error, mode, offscreen_scale)
    if not lv:
        return rt
    sigs = rt.asset.select(lv)
    lod_colors = bool(obj.vgeo.lod_colors)
    if rt.lod_colors != lod_colors:
        rt.valid[:] = False
        rt.lod_colors = lod_colors
    changed = np.nonzero(~rt.valid | (sigs != rt.applied))[0]
    rt.asset.prefetch(changed)
    materials = tuple(obj.data.materials)
    chunks = chunk_objects(obj, rt)
    for c in changed:  # fronts are rewritten in place: atomic within this call
        me = front_back(chunks[c])[0].data
        _sync_materials(me, materials)
        fill_mesh(me, rt.asset.extract(int(c)), materials, lod_colors)
        rt.applied[c] = sigs[c]
        rt.valid[c] = True
    if len(changed) == 0:
        for pair in chunks:
            _sync_materials(front_back(pair)[0].data, materials)
    clear_backs(obj, rt)  # renders must not carry hidden geometry
    rt.key = key
    rt.triangles = int(rt.asset.last.triangles)
    rt.clusters = int(rt.asset.last.clusters)
    rt.last_rebuilt = int(len(changed))
    rt.last_ms = (time.perf_counter() - t0) * 1000.0
    rt.updates += 1
    return rt


def apply_level(obj, depth):
    """Write a fixed DAG level (0 = full detail, -1 = coarsest). For previews and tests."""
    rt = runtime_for(obj)
    if rt.asset is None:
        return rt
    _discard_pending(rt)
    sigs = rt.asset.select_level(depth)
    chunks = chunk_objects(obj, rt)
    materials = tuple(obj.data.materials)
    for c in range(rt.asset.chunk_count):
        me = front_back(chunks[c])[0].data
        _sync_materials(me, materials)
        fill_mesh(me, rt.asset.extract(c), materials, bool(obj.vgeo.lod_colors))
        rt.applied[c] = sigs[c]
    clear_backs(obj, rt)
    rt.valid[:] = True
    rt.key = None
    rt.triangles = int(rt.asset.last.triangles)
    rt.clusters = int(rt.asset.last.clusters)
    return rt


def update_for_render(scene, depsgraph=None):
    view = camera_view(scene, depsgraph)
    if view is None:
        return
    from . import instances
    # instancers first: they choose which copies get a streamed cut and make it now
    for inst in instances.instancers(scene):
        if not inst.hide_render:
            instances.update(inst, [view], inst.vgeo_inst.render_pixel_error, render=True)
    for obj in proxies(scene):
        if obj.vgeo.collection is not None:
            sync_visibility(obj)
        if obj.hide_render or obj.vgeo.slot_of:  # streamed copies were cut by their instancer
            continue
        # final renders never cull: off-screen geometry still casts shadows and shows in reflections
        mode = "FULL" if obj.vgeo.offscreen == "FULL" else "COARSEN"
        apply_cut(obj, [view], obj.vgeo.render_pixel_error, mode, obj.vgeo.offscreen_scale)


# ---------------------------------------------------------------- live loop

_last_orphan_check = 0.0


def _orphan_check():
    """At most once a second: returns True when the check ran."""
    global _last_orphan_check
    now = time.perf_counter()
    if now - _last_orphan_check < 1.0:
        return False
    _last_orphan_check = now
    from . import instances
    instances.unlink_orphan_slots()
    unlink_orphans()
    return True


def _housekeeping(objs, vl):
    """Selection, visibility and collection links of the chunks follow the proxy."""
    redirect_selection(vl)
    for obj in proxies():
        if obj.vgeo.collection is not None:
            sync_visibility(obj, vl)
    if _orphan_check():
        for obj in objs:
            col = obj.vgeo.collection
            if col is not None and obj.library is None and any(
                    col.name not in c.children for c in obj.users_collection):
                link_chunks(obj)


def _tick():
    if _rendering:
        return 0.25
    try:
        from . import instances
        # idle streamed copies (slot_index < 0) belong to an instancer and show nothing
        objs = [o for o in proxies() if not o.vgeo.freeze and not (o.vgeo.slot_of and o.vgeo.slot_index < 0)]
        insts = [o for o in instances.instancers() if not o.vgeo_inst.freeze]
        if not objs and not insts:
            _orphan_check()  # a deleted proxy's chunks must not linger
            return 0.5
        vl = bpy.context.view_layer
        _housekeeping(objs, vl)
        views = viewport_views()
        if not views:
            return 0.25
        for inst in insts:   # per-placement level choice: cheap, no geometry is rebuilt
            try:
                if vl is None or inst.visible_get(view_layer=vl):
                    instances.update(inst, views, inst.vgeo_inst.pixel_error)
            except RuntimeError:
                continue
        if not objs:
            return TICK
        visible = []
        for obj in objs:
            try:
                if vl is None or obj.visible_get(view_layer=vl):
                    visible.append(obj)
            except RuntimeError:
                continue
        strategy = viewport_strategy()
        if strategy == 'CYCLES':
            # Cycles rebuilds its BVH on every geometry change and restarts sampling anyway:
            # update once the view has been still for a moment, in one step
            now = time.perf_counter()
            for obj in visible:
                v = obj.vgeo
                rt = runtime_for(obj)
                if rt.asset is None:
                    continue
                key = _key(obj, views, v.pixel_error, v.offscreen, v.offscreen_scale)
                if key != getattr(rt, "settle_key", None):
                    rt.settle_key, rt.settle_t = key, now
                elif now - rt.settle_t >= CYCLES_SETTLE and (key != rt.key or not rt.valid.all()):
                    apply_cut(obj, views, v.pixel_error, v.offscreen, v.offscreen_scale)
            for inst in insts:
                instances.sync_slots(inst)
            return TICK
        busy = False
        budget = TICK_BUDGET / max(1, len(visible))
        for obj in visible:
            v = obj.vgeo
            busy |= stream_step(obj, views, v.pixel_error, v.offscreen, v.offscreen_scale, budget, strategy)
        for inst in insts:   # copies whose streamed cut just landed take over in this same tick
            instances.sync_slots(inst)
        # keep cranking (UI events still run between ticks) until the swap lands
        return 0.0 if busy else TICK
    except Exception as e:  # never let the timer die
        print("VGEO stream:", e)
        return 1.0


@persistent
def _on_render_pre(scene, depsgraph=None):
    global _rendering
    _rendering = True
    try:
        update_for_render(scene, depsgraph)
    except Exception as e:
        print("VGEO render cut failed:", e)


@persistent
def _on_render_done(scene, depsgraph=None):
    global _rendering
    _rendering = False
    invalidate_all()


@persistent
def _on_load(_a=None, _b=None):
    invalidate_all(close=True)


@persistent
def _on_save(_a=None, _b=None):
    # hidden backs would double the saved geometry; only the shown cut goes into the file
    for obj in proxies():
        rt = _runtimes.get(obj.vgeo.uid)
        if rt is not None and rt.asset is not None:
            try:
                clear_backs(obj, rt)
            except Exception as e:
                print("VGEO save:", e)


@persistent
def _on_undo(_a=None, _b=None):
    invalidate_all()


def new_uid():
    return uuid.uuid4().hex[:12]


def register():
    h = bpy.app.handlers
    for lst, fn in ((h.render_pre, _on_render_pre), (h.render_complete, _on_render_done),
                    (h.render_cancel, _on_render_done), (h.load_post, _on_load), (h.save_pre, _on_save),
                    (h.undo_post, _on_undo), (h.redo_post, _on_undo)):
        if fn not in lst:
            lst.append(fn)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.5, persistent=True)


def unregister():
    h = bpy.app.handlers
    for lst, fn in ((h.render_pre, _on_render_pre), (h.render_complete, _on_render_done),
                    (h.render_cancel, _on_render_done), (h.load_post, _on_load), (h.save_pre, _on_save),
                    (h.undo_post, _on_undo), (h.redo_post, _on_undo)):
        while fn in lst:
            lst.remove(fn)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    invalidate_all(close=True)
