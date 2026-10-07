"""Many copies of one virtualized asset: full scenes from dense assets.

An instancer is a mesh object whose vertices are placements (with optional
per-point rotation and scale attributes). The asset's uniform LOD levels are
built once into shared "level meshes" (each is a complete, crack-free cut).
Every update picks, per placement, the coarsest level whose geometric error
stays under the pixel threshold for the current views, and writes it to an
integer attribute; a Geometry Nodes tree instances the chosen level mesh at
each point. Navigating never rebuilds geometry, and EEVEE and Cycles get
real instances (one copy of each level in memory, however many placements).

Whole-asset levels waste triangles on a copy seen up close: level 0 is the
entire asset at full detail, though most of it is far away or off-screen.
The few copies that want level 0 (`stream_slots`, nearest first) are shown
by streamed copies instead: hidden-from-selection VGEO proxies parented to
the instancer, placed like the copy, streaming a view-dependent cut through
the normal live loop. A copy is only switched over (its instance dropped via
the `vgeo_streamed` attribute) once its streamed cut has landed, in the same
tick, so the surface never shows a gap or two overlapping versions.
"""

import math

import bpy
import numpy as np

from . import native, stream

NODE_GROUP = "VGEO Instances"
LEVEL_ATTR = "vgeo_level"
ROT_ATTR = "vgeo_rot"
SCALE_ATTR = "vgeo_scale"
STREAMED_ATTR = "vgeo_streamed"   # bool per point: shown by a streamed copy, not an instance
STREAM_MIN_TRIS = 20_000          # below this, a whole-asset level 0 is cheap enough

_levels_cache = {}   # instancer uid -> (errors ndarray, tris ndarray)


def instancers(scene=None):
    objs = scene.objects if scene is not None else bpy.data.objects
    return [o for o in objs if o.type == 'MESH' and o.vgeo_inst.uid and o.vgeo_inst.levels is not None]


def ensure_node_group():
    ng = bpy.data.node_groups.get(NODE_GROUP)
    if ng and ng.bl_idname == "GeometryNodeTree" and "Levels" in ng.interface.items_tree:
        _ensure_streamed_selection(ng)
        return ng
    ng = bpy.data.node_groups.new(NODE_GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Levels", in_out="INPUT", socket_type="NodeSocketCollection")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, L = ng.nodes, ng.links
    gin = n.new("NodeGroupInput")
    info = n.new("GeometryNodeCollectionInfo")
    info.transform_space = "ORIGINAL"
    info.inputs["Separate Children"].default_value = True
    info.inputs["Reset Children"].default_value = True
    L.new(gin.outputs["Levels"], info.inputs["Collection"])

    def named(name, kind):
        a = n.new("GeometryNodeInputNamedAttribute")
        a.data_type = kind
        a.inputs["Name"].default_value = name
        return a

    lvl = named(LEVEL_ATTR, 'INT')
    rot = named(ROT_ATTR, 'FLOAT_VECTOR')
    scl = named(SCALE_ATTR, 'FLOAT')
    e2r = n.new("FunctionNodeEulerToRotation")
    L.new(rot.outputs["Attribute"], e2r.inputs[0])
    iop = n.new("GeometryNodeInstanceOnPoints")
    L.new(gin.outputs["Geometry"], iop.inputs["Points"])
    L.new(info.outputs["Instances"], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    L.new(lvl.outputs["Attribute"], iop.inputs["Instance Index"])
    L.new(e2r.outputs[0], iop.inputs["Rotation"])
    L.new(scl.outputs["Attribute"], iop.inputs["Scale"])
    out = n.new("NodeGroupOutput")
    L.new(iop.outputs["Instances"], out.inputs["Geometry"])
    _ensure_streamed_selection(ng)
    return ng


def _ensure_streamed_selection(ng):
    """Placements shown by a streamed copy are left out: Selection = not vgeo_streamed.
    Added in place, so node groups from older files keep working."""
    if ng.nodes.get("VGEO streamed") is not None:
        return
    iop = next((x for x in ng.nodes if x.bl_idname == "GeometryNodeInstanceOnPoints"), None)
    if iop is None:
        return
    a = ng.nodes.new("GeometryNodeInputNamedAttribute")
    a.name = a.label = "VGEO streamed"
    a.data_type = 'BOOLEAN'
    a.inputs["Name"].default_value = STREAMED_ATTR
    inv = ng.nodes.new("FunctionNodeBooleanMath")
    inv.operation = 'NOT'
    ng.links.new(a.outputs["Attribute"], inv.inputs[0])
    ng.links.new(inv.outputs[0], iop.inputs["Selection"])


def _socket_id(ng, name):
    return next(i.identifier for i in ng.interface.items_tree
                if getattr(i, "in_out", None) == "INPUT" and i.name == name)


def build_levels(inst, source):
    """Build (or rebuild) the shared level meshes for an instancer from a VGEO proxy."""
    path = stream.asset_path(source)
    asset = native.Asset(path)   # own handle: never disturbs the proxy's live selection
    try:
        depth_errors = asset.level_errors()
        uid = inst.vgeo_inst.uid
        col = inst.vgeo_inst.levels
        if col is None:
            col = bpy.data.collections.new(f".vgeo levels {uid}")
            inst.vgeo_inst.levels = col
        for ob in list(col.objects):   # previous build
            me = ob.data
            bpy.data.objects.remove(ob)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        materials = tuple(source.data.materials)
        tris, errors = [], []
        for level, (err, data) in enumerate(_error_levels(asset, depth_errors)):
            name = f"vgeo.{uid}.L{level:02d}"
            me = bpy.data.meshes.new(name)
            stream._sync_materials(me, materials)
            stream.fill_mesh(me, data, materials, False)
            ob = bpy.data.objects.new(name, me)
            col.objects.link(ob)
            tris.append(data["tri_count"] if data else 0)
            errors.append(err)
        lo, hi = np.array(asset.info["aabb_min"]), np.array(asset.info["aabb_max"])
        center = (lo + hi) / 2
        inst["vgeo_level_errors"] = [float(e) for e in errors]
        inst["vgeo_level_tris"] = [int(t) for t in tris]
        # bounding sphere around the asset origin (placements put the origin on the point)
        inst["vgeo_radius"] = float(np.linalg.norm(center) + np.linalg.norm(hi - lo) / 2)
        _levels_cache.pop(uid, None)
    finally:
        asset.close()
    _attach(inst)


LEVEL_STEP = 2.0     # each level allows twice the error of the one before


def _error_levels(asset, depth_errors):
    """(error, mesh data) per level: level 0 is the full asset, then the cheapest crack-free cut whose
    error stays under e, 2e, 4e, ... up to the coarsest. Uniform DAG depths took the worst cluster's error
    for the whole level (a displaced rock's depth 5 was 476k triangles for 7.7 mm, depth 6 jumped to
    4.6 cm), so copies sat at needlessly fine levels: ~80 triangles per pixel in a 1,000-copy field."""
    finite = sorted(e for e in depth_errors if 0 < e < 1e30)
    if not finite:
        for depth in range(len(depth_errors)):
            yield depth_errors[depth], asset.level_mesh_data(depth)
        return
    yield 0.0, asset.level_mesh_data(0)
    e, last_tris = finite[0] / LEVEL_STEP, None
    top = finite[-1] * LEVEL_STEP
    while e <= top:
        data = asset.error_cut_mesh_data(e)
        n = data["tri_count"] if data else 0
        if n and n != last_tris:
            yield e, data
            last_tris = n
        e *= LEVEL_STEP
    coarsest = asset.level_mesh_data(len(depth_errors) - 1)
    if coarsest and coarsest["tri_count"] != last_tris:
        yield max(top, depth_errors[-1] if depth_errors[-1] < 1e30 else top), coarsest


def _attach(inst):
    ng = ensure_node_group()
    mod = next((m for m in inst.modifiers if m.type == 'NODES' and m.node_group == ng), None)
    if mod is None:
        mod = inst.modifiers.new("VGEO Instances", 'NODES')
        mod.node_group = ng
    from . import mod_inputs
    mod_inputs.of(mod)[_socket_id(ng, "Levels")] = inst.vgeo_inst.levels


def _level_tables(inst):
    uid = inst.vgeo_inst.uid
    t = _levels_cache.get(uid)
    if t is None:
        t = (np.array(inst.get("vgeo_level_errors", [0.0]), dtype=np.float64),
             np.array(inst.get("vgeo_level_tris", [0]), dtype=np.int64))
        _levels_cache[uid] = t
    return t


def _points(inst):
    me = inst.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    scale = np.ones(n, np.float32)
    a = me.attributes.get(SCALE_ATTR)
    if a is not None and a.domain == 'POINT' and a.data_type == 'FLOAT':
        a.data.foreach_get("value", scale)
    return co.reshape(-1, 3), scale


def choose_levels(inst, views, pixel_error, mode="COARSEN", offscreen_scale=8.0):
    """Per placement, the coarsest level that keeps the error under the threshold in every view."""
    return _choose(inst, views, pixel_error, mode, offscreen_scale)[0]


def _choose(inst, views, pixel_error, mode="COARSEN", offscreen_scale=8.0):
    """(levels, allowed error in asset units) per placement."""
    errors, _tris = _level_tables(inst)
    co, scale = _points(inst)
    n = len(co)
    if n == 0:
        return np.zeros(0, np.int32), np.zeros(0)
    mw = np.array(inst.matrix_world, dtype=np.float64)
    world = co @ mw[:3, :3].T + mw[:3, 3]
    s = scale.astype(np.float64) * max(abs(x) for x in inst.matrix_world.to_scale())
    radius = float(inst.get("vgeo_radius", 1.0)) * s
    budget = np.full(n, np.inf)
    for view, window, height, persp, clip_start in views:
        cam = np.array(view.inverted().translation, dtype=np.float64)
        t = pixel_error / max(1, height)
        if persp:
            d = np.maximum(np.linalg.norm(world - cam, axis=1) - radius, max(clip_start, 1e-6))
            b = t * d / (window[1][1] * 0.5)
        else:
            b = np.full(n, t * (2.0 / window[1][1] if window[1][1] else 1.0))
        if mode != "FULL":
            planes = np.array(stream._frustum_planes(window @ view), dtype=np.float64)
            inside = np.all(world @ planes[:, :3].T + planes[:, 3] >= -radius[:, None], axis=1)
            if mode == "COARSEN":
                b = np.where(inside, b, b * offscreen_scale)
            else:   # CULL has no meaning for instances that cast shadows; treat as coarsen
                b = np.where(inside, b, b * max(offscreen_scale, 64.0))
        budget = np.minimum(budget, b / s)   # error allowed in asset units
    levels = np.searchsorted(errors, budget, side="right") - 1
    levels = np.clip(levels, int(inst.vgeo_inst.min_level), len(errors) - 1)
    return levels.astype(np.int32), budget


def apply_levels(inst, levels):
    """Write the level attribute if it changed. Returns True when geometry nodes must re-evaluate."""
    me = inst.data
    a = me.attributes.get(LEVEL_ATTR)
    if a is None or a.domain != 'POINT' or a.data_type != 'INT':
        if a is not None:
            me.attributes.remove(a)
        a = me.attributes.new(LEVEL_ATTR, 'INT', 'POINT')
    cur = np.empty(len(me.vertices), np.int32)
    a.data.foreach_get("value", cur)
    if np.array_equal(cur, levels):
        return False
    a.data.foreach_set("value", levels)
    me.update()
    return True


def triangles_for(inst, levels, streamed=None):
    _errors, tris = _level_tables(inst)
    if not len(levels):
        return 0
    if streamed is not None and streamed.any():
        return int(tris[levels[~streamed]].sum())
    return int(tris[levels].sum())


MIN_INTERVAL = 0.2   # s between level writes while the view moves (each write re-evaluates the instancer)
_last_write = {}     # instancer uid -> perf_counter of its last level write


def _settle(inst, levels):
    """Viewport only: refine at once wherever a copy is too coarse, but coarsen a copy only once it's two
    levels finer than needed, and write at most every MIN_INTERVAL. Flying through a 1,000-copy field,
    some copy crossed a level boundary nearly every tick, and every write re-evaluated the instancer
    (~45 ms a frame in EEVEE/Solid, against 27 ms for drawing it)."""
    import time
    me = inst.data
    a = me.attributes.get(LEVEL_ATTR)
    if a is None or a.domain != 'POINT' or a.data_type != 'INT' or len(a.data) != len(levels):
        return levels
    cur = np.empty(len(levels), np.int32)
    a.data.foreach_get("value", cur)
    keep = (cur <= levels) & (cur >= levels - 1)       # fine enough and at most one level too fine
    out = np.where(keep, cur, levels).astype(np.int32)
    if not np.array_equal(out, cur):
        now = time.perf_counter()
        uid = inst.vgeo_inst.uid
        if now - _last_write.get(uid, 0.0) < MIN_INTERVAL:
            return cur
        _last_write[uid] = now
    return out


def update(inst, views, pixel_error, mode="COARSEN", offscreen_scale=8.0, render=False, live=False):
    """Choose levels and streamed copies for these views. render=True: the streamed copies get
    their cut now and take over at once (final renders are synchronous). live=True (the viewport loop):
    level changes settle (see _settle) instead of landing every tick."""
    levels, budget = _choose(inst, views, pixel_error, mode, offscreen_scale)
    if live and not render:
        levels = _settle(inst, levels)
    changed = apply_levels(inst, levels)
    wanted = _wanted_slots(inst, levels, budget)
    changed |= assign_slots(inst, wanted, views if render else None, pixel_error, mode, offscreen_scale)
    streamed = _streamed_flags(inst)
    shown = triangles_for(inst, levels, streamed)
    for ob in slots(inst):
        rt = stream._runtimes.get(ob.vgeo.uid)
        if ob.vgeo.slot_index >= 0 and rt is not None and rt.asset is not None and streamed[ob.vgeo.slot_index]:
            shown += rt.triangles
    inst.vgeo_inst.shown_triangles = shown
    return changed


# ---------------------------------------------------------------- streamed copies

def slots(inst):
    """The streamed copies this instancer manages (children marked with its uid)."""
    uid = inst.vgeo_inst.uid
    return [o for o in inst.children if o.vgeo.slot_of == uid and o.vgeo.uid]


def _slot_name(inst, k):
    return f"vgeo.{inst.vgeo_inst.uid}.S{k:02d}"


def ensure_slots(inst):
    """Create or drop streamed copies to match stream_slots. Returns them in slot order."""
    src = inst.vgeo_inst.source
    want = int(inst.vgeo_inst.stream_slots) if src is not None and src.vgeo.path else 0
    have = sorted(slots(inst), key=lambda o: o.name)
    for ob in have[want:]:
        _release_slot(inst, ob)
        _remove_slot(ob)
    have = have[:want]
    names = {o.name for o in have}
    for k in range(want):
        name = _slot_name(inst, k)
        if name in names:
            continue
        me = bpy.data.meshes.new(name)
        stream._sync_materials(me, tuple(src.data.materials))
        ob = bpy.data.objects.new(name, me)
        for col in inst.users_collection:
            col.objects.link(ob)
        ob.parent = inst
        ob.matrix_parent_inverse.identity()
        ob.hide_select = True
        v = ob.vgeo
        v.uid = f"{inst.vgeo_inst.uid}s{k}"
        v.path = src.vgeo.path
        v.slot_of = inst.vgeo_inst.uid
        v.slot_index = -1         # idle: the live loop skips it
        v.offscreen = 'COARSEN'
        have.append(ob)
    for ob in have:
        v = ob.vgeo
        if v.pixel_error != inst.vgeo_inst.pixel_error:
            v.pixel_error = inst.vgeo_inst.pixel_error
        if v.render_pixel_error != inst.vgeo_inst.render_pixel_error:
            v.render_pixel_error = inst.vgeo_inst.render_pixel_error
        if ob.hide_render != inst.hide_render:
            ob.hide_render = inst.hide_render
        if ob.hide_viewport != inst.hide_viewport:
            ob.hide_viewport = inst.hide_viewport
        try:
            if ob.hide_get() != inst.hide_get():
                ob.hide_set(inst.hide_get())
        except RuntimeError:  # not in the current view layer
            pass
    return sorted(have, key=lambda o: o.name)


def _remove_slot(ob):
    col = ob.vgeo.collection
    rt = stream._runtimes.pop(ob.vgeo.uid, None)
    if rt:
        rt.close()
    if col is not None:
        for c in list(col.objects):
            me = c.data
            bpy.data.objects.remove(c)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        bpy.data.collections.remove(col)
    me = ob.data
    bpy.data.objects.remove(ob)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def _wanted_slots(inst, levels, budget):
    """Placements that should be streamed: the level they want is a big whole-asset mesh (a view-dependent
    cut of their own is lighter), nearest first. With error-based levels a near copy wants a fine level
    rather than exactly level 0, so the test is the level's size, not its number."""
    k = int(inst.vgeo_inst.stream_slots)
    _errors, tris = _level_tables(inst)
    if k <= 0 or not len(levels) or tris[0] < STREAM_MIN_TRIS or int(inst.vgeo_inst.min_level) > 0:
        return []
    cand = np.nonzero(np.asarray(tris)[levels] >= STREAM_MIN_TRIS)[0]
    if not len(cand):
        return []
    return [int(i) for i in cand[np.argsort(budget[cand], kind="stable")][:k]]


def _streamed_flags(inst):
    me = inst.data
    n = len(me.vertices)
    a = me.attributes.get(STREAMED_ATTR)
    out = np.zeros(n, bool)
    if a is not None and a.domain == 'POINT' and a.data_type == 'BOOLEAN':
        a.data.foreach_get("value", out)
    return out


def _write_streamed(inst, flags):
    me = inst.data
    a = me.attributes.get(STREAMED_ATTR)
    if a is None or a.domain != 'POINT' or a.data_type != 'BOOLEAN':
        if a is not None:
            me.attributes.remove(a)
        a = me.attributes.new(STREAMED_ATTR, 'BOOLEAN', 'POINT')
    a.data.foreach_set("value", flags)
    me.update()


def placement_matrix(inst, i):
    """A placement's matrix in the instancer's space (what Instance on Points uses)."""
    from mathutils import Euler, Matrix, Vector
    me = inst.data
    co = Vector(me.vertices[i].co)
    rot = Euler((0.0, 0.0, 0.0))
    a = me.attributes.get(ROT_ATTR)
    if a is not None and a.domain == 'POINT':
        rot = Euler(tuple(a.data[i].vector))
    sc = 1.0
    a = me.attributes.get(SCALE_ATTR)
    if a is not None and a.domain == 'POINT':
        sc = float(a.data[i].value)
    return Matrix.Translation(co) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal((sc, sc, sc, 1.0))


def _clear_fronts(ob):
    if ob.vgeo.collection is None:   # never streamed: no chunk objects to empty (none are created)
        return
    rt = stream.runtime_for(ob)
    if rt.asset is None:
        return
    for pair in stream.chunk_objects(ob, rt):
        for c in pair:
            if len(c.data.vertices):
                c.data.clear_geometry()
    rt.invalidate(keep_objects=True)


def _release_slot(inst, ob, flags=None):
    """Hand a placement back to its instance; empties the copy. Returns True if flags changed."""
    i = ob.vgeo.slot_index
    ob.vgeo.slot_index = -1
    _clear_fronts(ob)
    if flags is not None and 0 <= i < len(flags) and flags[i]:
        flags[i] = False
        return True
    return False


def assign_slots(inst, wanted, render_views=None, pixel_error=1.0, mode="COARSEN", offscreen_scale=8.0):
    """Give the wanted placements a streamed copy (keeping copies that still have one).
    A new assignment starts empty and takes over in sync_slots once its cut has landed;
    with render_views the cut is made now. Returns True if the streamed flags changed."""
    obs = ensure_slots(inst)
    if not obs and not _streamed_flags(inst).any():
        return False
    flags = _streamed_flags(inst)
    before = flags.copy()
    wanted_set = set(wanted)
    kept = set()
    free = []
    for ob in obs:
        i = ob.vgeo.slot_index
        if i in wanted_set and i not in kept:
            kept.add(i)
        else:
            if i >= 0:                     # an idle copy is already empty: releasing it again every
                _release_slot(inst, ob, flags)   # tick emptied hundreds of chunks for nothing
            free.append(ob)
    for i in [i for i in wanted if i not in kept]:
        if not free:
            break
        ob = free.pop(0)
        ob.vgeo.slot_index = i
        # matrix_world, not matrix_basis: it is current at once (a render cut follows right away)
        ob.matrix_world = inst.matrix_world @ placement_matrix(inst, i)
        rt = stream.runtime_for(ob)
        ob["vgeo_slot_updates"] = rt.updates if rt.asset is not None else 0
    # anything flagged without a copy goes back to its instance
    owned = {ob.vgeo.slot_index for ob in obs if ob.vgeo.slot_index >= 0}
    for i in np.nonzero(flags)[0]:
        if int(i) not in owned:
            flags[i] = False
    if render_views is not None:
        for ob in obs:
            if ob.vgeo.slot_index >= 0:
                stream.apply_cut(ob, render_views, ob.vgeo.render_pixel_error, "COARSEN", offscreen_scale)
                flags[ob.vgeo.slot_index] = True
    if not np.array_equal(flags, before):
        _write_streamed(inst, flags)
        return True
    return False


def sync_slots(inst):
    """Switch placements over to their streamed copy once its first cut has landed. Called right
    after the live loop streamed, so the swap happens in the frame the cut appears."""
    obs = slots(inst)
    if not obs:
        return False
    flags = _streamed_flags(inst)
    before = flags.copy()
    for ob in obs:
        i = ob.vgeo.slot_index
        if i < 0 or i >= len(flags) or flags[i]:
            continue
        rt = stream._runtimes.get(ob.vgeo.uid)
        if (rt is not None and rt.asset is not None and rt.target is None and rt.key is not None
                and rt.valid.all() and rt.updates > ob.get("vgeo_slot_updates", 0)):
            flags[i] = True
    if not np.array_equal(flags, before):
        _write_streamed(inst, flags)
        return True
    return False


def unlink_orphan_slots():
    """Streamed copies whose instancer was deleted: take them out of the scene."""
    live = {o.vgeo_inst.uid for o in bpy.data.objects if o.vgeo_inst.uid and o.users}
    for ob in [o for o in bpy.data.objects if o.vgeo.slot_of and o.users]:
        if ob.vgeo.slot_of not in live or ob.parent is None or ob.parent.vgeo_inst.uid != ob.vgeo.slot_of:
            for col in list(ob.users_collection):
                col.objects.unlink(ob)


# ---------------------------------------------------------------- scattering

def _euler_from_matrices(m):
    """XYZ Euler angles from rotation matrices (n, 3, 3), matching Blender's convention."""
    sy = np.sqrt(m[:, 0, 0] ** 2 + m[:, 1, 0] ** 2)
    x = np.arctan2(m[:, 2, 1], m[:, 2, 2])
    y = np.arctan2(-m[:, 2, 0], sy)
    z = np.arctan2(m[:, 1, 0], m[:, 0, 0])
    return np.stack([x, y, z], 1)


def scatter_points(surface, count, seed=0, scale_range=(0.7, 1.3), align=True, depsgraph=None):
    """Area-weighted random placements on a mesh surface (world space), with rotation and scale."""
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    ev = surface.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        T = len(me.loop_triangles)
        if T == 0:
            raise ValueError(f"'{surface.name}' has no faces to scatter on")
        tv = np.empty(T * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tv)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
    finally:
        ev.to_mesh_clear()
    mw = np.array(surface.matrix_world, dtype=np.float64)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    tri = co[tv.reshape(-1, 3)]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    area = np.linalg.norm(cr, axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(T, size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pos = tri[pick, 0] + e1[pick] * u[:, None] + e2[pick] * v[:, None]
    nrm = cr[pick] / np.maximum(area[pick], 1e-12)[:, None]
    spin = rng.random(count) * 2 * math.pi
    if align:
        z = nrm
    else:
        z = np.tile([0.0, 0.0, 1.0], (count, 1))
    ref = np.where(np.abs(z[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = np.cross(z, x)
    c, s = np.cos(spin)[:, None], np.sin(spin)[:, None]
    x, y = x * c + y * s, y * c - x * s
    rot = np.stack([x, y, z], 2)   # columns = local axes in world space
    scale = rng.uniform(scale_range[0], scale_range[1], count)
    return pos.astype(np.float32), _euler_from_matrices(rot).astype(np.float32), scale.astype(np.float32)


def create_instancer(context, source, surface, count, seed=0, scale_range=(0.7, 1.3), align=True):
    pos, rot, scale = scatter_points(surface, count, seed, scale_range, align)
    me = bpy.data.meshes.new(f"{source.name} Instances")
    me.vertices.add(len(pos))
    me.vertices.foreach_set("co", pos.ravel())
    me.attributes.new(ROT_ATTR, 'FLOAT_VECTOR', 'POINT').data.foreach_set("vector", rot.ravel())
    me.attributes.new(SCALE_ATTR, 'FLOAT', 'POINT').data.foreach_set("value", scale)
    me.attributes.new(LEVEL_ATTR, 'INT', 'POINT')
    me.update()
    inst = bpy.data.objects.new(f"{source.name} Instances", me)
    for col in surface.users_collection:
        col.objects.link(inst)
    inst.vgeo_inst.uid = stream.new_uid()
    inst.vgeo_inst.source = source
    inst.vgeo_inst.pixel_error = source.vgeo.pixel_error
    inst.vgeo_inst.render_pixel_error = source.vgeo.render_pixel_error
    build_levels(inst, source)
    return inst
