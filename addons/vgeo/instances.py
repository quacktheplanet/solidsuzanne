"""Many copies of one virtualized asset: full scenes from dense assets.

An instancer is a mesh object whose vertices are placements (with optional
per-point rotation and scale attributes). The asset's uniform LOD levels are
built once into shared "level meshes" (each is a complete, crack-free cut).
Every update picks, per placement, the coarsest level whose geometric error
stays under the pixel threshold for the current views, and writes it to an
integer attribute; a Geometry Nodes tree instances the chosen level mesh at
each point. Navigating never rebuilds geometry, and EEVEE and Cycles get
real instances (one copy of each level in memory, however many placements).

Whole-asset levels waste triangles on a copy seen up close: its level must
be fine enough for its nearest point, so the rest of it (far away or
off-screen) is drawn at that detail too. The copies whose level meshes cost
the most triangles (`stream_slots` of them, at any level) are shown
by streamed copies instead: hidden-from-selection VGEO proxies parented to
the instancer, placed like the copy, streaming a view-dependent cut through
the normal live loop. A copy is only switched over (its instance dropped via
the `vgeo_streamed` attribute) once its streamed cut has landed, in the same
tick, so the surface never shows a gap or two overlapping versions.
"""

import math
import os
import time

import bpy
import numpy as np
from mathutils import Vector

from . import native, stream

NODE_GROUP = "VGEO Instances"
LEVEL_ATTR = "vgeo_level"
ROT_ATTR = "vgeo_rot"
SCALE_ATTR = "vgeo_scale"
STREAMED_ATTR = "vgeo_streamed"   # bool per point: shown by a streamed copy, not an instance
STREAM_MIN_TRIS = 20_000          # a copy whose level mesh is smaller than this is cheap enough as it is
SLOT_KEEP_BONUS = 1.5             # a copy keeps its streamed cut until another costs this much more

_levels_cache = {}   # instancer uid -> (errors ndarray, tris ndarray)


def instancers(scene=None):
    objs = scene.objects if scene is not None else bpy.data.objects
    return [o for o in objs if o.type == 'MESH' and o.vgeo_inst.uid and o.vgeo_inst.levels is not None]


def ensure_node_group():
    ng = bpy.data.node_groups.get(NODE_GROUP)
    if ng and ng.bl_idname == "GeometryNodeTree" and "Levels" in ng.interface.items_tree:
        _ensure_streamed_selection(ng)
        _ensure_culled_selection(ng)
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
    _ensure_culled_selection(ng)
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


def _ensure_culled_selection(ng):
    """Culled placements (level -1) are left out too: Selection = not vgeo_streamed and vgeo_level >= 0.
    Added in place, so node groups from older files keep working."""
    if ng.nodes.get("VGEO shown") is not None:
        return
    iop = next((x for x in ng.nodes if x.bl_idname == "GeometryNodeInstanceOnPoints"), None)
    if iop is None:
        return
    sel = iop.inputs["Selection"]
    prev = sel.links[0].from_socket if sel.is_linked else None
    lvl = next((x for x in ng.nodes if x.bl_idname == "GeometryNodeInputNamedAttribute"
                and x.inputs["Name"].default_value == LEVEL_ATTR), None)
    if lvl is None:
        lvl = ng.nodes.new("GeometryNodeInputNamedAttribute")
        lvl.data_type = 'INT'
        lvl.inputs["Name"].default_value = LEVEL_ATTR
    cmp = ng.nodes.new("FunctionNodeCompare")
    cmp.name = cmp.label = "VGEO not culled"
    cmp.data_type = 'INT'
    cmp.operation = 'GREATER_EQUAL'
    a_in, b_in = [x for x in cmp.inputs if x.type == 'INT'][:2]
    b_in.default_value = 0
    ng.links.new(lvl.outputs["Attribute"], a_in)
    both = ng.nodes.new("FunctionNodeBooleanMath")
    both.name = both.label = "VGEO shown"
    both.operation = 'AND'
    if prev is not None:
        ng.links.new(prev, both.inputs[0])
    else:
        both.inputs[0].default_value = True
    ng.links.new(cmp.outputs[0], both.inputs[1])
    ng.links.new(both.outputs[0], sel)


def _socket_id(ng, name):
    return next(i.identifier for i in ng.interface.items_tree
                if getattr(i, "in_out", None) == "INPUT" and i.name == name)


def build_levels(inst, source):
    """Build (or rebuild) the shared level meshes for an instancer from a VGEO proxy."""
    path = stream.asset_path(source)
    asset = native.Asset(path)   # own handle: never disturbs the proxy's live selection
    try:
        errors = asset.level_errors()
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
        tris = []
        for depth in range(len(errors)):
            name = f"vgeo.{uid}.L{depth:02d}"
            me = bpy.data.meshes.new(name)
            stream._sync_materials(me, materials)
            data = asset.level_mesh_data(depth)
            stream.fill_mesh(me, data, materials, False)
            ob = bpy.data.objects.new(name, me)
            col.objects.link(ob)
            tris.append(data["tri_count"] if data else 0)
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


def _attach(inst):
    ng = ensure_node_group()
    mod = next((m for m in inst.modifiers if m.type == 'NODES' and m.node_group == ng), None)
    if mod is None:
        mod = inst.modifiers.new("VGEO Instances", 'NODES')
        mod.node_group = ng
    mod[_socket_id(ng, "Levels")] = inst.vgeo_inst.levels


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


def choose_levels(inst, views, pixel_error, mode=None, offscreen_scale=None):
    """Per placement, the coarsest level that keeps the error under the threshold in every view
    (-1: the copy is not drawn)."""
    return _choose(inst, views, pixel_error, mode, offscreen_scale)[0]


def _scene_of(inst):
    return inst.users_scene[0] if inst.users_scene else bpy.context.scene


_lights_cache = None   # scene name -> shadow_lights(scene), only while a tick or render update runs


class tick_scope:
    """Cache the scene's shadow-casting lights for the duration of one live tick or render update
    (every instancer's update reads them; scanning scene.objects per instancer is wasted)."""

    def __enter__(self):
        global _lights_cache
        self.outer = _lights_cache is not None
        if not self.outer:
            _lights_cache = {}
        return self

    def __exit__(self, *exc):
        global _lights_cache
        if not self.outer:
            _lights_cache = None
        return False


def _light_range(light, scene):
    """Distance beyond which a point or spot light no longer lights anything: its custom distance,
    else EEVEE's influence radius (where its power falls below the scene's light threshold)."""
    if getattr(light, "use_custom_distance", False):
        return float(light.cutoff_distance)
    threshold = 0.01
    ee = getattr(scene, "eevee", None)
    if ee is not None:
        threshold = float(getattr(ee, "light_threshold", threshold))
    power = max(light.color) * abs(light.energy) / 100.0 * max(1.0, getattr(light, "specular_factor", 1.0))
    return math.sqrt(max(1e-16, power / max(1e-16, threshold)))


def shadow_lights(scene):
    """(suns, locals): world-space unit directions the light of each shadow-casting sun travels in,
    and (position, range, axis, half_angle) of each shadow-casting point or spot light
    (axis None and half_angle pi for point lights)."""
    key = scene.name if scene is not None else ""
    if _lights_cache is not None and key in _lights_cache:
        return _lights_cache[key]
    suns, local = [], []
    for ob in scene.objects if scene is not None else ():
        if ob.type != 'LIGHT' or ob.hide_render or not getattr(ob.data, "use_shadow", True):
            continue
        la = ob.data
        if la.type == 'SUN':
            d = np.array(ob.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0)), dtype=np.float64)
            n = np.linalg.norm(d)
            if n > 0:
                suns.append(d / n)
        elif la.type in ('POINT', 'SPOT'):
            pos = np.array(ob.matrix_world.translation, dtype=np.float64)
            rng = _light_range(la, scene) + float(getattr(la, "shadow_soft_size", 0.0))
            axis, half = None, math.pi
            if la.type == 'SPOT':
                a = np.array(ob.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0)), dtype=np.float64)
                n = np.linalg.norm(a)
                if n > 0:
                    axis, half = a / n, min(math.pi, float(la.spot_size) * 0.5)
            local.append((pos, rng, axis, half))
    out = (suns, local)
    if _lights_cache is not None:
        _lights_cache[key] = out
    return out


def sun_directions(scene):
    """World-space directions the light of each shadow-casting sun lamp travels in (unit vectors)."""
    return shadow_lights(scene)[0]


def _lit_by_local(world, radius, local, planes):
    """Copies a shadow-casting point or spot light reaches, for lights whose range reaches the view:
    such a copy can throw a shadow onto what the view sees (conservative: no occlusion test)."""
    hit = np.zeros(len(world), bool)
    for pos, rng, axis, half in local:
        # the light's sphere of influence must touch the view, or nothing it lights is seen
        if np.any(planes[:, :3] @ pos + planes[:, 3] < -rng):
            continue
        v = world - pos
        dist = np.linalg.norm(v, axis=1)
        near = dist - radius <= rng
        if axis is not None and half < math.pi * 0.5 + 1e-6:
            inside = dist <= radius
            cosang = np.clip((v @ axis) / np.maximum(dist, 1e-12), -1.0, 1.0)
            ang = np.arccos(cosang) - np.arcsin(np.clip(radius / np.maximum(dist, 1e-12), 0.0, 1.0))
            near &= inside | (ang <= half)
        hit |= near
    return hit


def _shadow_reaches(dist, planes, radius, suns, reach):
    """Copies outside the view whose shadow can land in it: the copy's bounding sphere swept along
    each sun's light by `reach` must not lie fully outside any one frustum plane (conservative).
    dist: (n, 6) signed distances of the copy centres to the planes."""
    hit = np.zeros(len(dist), bool)
    for d in suns:
        nd = planes[:, :3] @ d                         # how fast each plane distance changes along d
        if reach is None:   # the shadow of a copy about 2r tall, cast onto ground at its base
            r = 2.0 * radius / max(-d[2], 0.1)
        else:
            r = np.full(len(dist), reach)
        swept = dist + r[:, None] * np.maximum(nd, 0.0)[None, :]
        hit |= np.all(swept >= -radius[:, None], axis=1)
    return hit


def _choose(inst, views, pixel_error, mode=None, offscreen_scale=None):
    """(levels, allowed error in asset units) per placement. Level -1: not drawn (culled)."""
    vi = inst.vgeo_inst
    mode = mode or getattr(vi, "offscreen", "COARSEN")
    offscreen_scale = offscreen_scale or getattr(vi, "offscreen_scale", 8.0)
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
    culling = mode in ("CULL", "SHADOWS")
    keep = np.zeros(n, bool) if culling else np.ones(n, bool)
    suns, local = shadow_lights(_scene_of(inst)) if mode == "SHADOWS" else ([], [])
    reach = getattr(vi, "shadow_distance", 0.0) or None
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
            dist = world @ planes[:, :3].T + planes[:, 3]
            inside = np.all(dist >= -radius[:, None], axis=1)
            b = np.where(inside, b, b * offscreen_scale)
            if culling:
                keep |= inside
                if suns:
                    keep |= _shadow_reaches(dist, planes, radius, suns, reach)
                if local:
                    keep |= _lit_by_local(world, radius, local, planes)
        budget = np.minimum(budget, b / s)   # error allowed in asset units
    levels = np.searchsorted(errors, budget, side="right") - 1
    levels = np.clip(levels, int(vi.min_level), len(errors) - 1).astype(np.int32)
    levels[~keep] = -1
    budget[~keep] = np.inf
    return levels, budget


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
    if (levels < 0).any():
        ensure_node_group()   # older node groups lack the selection that leaves culled copies out
    a.data.foreach_set("value", levels)
    me.update()
    return True


def triangles_for(inst, levels, streamed=None):
    _errors, tris = _level_tables(inst)
    if not len(levels):
        return 0
    shown = levels >= 0
    if streamed is not None and streamed.any():
        shown &= ~streamed
    return int(tris[levels[shown]].sum())


def update(inst, views, pixel_error, mode=None, offscreen_scale=None, render=False, paced=False):
    """Choose levels and streamed copies for these views. render=True: the streamed copies get
    their cut now and take over at once (final renders are synchronous). mode and offscreen_scale
    default to the instancer's Off-screen settings."""
    mode = mode or getattr(inst.vgeo_inst, "offscreen", "COARSEN")
    offscreen_scale = offscreen_scale or getattr(inst.vgeo_inst, "offscreen_scale", 8.0)
    if getattr(inst.vgeo_inst, "use_atoms", False) and inst.vgeo_inst.atoms is not None and native.has_atoms():
        return update_atoms(inst, views, pixel_error, mode, offscreen_scale, paced and not render)
    levels, budget = _choose(inst, views, pixel_error, mode, offscreen_scale)
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


def _held_slots(inst, n):
    """Placements that have a streamed copy assigned (landed or not)."""
    held = np.zeros(n, bool)
    for ob in slots(inst):
        if 0 <= ob.vgeo.slot_index < n:
            held[ob.vgeo.slot_index] = True
    return held


def _wanted_slots(inst, levels, budget):
    """Placements that should be streamed: the ones whose whole-copy level costs the most triangles.

    A whole-copy level must be fine enough for the copy's nearest point, so a big copy close to the
    camera draws its far side at that detail too; a streamed cut is fine only where it needs to be.
    The heaviest level meshes gain the most, whatever their level. Copies that already hold a slot
    keep it unless another costs clearly more, so slots do not flip back and forth as the view moves."""
    k = int(inst.vgeo_inst.stream_slots)
    _errors, tris = _level_tables(inst)
    if k <= 0 or not len(levels) or int(inst.vgeo_inst.min_level) > 0:
        return []
    cost = np.where(levels >= 0, tris[np.clip(levels, 0, len(tris) - 1)], 0).astype(np.float64)
    cand = np.nonzero(cost >= STREAM_MIN_TRIS)[0]
    if not len(cand):
        return []
    score = cost[cand]
    score[_held_slots(inst, len(levels))[cand]] *= SLOT_KEEP_BONUS
    order = np.lexsort((budget[cand], -score))   # costliest first; the nearer copy wins a tie
    return [int(i) for i in cand[order][:k]]


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


def copy_transforms(inst):
    """World matrices of every placement as (n, 3, 4) float32 rows, and their uniform scales (n,)."""
    co, scale = _points(inst)
    n = len(co)
    rot = np.zeros((n, 3), np.float32)
    a = inst.data.attributes.get(ROT_ATTR)
    if a is not None and a.domain == 'POINT' and a.data_type == 'FLOAT_VECTOR':
        a.data.foreach_get("vector", rot.ravel())
    x, y, z = (rot[:, k].astype(np.float64) for k in range(3))
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    # Euler XYZ (Blender): R = Rz @ Ry @ Rx
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = cz * cy
    R[:, 0, 1] = cz * sy * sx - sz * cx
    R[:, 0, 2] = cz * sy * cx + sz * sx
    R[:, 1, 0] = sz * cy
    R[:, 1, 1] = sz * sy * sx + cz * cx
    R[:, 1, 2] = sz * sy * cx - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cy * cx
    local = np.zeros((n, 4, 4))
    local[:, :3, :3] = R * scale[:, None, None]
    local[:, :3, 3] = co
    local[:, 3, 3] = 1.0
    mw = np.array(inst.matrix_world, dtype=np.float64)
    world = mw[None] @ local
    s = scale.astype(np.float64) * max(abs(v) for v in inst.matrix_world.to_scale())
    return np.ascontiguousarray(world[:, :3, :], dtype=np.float32), s.astype(np.float32)


def native_lights(scene, reach=0.0):
    """Shadow-casting lamps of the scene as native.ShadowLight structs (for vgeo_select_instances)."""
    suns, local = shadow_lights(scene)
    out = []
    for d in suns:
        L = native.ShadowLight()
        L.kind = 0
        L.dir[:] = [float(v) for v in d]
        L.reach = float(reach or 0.0)
        out.append(L)
    for pos, rng, axis, half in local:
        L = native.ShadowLight()
        L.kind = 1
        L.pos[:] = [float(v) for v in pos]
        L.range = float(rng)
        if axis is not None:
            L.axis[:] = [float(v) for v in axis]
        L.half_angle = float(half)
        out.append(L)
    return out


def world_views(views, pixel_error):
    """The live loop's views as world-space native views (vgeo_select_instances)."""
    out = []
    for view, window, height, persp, clip_start in views:
        planes = stream._frustum_planes(window @ view)
        if persp:
            out.append(native.make_view(view.inverted().translation, window[1][1], max(clip_start, 1e-6),
                                        pixel_error / max(1, height), planes=planes, mode="FULL"))
        else:
            h = 2.0 / window[1][1] if window[1][1] else 1.0
            out.append(native.make_view(view.inverted().translation, window[1][1], max(clip_start, 1e-6),
                                        pixel_error / max(1, height), ortho=True, ortho_height=h,
                                        planes=planes, mode="FULL"))
    return out


OCCLUSION_SIZE = tuple(int(x) for x in os.environ.get("VGEO_OCCLUSION_SIZE", "256x144").split("x"))
OCCLUDER_COPIES = int(os.environ.get("VGEO_OCCLUDER_COPIES", 128))   # nearest copies in view drawn
OCCLUDER_TRIANGLES = int(os.environ.get("VGEO_OCCLUDER_TRIANGLES", 200_000))

_occlusion = None          # native.Occlusion, reused
_occlusion_current = None  # built for this tick's view by prepare_occlusion, used by select_atoms


def build_occlusion(insts, views):
    """Draw the nearest copies of these instancers (coarse levels) into the occlusion buffer for a
    single perspective view. Returns the native.Occlusion, or None when it does not apply."""
    global _occlusion
    if len(views) != 1 or not native.has_occlusion():
        return None
    view, window, height, persp, clip_start = views[0]
    if not persp:
        return None
    if _occlusion is None:
        _occlusion = native.Occlusion(*OCCLUSION_SIZE)
        if "VGEO_OCCLUSION_CONFIG" in os.environ:   # experiments: "copy_tris,erode,push"
            t, e, p = (int(x) for x in os.environ["VGEO_OCCLUSION_CONFIG"].split(","))
            _occlusion.config(t, e, p)
    occ = _occlusion
    cam = np.array(view.inverted().translation, dtype=np.float64)
    occ.begin(np.array(window @ view, dtype=np.float32), cam, clip_start)
    planes = np.array(stream._frustum_planes(window @ view), dtype=np.float64)
    cands = []
    for inst in insts:
        if inst.vgeo_inst.source is None:
            continue
        tr, sc = copy_transforms(inst)
        c = tr[:, :, 3].astype(np.float64)
        r = float(inst.get("vgeo_radius", 1.0)) * sc
        inside = np.all(c @ planes[:, :3].T + planes[:, 3] >= -r[:, None], axis=1)
        d = np.linalg.norm(c - cam, axis=1) - r
        idx = np.nonzero(inside)[0]
        cands += [(float(d[i]), inst, int(i)) for i in idx[np.argsort(d[idx])][:OCCLUDER_COPIES]]
    cands.sort(key=lambda x: x[0])
    cands = cands[:OCCLUDER_COPIES]
    by_inst = {}
    for _d, inst, i in cands:
        by_inst.setdefault(inst.name, (inst, []))[1].append(i)
    for inst, idx in by_inst.values():
        tr, sc = copy_transforms(inst)
        occ.add(_atom_asset(inst), tr[idx], sc[idx], pixel_error=float(os.environ.get("VGEO_OCCLUDER_ERROR", 1.0)),
                max_triangles=OCCLUDER_TRIANGLES)
    occ.finish()
    return occ


def prepare_occlusion(insts, views):
    """Before a tick's or render's instancer updates: one occlusion buffer for all of them."""
    global _occlusion_current
    users = [i for i in insts if getattr(i.vgeo_inst, "use_atoms", False) and getattr(i.vgeo_inst, "use_occlusion", False)]
    _occlusion_current = build_occlusion(insts, views) if users else None
    return _occlusion_current


def select_atoms(inst, asset, views, pixel_error, mode=None, offscreen_scale=None, transforms=None,
                 occlusion=False):
    """A meshlet cut on every copy in one native call: ((m, 2) uint32 (copy, atom) pairs, stats).
    occlusion: a native.Occlusion, or True for this tick's (prepare_occlusion), False for none."""
    vi = inst.vgeo_inst
    mode = mode or getattr(vi, "offscreen", "COARSEN")
    offscreen_scale = offscreen_scale or getattr(vi, "offscreen_scale", 8.0)
    tr, sc = transforms if transforms is not None else copy_transforms(inst)
    lights = native_lights(_scene_of(inst), getattr(vi, "shadow_distance", 0.0)) if mode == "SHADOWS" else []
    if occlusion is True:
        occlusion = _occlusion_current if getattr(vi, "use_occlusion", False) else None
    blocks = len(inst.get("vgeo_atom_tris", ())) > int(inst.get("vgeo_atom_count", 1 << 30))
    return asset.select_instances(world_views(views, pixel_error), tr, sc, mode=mode,
                                  offscreen_scale=offscreen_scale, lights=lights, occlusion=occlusion or None,
                                  blocks=blocks)


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
    rt.invalidate()


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
            _release_slot(inst, ob, flags)
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
    """Streamed copies and atom clouds whose instancer was deleted: take them out of the scene."""
    live = {o.vgeo_inst.uid for o in bpy.data.objects if o.vgeo_inst.uid and o.users}
    for ob in [o for o in bpy.data.objects if o.vgeo.slot_of and o.users]:
        if ob.vgeo.slot_of not in live or ob.parent is None or ob.parent.vgeo_inst.uid != ob.vgeo.slot_of:
            for col in list(ob.users_collection):
                col.objects.unlink(ob)
    for ob in [o for o in bpy.data.objects if SLICE_PROP in o and o.users_collection]:
        p = ob.parent
        if p is None or not p.vgeo_inst.uid or not p.users_collection:
            for col in list(ob.users_collection):
                col.objects.unlink(ob)


def sync_cloud_visibility(inst, vl=None):
    """Atom clouds are separate objects: they hide and show with their instancer (viewport, render,
    and the view layer's hide), and stay hidden while Meshlet Instancing is off."""
    on = bool(getattr(inst.vgeo_inst, "use_atoms", False))
    try:
        hidden = inst.hide_get(view_layer=vl) if vl is not None else inst.hide_get()
    except RuntimeError:
        hidden = False
    hv, hr = inst.hide_viewport or not on, inst.hide_render or not on
    for ob in atom_clouds(inst).values():
        if ob.hide_viewport != hv:
            ob.hide_viewport = hv
        if ob.hide_render != hr:
            ob.hide_render = hr
        try:
            cur = ob.hide_get(view_layer=vl) if vl is not None else ob.hide_get()
            if cur != hidden:
                if vl is not None:
                    ob.hide_set(hidden, view_layer=vl)
                else:
                    ob.hide_set(hidden)
        except RuntimeError:
            pass


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
    # new instancers drop off-screen copies unless their shadow can reach the view; the property
    # default stays Coarsen so files saved before this keep rendering what they did
    inst.vgeo_inst.offscreen = 'SHADOWS'
    build_levels(inst, source)
    return inst


# ---------------------------------------------------------------- meshlet instancing (cut atoms)
#
# Every copy shows its own meshlet cut, made of instanced atoms: an atom is the set of clusters sharing
# a (group, refined) pair, so it is always in or out of the cut as a whole. Every update the native
# runtime selects (copy, atom) pairs for all copies in one call.
#
# Blender side (measured in the phase 2 spike): atoms must not be objects (every depsgraph update of a
# scene with thousands of objects pays for them), and a Geometry Nodes evaluation costs in proportion
# to the instance library it reads. So the atoms are split into slices of ATOM_SLICE. Each slice is one
# mesh (its atoms side by side, a face attribute holding the atom's index in the slice) on a library
# object whose Split to Instances makes one instance per atom, evaluated once. Each slice has a point
# cloud parented to the instancer (one point per pair: the copy's placement, the atom's index) whose
# Object Info + Instance on Points (Pick Instance) instances the library. An update rewrites only the
# clouds whose pairs changed; clouds keep spare points (atom -1, left out) so most writes keep the size.

ATOM_ATTR = "vgeo_atom"
ATOM_FACE_ATTR = "vgeo_atom_local"
ATOMS_NODE_GROUP = "VGEO Atoms"
LIBRARY_NODE_GROUP = "VGEO Atom Library"
ATOM_SLICE = 512
USE_BLOCKS = os.environ.get("VGEO_BLOCKS", "0") == "1"   # blocks: fewer instances, one more DAG of memory
SLICE_PROP = "vgeo_atom_slice"   # on a cloud object: its slice index
LIB_PROP = "vgeo_atom_library"   # on a library object: its slice index

ATOM_MOVING_INTERVAL = float(os.environ.get("VGEO_ATOM_INTERVAL", 0.2))   # seconds between viewport cut
                                                                         # updates while the view moves

# EEVEE spends ~2 us per instance per frame however small (phase 2 spike), and how many triangles
# that is worth depends on the scene (300 on the statue avenue, 2,700 on the synthetic field). So the
# viewport has an instance budget per instancer: every drawn copy starts as one whole level mesh, and
# copies switch to their atoms in order of triangles saved per extra instance while the budget lasts.
INSTANCE_BUDGET = int(os.environ.get("VGEO_INSTANCE_BUDGET", 8000))
# ...and a copy switches only if each extra instance saves at least this many triangles
MIN_GAIN = float(os.environ.get("VGEO_MIN_GAIN", 1000))

_atom_assets = {}   # instancer uid -> native.Asset (own handle, memory-mapped)
_atom_state = {}    # (instancer uid, slice) -> last pairs written to that cloud
_atom_clouds = {}   # instancer uid -> {slice: cloud object}, checked by _ensure_clouds
_atom_pace = {}     # instancer uid -> [time of last update, views key then, views key seen last tick]


def _named(ng, name, kind):
    a = ng.nodes.new("GeometryNodeInputNamedAttribute")
    a.data_type = kind
    a.inputs["Name"].default_value = name
    return a


def ensure_library_node_group():
    ng = bpy.data.node_groups.get(LIBRARY_NODE_GROUP)
    if ng and ng.bl_idname == "GeometryNodeTree":
        return ng
    ng = bpy.data.node_groups.new(LIBRARY_NODE_GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    split = ng.nodes.new("GeometryNodeSplitToInstances")
    split.domain = 'FACE'
    ng.links.new(gin.outputs["Geometry"], split.inputs["Geometry"])
    ng.links.new(_named(ng, ATOM_FACE_ATTR, 'INT').outputs["Attribute"], split.inputs["Group ID"])
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(split.outputs["Instances"], out.inputs["Geometry"])
    return ng


def ensure_atoms_node_group():
    ng = bpy.data.node_groups.get(ATOMS_NODE_GROUP)
    if ng and ng.bl_idname == "GeometryNodeTree" and "Library" in ng.interface.items_tree:
        return ng
    if ng is not None:
        ng.name = ATOMS_NODE_GROUP + " (old)"
    ng = bpy.data.node_groups.new(ATOMS_NODE_GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Library", in_out="INPUT", socket_type="NodeSocketObject")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, L = ng.nodes, ng.links
    gin = n.new("NodeGroupInput")
    info = n.new("GeometryNodeObjectInfo")
    info.transform_space = "ORIGINAL"
    L.new(gin.outputs["Library"], info.inputs["Object"])
    idx = _named(ng, ATOM_ATTR, 'INT')
    rot = _named(ng, ROT_ATTR, 'FLOAT_VECTOR')
    scl = _named(ng, SCALE_ATTR, 'FLOAT')
    e2r = n.new("FunctionNodeEulerToRotation")
    L.new(rot.outputs["Attribute"], e2r.inputs[0])
    used = n.new("FunctionNodeCompare")   # spare points carry atom -1
    used.data_type = 'INT'
    used.operation = 'GREATER_EQUAL'
    a_in, b_in = [x for x in used.inputs if x.type == 'INT'][:2]
    b_in.default_value = 0
    L.new(idx.outputs["Attribute"], a_in)
    iop = n.new("GeometryNodeInstanceOnPoints")
    L.new(gin.outputs["Geometry"], iop.inputs["Points"])
    L.new(used.outputs[0], iop.inputs["Selection"])
    L.new(info.outputs["Geometry"], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    L.new(idx.outputs["Attribute"], iop.inputs["Instance Index"])
    L.new(e2r.outputs[0], iop.inputs["Rotation"])
    L.new(scl.outputs["Attribute"], iop.inputs["Scale"])
    out = n.new("NodeGroupOutput")
    L.new(iop.outputs["Instances"], out.inputs["Geometry"])
    return ng


def _atom_asset(inst):
    uid = inst.vgeo_inst.uid
    a = _atom_assets.get(uid)
    if a is None:
        a = native.Asset(stream.asset_path(inst.vgeo_inst.source))
        _atom_assets[uid] = a
    return a


def close_atom_assets():
    for a in _atom_assets.values():
        a.close()
    _atom_assets.clear()
    _atom_state.clear()
    _atom_clouds.clear()
    _atom_pace.clear()


def atom_libraries(inst):
    """{slice: library object} of an instancer."""
    col = inst.vgeo_inst.atoms
    if col is None:
        return {}
    return {int(o[LIB_PROP]): o for o in col.objects if LIB_PROP in o}


def atom_clouds(inst):
    """{slice: cloud object} of an instancer."""
    return {int(o[SLICE_PROP]): o for o in inst.children if SLICE_PROP in o}


def _concat(parts):
    out = {k: [] for k in ("positions", "normals", "uvs", "corner_verts", "face_materials", "face_lod",
                           "edge_verts", "corner_edges")}
    v_off = e_off = 0
    for d in parts:
        out["positions"].append(d["positions"])
        out["normals"].append(d["normals"])
        if d["uvs"] is not None:
            out["uvs"].append(d["uvs"])
        out["corner_verts"].append(d["corner_verts"] + v_off)
        out["face_materials"].append(d["face_materials"])
        out["face_lod"].append(d["face_lod"])
        out["edge_verts"].append(d["edge_verts"] + v_off)
        out["corner_edges"].append(d["corner_edges"] + e_off)
        v_off += d["vertex_count"]
        e_off += d["edge_count"]
    res = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
    if len(out["uvs"]) != len(parts):
        res["uvs"] = None
    res["vertex_count"], res["edge_count"], res["tri_count"] = v_off, e_off, len(res["face_materials"])
    return res


def _remove_objects(obs):
    for ob in list(obs):
        me = ob.data
        bpy.data.objects.remove(ob)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)


def build_atoms(inst):
    """Build the instancer's atom libraries: one mesh per slice of ATOM_SLICE atoms, with the atom's
    index in the slice as a face attribute, on a library object (in a collection only Geometry Nodes
    reads) that splits it into one instance per atom."""
    vi = inst.vgeo_inst
    uid = vi.uid
    asset = _atom_asset(inst)
    atoms = asset.atoms()
    blocks = asset.atoms(blocks=True) if USE_BLOCKS else atoms[:0]
    units = len(atoms) + len(blocks)   # unit ids: atoms, then blocks
    col = vi.atoms
    if col is None:
        col = bpy.data.collections.new(f".vgeo atoms {uid}")
        vi.atoms = col
    for sub in list(col.children):   # an older layout: object per atom, collection per slice
        _remove_objects(sub.objects)
        bpy.data.collections.remove(sub)
    _remove_objects(col.objects)
    _remove_objects(atom_clouds(inst).values())
    materials = tuple(vi.source.data.materials)
    lib_ng = ensure_library_node_group()
    for k in range(-(-units // ATOM_SLICE)):
        lo, hi = k * ATOM_SLICE, min(units, (k + 1) * ATOM_SLICE)
        parts, local = [], []
        for a in range(lo, hi):
            d = asset.extract_atom(a)
            if d is None:   # every cluster corrupt (reported by the asset): later indices would shift
                continue
            parts.append(d)
            local.append(np.full(d["tri_count"], a - lo, np.int32))
        name = f"vgeo.{uid}.atoms{k:03d}"
        me = bpy.data.meshes.new(name)
        stream._sync_materials(me, materials)
        data = _concat(parts)
        stream.fill_mesh(me, data, materials, False)
        me.attributes.new(ATOM_FACE_ATTR, 'INT', 'FACE').data.foreach_set("value", np.concatenate(local))
        ob = bpy.data.objects.new(name, me)
        ob[LIB_PROP] = k
        mod = ob.modifiers.new("VGEO Atom Library", 'NODES')
        mod.node_group = lib_ng
        col.objects.link(ob)
    inst["vgeo_atom_tris"] = [int(t) for t in np.concatenate([atoms["tri_count"], blocks["tri_count"]])]
    inst["vgeo_atom_count"] = len(atoms)
    for key in [k for k in _atom_state if k[0] == uid]:
        del _atom_state[key]
    _atom_clouds.pop(uid, None)
    _atom_pace.pop(uid, None)
    return len(atoms)


def rebuild_atoms(inst):
    """Rebuild the atom libraries from the asset as it is now (a fresh native handle: the file may
    have been re-virtualized), keeping the clouds."""
    uid = inst.vgeo_inst.uid
    a = _atom_assets.pop(uid, None)
    if a is not None:
        a.close()
    n = build_atoms(inst)
    if inst.vgeo_inst.use_atoms:
        _ensure_clouds(inst)
        sync_cloud_visibility(inst)
    return n


def _ensure_clouds(inst):
    """One point cloud per atom slice, parented to the instancer, linked where it is."""
    vi = inst.vgeo_inst
    ng = ensure_atoms_node_group()
    have = atom_clouds(inst)
    out = {}
    for k, lib in sorted(atom_libraries(inst).items()):
        ob = have.get(k)
        if ob is None:
            name = f"{inst.name} Atoms {k:03d}"
            ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
            ob[SLICE_PROP] = k
            ob.parent = inst
            ob.hide_select = True
        mod = next((m for m in ob.modifiers if m.type == 'NODES'), None)
        if mod is None:
            mod = ob.modifiers.new("VGEO Atoms", 'NODES')
        if mod.node_group != ng:
            mod.node_group = ng
        if mod[_socket_id(ng, "Library")] != lib:
            mod[_socket_id(ng, "Library")] = lib
        for col in inst.users_collection:
            if ob.name not in col.objects:
                col.objects.link(ob)
        out[k] = ob
    vi.atom_points = out.get(0)
    return out


def set_use_atoms(inst, on):
    """Switch an instancer between whole-asset levels (with streamed copies) and meshlet instancing."""
    vi = inst.vgeo_inst
    if on:
        if not native.has_atoms():
            print("VGEO: this build of the native library has no cut atoms")
            return
        if vi.atoms is None or not atom_libraries(inst):
            build_atoms(inst)
        _ensure_clouds(inst)
        sync_cloud_visibility(inst)
        flags = _streamed_flags(inst)   # streamed copies are not needed: every copy is cut
        for ob in slots(inst):
            if ob.vgeo.slot_index >= 0:
                _release_slot(inst, ob, flags)
        _write_streamed(inst, flags)
    else:
        sync_cloud_visibility(inst)
        for key in [k for k in _atom_state if k[0] == vi.uid]:
            del _atom_state[key]
    _atom_clouds.pop(vi.uid, None)
    _atom_pace.pop(vi.uid, None)


def _cloud_attr(me, name, kind):
    a = me.attributes.get(name)
    if a is None or a.domain != 'POINT' or a.data_type != kind:
        if a is not None:
            me.attributes.remove(a)
        a = me.attributes.new(name, kind, 'POINT')
    return a


def _write_cloud(ob, copy, local, co, rot, scale):
    """Write one cloud's points. The point count only grows (by half again when it must); spare
    points get atom -1, which the node group leaves out, so most writes keep the mesh's size."""
    me = ob.data
    m = len(copy)
    cap = len(me.vertices)
    if m > cap or cap > 4 * max(m, 64):
        me.clear_geometry()
        cap = max(64, int(m * 1.5))
        me.vertices.add(cap)
    pos = np.zeros((cap, 3), np.float32)
    rt = np.zeros((cap, 3), np.float32)
    sc = np.ones(cap, np.float32)
    idx = np.full(cap, -1, np.int32)
    pos[:m] = co[copy]
    rt[:m] = rot[copy]
    sc[:m] = scale[copy]
    idx[:m] = local
    me.vertices.foreach_set("co", pos.ravel())
    _cloud_attr(me, ATOM_ATTR, 'INT').data.foreach_set("value", idx)
    _cloud_attr(me, ROT_ATTR, 'FLOAT_VECTOR').data.foreach_set("vector", rt.ravel())
    _cloud_attr(me, SCALE_ATTR, 'FLOAT').data.foreach_set("value", sc)
    me.update()


def _clouds_fast(inst):
    """The instancer's clouds, re-checked in full only when one was deleted or the set is unknown."""
    uid = inst.vgeo_inst.uid
    clouds = _atom_clouds.get(uid)
    if clouds is not None:
        try:
            if all(ob.parent == inst for ob in clouds.values()):
                return clouds
        except ReferenceError:
            pass
    clouds = _ensure_clouds(inst)
    _atom_clouds[uid] = clouds
    return clouds


def _write_atom_points(inst, pairs):
    """Route the (copy, atom) pairs to their slices' clouds; rewrite only the clouds that changed.
    Returns the number of clouds rewritten."""
    vi = inst.vgeo_inst
    clouds = _clouds_fast(inst)
    if not clouds:
        return 0
    sl = (pairs[:, 1] // ATOM_SLICE).astype(np.int64)
    order = np.argsort(sl, kind="stable")
    sp = pairs[order]
    bounds = np.searchsorted(sl[order], np.arange(max(clouds) + 2))
    data = None
    written = 0
    for k, ob in clouds.items():
        part = sp[bounds[k]:bounds[k + 1]]
        key = (vi.uid, k)
        prev = _atom_state.get(key)
        if prev is not None and prev.shape == part.shape and np.array_equal(prev, part):
            continue
        if data is None:
            co, scale = _points(inst)
            rot = np.zeros((len(co), 3), np.float32)
            a = inst.data.attributes.get(ROT_ATTR)
            if a is not None and a.domain == 'POINT' and a.data_type == 'FLOAT_VECTOR':
                a.data.foreach_get("vector", rot.ravel())
            data = (co, rot, scale)
        _write_cloud(ob, part[:, 0].astype(np.int64), (part[:, 1] % ATOM_SLICE).astype(np.int32), *data)
        _atom_state[key] = part.copy()
        written += 1
    return written


def _views_key(inst, views, pixel_error, mode, offscreen_scale):
    """What the cut depends on: the views, settings, placements, the instancer's transform, lamps."""
    co, scale = _points(inst)
    a = inst.data.attributes.get(ROT_ATTR)
    rot = np.zeros(len(co) * 3, np.float32)
    if a is not None and a.domain == 'POINT' and a.data_type == 'FLOAT_VECTOR':
        a.data.foreach_get("vector", rot)
    parts = [round(pixel_error, 4), mode, round(offscreen_scale or 0, 3),
             hash(co.tobytes()), hash(scale.tobytes()), hash(rot.tobytes()),
             tuple(round(x, 6) for row in inst.matrix_world for x in row)]
    if mode == "SHADOWS":
        suns, local = shadow_lights(_scene_of(inst))
        parts.append(tuple(tuple(np.round(d, 5)) for d in suns))
        parts.append(tuple((tuple(np.round(p, 4)), round(r, 4)) for p, r, _a, _h in local))
    for view, window, height, persp, clip_start in views:
        parts.append(tuple(round(x, 5) for row in view for x in row))
        parts.append(tuple(round(x, 5) for row in window for x in row))
        parts.append(height)
    return tuple(parts)


def update_atoms(inst, views, pixel_error, mode=None, offscreen_scale=None, paced=False):
    """Meshlet instancing update: one native selection for every copy, then rewrite changed clouds.

    paced (the live loop): a moving view gets a new cut at most every ATOM_MOVING_INTERVAL (each
    update costs EEVEE a re-sync of the changed clouds and their shadows); once the view stops, the
    cut for where it stopped lands on the next tick. Unpaced calls (final renders, scripts) cut now."""
    uid = inst.vgeo_inst.uid
    if paced:
        key = _views_key(inst, views, pixel_error, mode, offscreen_scale)
        pace = _atom_pace.get(uid)
        now = time.perf_counter()
        if pace is not None:
            if key == pace[1]:
                pace[2] = key
                return False   # this view's cut is already shown
            moving = key != pace[2]
            pace[2] = key
            if moving and now - pace[0] < ATOM_MOVING_INTERVAL:
                return False
        _atom_pace[uid] = [now, key, key]
    else:
        _atom_pace.pop(uid, None)
    asset = _atom_asset(inst)
    pairs, st = select_atoms(inst, asset, views, pixel_error, mode, offscreen_scale, occlusion=True)
    n = len(inst.data.vertices)
    levels = np.full(n, -1, np.int32)   # copies drawn as atoms show no level mesh
    tris = int(st.triangles)
    if paced and INSTANCE_BUDGET > 0 and len(pairs):
        # viewport over budget: some copies are drawn as one whole level mesh instead of their atoms
        # (final renders always draw atoms: fewer triangles, and renderers cope with the instances)
        atom_tris = np.asarray(inst.get("vgeo_atom_tris", []), np.int64)
        if len(atom_tris):
            lv, _budget = _choose(inst, views, pixel_error, mode, offscreen_scale)
            _errors, lvl_tris = _level_tables(inst)
            c = pairs[:, 0].astype(np.int64)
            a_tris = np.bincount(c, weights=atom_tris[pairs[:, 1]], minlength=n)
            a_count = np.bincount(c, minlength=n)
            l_tris = np.where(lv >= 0, lvl_tris[np.maximum(lv, 0)], 0)
            use_level = (lv >= 0) & (a_count > 1)          # candidates for the level fallback
            # instances if every candidate were a level mesh, and what switching each back to atoms
            # costs (extra instances) and saves (triangles)
            base = int(a_count[~use_level].sum() + use_level.sum())
            extra = (a_count - 1).astype(np.float64)
            gain = (l_tris - a_tris).astype(np.float64)
            cand = np.nonzero(use_level & (gain > 0) & (gain >= MIN_GAIN * extra))[0]
            order = cand[np.argsort(-gain[cand] / extra[cand], kind="stable")]
            fits = np.cumsum(extra[order]) <= max(0, INSTANCE_BUDGET - base)
            use_level[order[fits]] = False
            if use_level.any():
                levels[use_level] = lv[use_level]
                tris = int(a_tris[~use_level].sum() + l_tris[use_level].sum())
                pairs = pairs[~use_level[c]]
    changed = _write_atom_points(inst, pairs) > 0
    changed |= apply_levels(inst, levels)
    inst.vgeo_inst.shown_triangles = int(min(tris, 2**31 - 1))
    return changed
