"""Virtualize Instances: turn an existing scatter into VGEO instancers.

Any object whose evaluation yields instances of a mesh that has a VGEO proxy (a Geometry Nodes
Instance on Points with Object Info or Collection Info, a particle system rendering objects, a
collection instance, instancing on vertices or faces) can hand those instances to a VGEO
instancer: one instancer per instanced mesh, with the same world placements. The scatter's own
instances are then switched off without touching its other geometry:

- Geometry Nodes: a "VGEO Hide Instances" node group is appended to its modifier stack (it keeps
  every component except instances); removing it brings them back.
- Particle systems that render objects or collections: render type None (the old one is kept).
- Collection instances and instancing on vertices or faces: instance type None (kept likewise).

Restore Instances undoes all of it and deletes the instancers it made.
"""

import math

import bpy
import numpy as np

from . import instances, stream

HIDE_GROUP = "VGEO Hide Instances"
HIDE_MODIFIER = "VGEO Hide Instances"
MADE_PROP = "vgeo_made_from"          # on an instancer: the scatter object it was made from
PSYS_PROP = "vgeo_psys_render_types"  # on a scatter: {particle settings name: old render type}
TYPE_PROP = "vgeo_instance_type"      # on a scatter: its old instance_type


def proxy_for(ob):
    """The VGEO proxy showing this mesh (the object itself if it is one), or None."""
    if ob is None:
        return None
    if ob.vgeo.uid and not ob.vgeo.slot_of:
        return ob
    for p in bpy.data.objects:
        if p.vgeo.uid and not p.vgeo.slot_of and p.vgeo.source == ob:
            return p
    return None


def scatter_instances(scatter, depsgraph):
    """{instanced original object: [world matrices]} of the scatter's evaluated instances."""
    out = {}
    for inst in depsgraph.object_instances:
        if not inst.is_instance or inst.parent is None or inst.parent.original != scatter:
            continue
        ob = inst.object.original if inst.object is not None else None
        if ob is None or ob == scatter or ob.type != 'MESH':
            continue
        out.setdefault(ob, []).append(np.array(inst.matrix_world, dtype=np.float64))
    return out


def _decompose(mats):
    """Placements (n, 4, 4) -> positions, XYZ Euler rotations, uniform scales, and the largest
    relative spread of the scale's axes (non-uniform scale is approximated by the mean)."""
    m = np.asarray(mats, dtype=np.float64)
    pos = m[:, :3, 3]
    lin = m[:, :3, :3]
    axes = np.linalg.norm(lin, axis=1)                  # column lengths: scale per local axis
    scale = axes.mean(axis=1)
    spread = float(((axes.max(axis=1) - axes.min(axis=1)) / np.maximum(scale, 1e-12)).max()) if len(m) else 0.0
    rot = lin / np.maximum(axes[:, None, :], 1e-12)
    # mirrored placements (negative determinant) cannot be expressed: flip one axis
    neg = np.linalg.det(rot) < 0
    rot[neg, :, 0] *= -1
    return pos, instances._euler_from_matrices(rot), scale, spread


def _hide_group():
    ng = bpy.data.node_groups.get(HIDE_GROUP)
    if ng is not None and ng.bl_idname == "GeometryNodeTree":
        return ng
    ng = bpy.data.node_groups.new(HIDE_GROUP, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    sep = ng.nodes.new("GeometryNodeSeparateComponents")
    join = ng.nodes.new("GeometryNodeJoinGeometry")
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(gin.outputs[0], sep.inputs[0])
    for name in ("Mesh", "Curve", "Point Cloud", "Volume", "Grease Pencil"):
        if name in sep.outputs:
            ng.links.new(sep.outputs[name], join.inputs[0])
    ng.links.new(join.outputs[0], out.inputs[0])
    return ng


def hide_scatter_instances(scatter):
    """Switch off the scatter's own instances, remembering how to bring them back."""
    if any(m.type == 'NODES' for m in scatter.modifiers) and scatter.modifiers.get(HIDE_MODIFIER) is None:
        mod = scatter.modifiers.new(HIDE_MODIFIER, 'NODES')
        mod.node_group = _hide_group()
    old = {}
    for ps in getattr(scatter, "particle_systems", ()):
        st = ps.settings
        if st.render_type in ('OBJECT', 'COLLECTION'):
            old[st.name] = st.render_type
            st.render_type = 'NONE'
    if old:
        scatter[PSYS_PROP] = old
    if scatter.instance_type in ('COLLECTION', 'VERTS', 'FACES'):
        scatter[TYPE_PROP] = scatter.instance_type
        scatter.instance_type = 'NONE'


def restore_scatter_instances(scatter):
    mod = scatter.modifiers.get(HIDE_MODIFIER)
    if mod is not None:
        scatter.modifiers.remove(mod)
    old = scatter.get(PSYS_PROP)
    if old:
        for ps in getattr(scatter, "particle_systems", ()):
            if ps.settings.name in old:
                ps.settings.render_type = old[ps.settings.name]
        del scatter[PSYS_PROP]
    if TYPE_PROP in scatter:
        scatter.instance_type = scatter[TYPE_PROP]
        del scatter[TYPE_PROP]


def make_instancer(context, proxy, mats, name, collections):
    """A VGEO instancer (at the world origin) with one placement per world matrix."""
    pos, rot, scale, spread = _decompose(mats)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(pos))
    me.vertices.foreach_set("co", pos.astype(np.float32).ravel())
    me.attributes.new(instances.ROT_ATTR, 'FLOAT_VECTOR', 'POINT').data.foreach_set(
        "vector", rot.astype(np.float32).ravel())
    me.attributes.new(instances.SCALE_ATTR, 'FLOAT', 'POINT').data.foreach_set("value", scale.astype(np.float32))
    me.attributes.new(instances.LEVEL_ATTR, 'INT', 'POINT')
    me.update()
    inst = bpy.data.objects.new(name, me)
    for col in collections:
        col.objects.link(inst)
    vi = inst.vgeo_inst
    vi.uid = stream.new_uid()
    vi.source = proxy
    vi.pixel_error = proxy.vgeo.pixel_error
    vi.render_pixel_error = proxy.vgeo.render_pixel_error
    vi.offscreen = 'SHADOWS'
    try:
        instances.build_levels(inst, proxy)
    except RuntimeError:
        col = vi.levels
        bpy.data.objects.remove(inst)
        bpy.data.meshes.remove(me)
        if col is not None and not col.objects:
            bpy.data.collections.remove(col)
        raise
    return inst, spread


def virtualize_instances(context, scatter, min_copies=2):
    """Make instancers for the scatter's instances of virtualized meshes. Returns (instancers,
    {mesh name: copies} it could not take because the mesh has no VGEO proxy, largest scale spread)."""
    dg = context.evaluated_depsgraph_get()
    groups = scatter_instances(scatter, dg)
    made, missing, spread = [], {}, 0.0
    cols = list(scatter.users_collection) or [context.scene.collection]
    for ob, mats in groups.items():
        if len(mats) < min_copies:
            continue
        proxy = proxy_for(ob)
        if proxy is None:
            missing[ob.name] = len(mats)
            continue
        try:
            inst, s = make_instancer(context, proxy, mats, f"{scatter.name} {ob.name} VGEO Instances", cols)
        except RuntimeError as e:   # the proxy's asset file cannot be opened
            missing[ob.name] = f"{len(mats)} ({e})"
            continue
        inst[MADE_PROP] = scatter.name
        made.append(inst)
        spread = max(spread, s)
    if made:
        hide_scatter_instances(scatter)
    return made, missing, spread


def restore_instances(scatter):
    """Bring back the scatter's own instances and delete the instancers made from it."""
    restore_scatter_instances(scatter)
    for ob in [o for o in bpy.data.objects if o.get(MADE_PROP) == scatter.name and o.vgeo_inst.uid]:
        for child in list(ob.children):
            bpy.data.objects.remove(child)
        me = ob.data
        bpy.data.objects.remove(ob)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    instances.unlink_orphan_slots()
