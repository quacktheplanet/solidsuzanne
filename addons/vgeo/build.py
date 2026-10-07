"""Turn a Blender mesh object into a virtualized (.vgeo) proxy."""

import os
import threading

import bpy
import numpy as np

from . import native, stream


def mesh_arrays(obj, depsgraph):
    """Triangle arrays of the evaluated mesh (modifiers applied), in object space.

    Returns a dict for native.build. Smooth meshes without UV seams use the
    indexed layout (one row per vertex); anything with hard edges, split
    normals or UV seams falls back to one row per triangle corner.
    """
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        T = len(me.loop_triangles)
        if T == 0:
            raise ValueError(f"'{obj.name}' has no faces")
        V, C = len(me.vertices), len(me.loops)
        tl = np.empty(T * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tl)
        tp = np.empty(T, np.int32)
        me.loop_triangles.foreach_get("polygon_index", tp)
        cv = np.empty(C, np.int32)
        me.loops.foreach_get("vertex_index", cv)
        co = np.empty(V * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        cn = np.empty(C * 3, np.float32)
        me.corner_normals.foreach_get("vector", cn)
        cn = cn.reshape(-1, 3)
        uv = None
        layer = me.uv_layers.active
        if layer is not None:
            uv = np.empty(C * 2, np.float32)
            layer.data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)
        extras, desc = _extra_channels(me, cv, uv_name=layer.name if layer is not None else "UVMap")
        pm = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("material_index", pm)
        mat_list = evaluated_materials(obj, ev, me)
        slots = max(1, len(mat_list))
        materials = np.clip(pm[tp], 0, slots - 1).astype(np.uint16)

        # per-vertex attributes taken from any corner of each vertex
        first = np.full(V, -1, np.int64)
        first[cv[::-1]] = np.arange(C - 1, -1, -1)
        used = first >= 0
        first = np.where(used, first, 0)
        vn = cn[first]
        smooth = np.allclose(cn, vn[cv], atol=1e-5)
        seamless = uv is None or np.allclose(uv, uv[first][cv], atol=1e-6)
        if extras is not None:
            seamless = seamless and np.allclose(extras, extras[first][cv], atol=1e-6)
        if smooth and seamless:
            return {"positions": co, "normals": vn, "uvs": None if uv is None else uv[first],
                    "materials": materials, "indices": cv[tl].astype(np.uint32), "material_list": mat_list,
                    "extras": None if extras is None else extras[first], "extra_desc": desc}
        return {"positions": co[cv[tl]], "normals": cn[tl], "uvs": None if uv is None else uv[tl],
                "materials": materials, "indices": None, "material_list": mat_list,
                "extras": None if extras is None else extras[tl], "extra_desc": desc}
    finally:
        ev.to_mesh_clear()


# what a material can read besides the active UV map, carried into every cut (library version 3)
_EXTRA_KINDS = {"FLOAT2": ("vector", 2), "FLOAT_COLOR": ("color", 4), "BYTE_COLOR": ("color", 4),
                "FLOAT": ("value", 1), "FLOAT_VECTOR": ("vector", 3)}
_SKIP_ATTRS = {"position", "material_index", "sharp_face", "sharp_edge", "custom_normal", "crease_vert",
               "crease_edge", "bevel_weight_vert", "bevel_weight_edge"}
MAX_EXTRA_FLOATS = 64


def _extra_channels(me, cv, uv_name="UVMap"):
    """(per-corner float array or None, JSON description) of the mesh's other UV maps and its colour and
    float attributes on points or corners, plus the active UV's name and the texture space (so Generated
    coordinates match on every streamed chunk). Points are spread to their corners."""
    import json
    C = len(me.loops)
    cols, channels, off = [], [], 0
    active = me.uv_layers.active
    uv_names = {l.name for l in me.uv_layers}
    render_uv = next((l.name for l in me.uv_layers if l.active_render), uv_name)
    for l in me.uv_layers:
        if l == active:
            continue
        a = np.empty(C * 2, np.float32)
        l.data.foreach_get("uv", a)
        cols.append(a.reshape(-1, 2))
        channels.append({"name": l.name, "type": "FLOAT2", "size": 2, "offset": off, "uv": True})
        off += 2
    ac = getattr(me.color_attributes, "active_color_name", "") or ""
    rc_index = getattr(me.color_attributes, "render_color_index", -1)
    render_col = me.color_attributes[rc_index].name if 0 <= rc_index < len(me.color_attributes) else ""
    color_names = {c.name for c in me.color_attributes}
    for at in me.attributes:
        if (at.name in uv_names or at.name in _SKIP_ATTRS or at.name.startswith(".")
                or getattr(at, "is_internal", False) or at.domain not in ("POINT", "CORNER")
                or at.data_type not in _EXTRA_KINDS):
            continue
        prop, n = _EXTRA_KINDS[at.data_type]
        if off + n > MAX_EXTRA_FLOATS:
            print(f"VGEO: '{at.name}' left out (more than {MAX_EXTRA_FLOATS} floats of extra data)")
            continue
        count = len(me.vertices) if at.domain == "POINT" else C
        a = np.empty(count * n, np.float32)
        at.data.foreach_get(prop, a)
        a = a.reshape(-1, n)
        if at.domain == "POINT":
            a = a[cv]
        cols.append(a)
        is_color = at.name in color_names
        channels.append({"name": at.name, "type": "FLOAT_COLOR" if is_color else at.data_type, "size": n,
                         "offset": off, "color": is_color})
        off += n
    try:
        texspace = [list(me.texspace_location), list(me.texspace_size)]
    except AttributeError:
        texspace = None
    desc = {"uv_name": uv_name, "render_uv": render_uv, "channels": channels, "active_color": ac,
            "render_color": render_col, "texspace": texspace}
    extras = np.ascontiguousarray(np.hstack(cols), dtype=np.float32) if cols else None
    return extras, json.dumps(desc)


def evaluated_materials(obj, ev, me):
    """The materials Blender actually renders the evaluated mesh with.

    Geometry Nodes can replace the mesh and its materials (Set Material), so
    the original object's slots are not authoritative. Object-linked slots
    still override the mesh, as in Blender.
    """
    mesh_mats = list(me.materials)
    out = []
    for i in range(max(len(mesh_mats), len(ev.material_slots))):
        slot = ev.material_slots[i] if i < len(ev.material_slots) else None
        if slot is not None and slot.link == 'OBJECT' and slot.material is not None:
            out.append(slot.material)
        elif i < len(mesh_mats):
            out.append(mesh_mats[i])
        else:
            out.append(slot.material if slot is not None else None)
    return out


def material_params(materials):
    """Per material (base color rgb, roughness) for viewers outside Blender.

    Uses the Principled BSDF's unlinked inputs; when an input is driven by
    nodes, falls back to the material's viewport display color/roughness.
    """
    out = []
    for m in materials:
        color, rough = (0.7, 0.7, 0.7), 0.6
        if m is not None:
            color, rough = tuple(m.diffuse_color[:3]), float(m.roughness)
            nt = m.node_tree if m.use_nodes else None
            bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None) if nt else None
            if bsdf is not None:
                bc, rg = bsdf.inputs.get("Base Color"), bsdf.inputs.get("Roughness")
                if bc is not None and not bc.is_linked:
                    color = tuple(bc.default_value[:3])
                if rg is not None and not rg.is_linked:
                    rough = float(rg.default_value)
        out.append((*color, rough))
    return out


def default_path(obj, uid):
    name = f"{bpy.path.clean_name(obj.name)}_{uid[:6]}.vgeo"
    if bpy.data.filepath:
        folder = bpy.path.abspath("//vgeo")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, name), "//vgeo/" + name
    folder = bpy.utils.user_resource('DATAFILES', path="vgeo", create=True)
    full = os.path.join(folder, name)
    return full, full


class Job:
    """Runs native.build on a worker thread; the GIL is released inside the DLL."""

    def __init__(self, arrays, path, material_names, max_triangles, target_chunks=0, material_params=None):
        self.material_params = material_params
        self.arrays = arrays
        self.path = path
        self.material_names = material_names
        self.max_triangles = max_triangles
        self.target_chunks = target_chunks
        self.stage = 0
        self.fraction = 0.0
        self.cancel = False
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _progress(self, stage, fraction):
        self.stage, self.fraction = stage, fraction
        return self.cancel

    def _run(self):
        try:
            a = self.arrays
            self.result = native.build(self.path, a["positions"], a["normals"], a["uvs"], a["materials"],
                                       self.material_names, max_triangles=self.max_triangles,
                                       target_chunks=self.target_chunks, progress=self._progress,
                                       indices=a["indices"], material_params=self.material_params,
                                       extras=a.get("extras"), extra_desc=a.get("extra_desc", ""))
        except BaseException as e:
            self.error = e
        finally:
            self.arrays = None

    @property
    def overall(self):
        # welding is fast, the DAG dominates
        return {0: 0.05 * self.fraction, 1: 0.05 + 0.9 * self.fraction, 2: 0.95 + 0.05 * self.fraction}.get(self.stage, 0.0)


def create_proxy(context, src, path_setting, uid, build_stats, remove_source=False, materials=None):
    """Replace src with a streaming proxy that has the same transform, parent, collections and materials."""
    me = bpy.data.meshes.new(f"{src.data.name} VGEO")
    for m in (materials if materials is not None else [s.material for s in src.material_slots]):
        me.materials.append(m)
    proxy = bpy.data.objects.new(f"{src.name} VGEO", me)
    for col in src.users_collection:
        col.objects.link(proxy)
    proxy.parent = src.parent
    proxy.parent_type = src.parent_type
    proxy.matrix_parent_inverse = src.matrix_parent_inverse.copy()
    proxy.matrix_basis = src.matrix_basis.copy()

    v = proxy.vgeo
    v.uid = uid
    v.path = path_setting
    v.source_triangles = int(build_stats.get("source_triangles", 0))
    v.file_bytes = int(build_stats.get("file_bytes", 0))
    v.collection = bpy.data.collections.new(f".vgeo {uid}")
    stream.link_chunks(proxy)

    # a first cut so there is something to see before the live loop runs
    rt = stream.runtime_for(proxy)
    if rt.asset is not None:
        views = stream.viewport_views() if context.window_manager.windows else []
        if views:
            stream.apply_cut(proxy, views, v.pixel_error, v.offscreen, v.offscreen_scale)
        else:
            stream.apply_level(proxy, -1)

    if remove_source:
        src_mesh = src.data
        bpy.data.objects.remove(src)
        if src_mesh.users == 0:
            bpy.data.meshes.remove(src_mesh)
    else:
        v.source = src
        src.hide_set(True)
        src.hide_render = True

    for o in context.view_layer.objects:
        o.select_set(False)
    proxy.select_set(True)
    context.view_layer.objects.active = proxy
    # handlers modify scene data during F12 renders; that needs the interface locked
    context.scene.render.use_lock_interface = True
    return proxy
