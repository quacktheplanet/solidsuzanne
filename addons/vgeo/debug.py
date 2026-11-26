"""Debug views and the stats overlay.

Debug View (scene setting, VGEO panel): every mesh VGEO writes (streamed chunks, level meshes, atom
libraries) gets a corner color attribute `vgeo_viz`, and Solid shading shows it:

- Meshlets: a color per cluster.
- LOD: an ordered ramp from full detail (blue) to the coarsest level (red).
- Triangles: a color per triangle (flat facets).

Its alpha holds the DAG depth / 16, so a material can read the same attribute (an Attribute node)
and mix it in, as the statue avenue materials do. The runtime packs (cluster << 6) | depth into each
face's LOD value, which is where the cluster comes from.

The stats overlay prints the scene's source triangles, what is drawn, and how (copies, instances,
streamed cuts) in the corner of every 3D view.
"""

import bpy
import numpy as np

ATTR = "vgeo_viz"
MODES = [('NONE', "Off", "Materials as they are"),
         ('MESHLETS', "Meshlets", "A color per cluster"),
         ('LOD', "LOD", "An ordered ramp from full detail (blue) to the coarsest level (red)"),
         ('TRIANGLES', "Triangles", "A color per triangle")]

mode = 'NONE'      # mirrors the active scene's setting; read by stream.fill_mesh
_shading = {}      # 3D view space pointer -> (shading type, color type) before a debug view
_overlay = None


def _hash_colors(ids):
    """Stable bright-ish colors from integer ids (n,) -> (n, 3) float."""
    x = ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(29)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(32)
    r = ((x >> np.uint64(0)) & np.uint64(255)).astype(np.float32)
    g = ((x >> np.uint64(8)) & np.uint64(255)).astype(np.float32)
    b = ((x >> np.uint64(16)) & np.uint64(255)).astype(np.float32)
    return 0.25 + 0.75 * np.stack([r, g, b], 1) / 255.0


def _ramp(t):
    """Blue -> cyan -> green -> yellow -> red for t in [0, 1]."""
    stops = np.array([[0.10, 0.25, 0.95], [0.10, 0.80, 0.90], [0.25, 0.85, 0.30],
                      [0.95, 0.85, 0.20], [0.90, 0.20, 0.15]], np.float32)
    x = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(x.astype(np.int64), len(stops) - 2)
    f = (x - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def face_colors(face_lod, view=None, max_depth=15):
    """(tri_count, 4) float RGBA for a chunk's packed per-face LOD values."""
    view = view or mode
    fl = np.asarray(face_lod, np.int64)
    depth = fl & 63
    cluster = fl >> 6
    if view == 'MESHLETS':
        rgb = _hash_colors(cluster)
    elif view == 'LOD':
        rgb = _ramp(depth / max(1, max_depth))
    else:   # TRIANGLES
        rgb = _hash_colors(cluster * 4096 + np.arange(len(fl)))
    a = np.minimum(depth / 16.0, 1.0)[:, None].astype(np.float32)
    return np.concatenate([rgb.astype(np.float32), a], 1)


def write_viz(me, data):
    """Write the debug color attribute into a freshly filled mesh (BYTE_COLOR, corner domain;
    written with foreach_set on float RGBA, never a raw copy)."""
    if mode == 'NONE' or data is None or not data.get("tri_count"):
        return
    col = np.repeat(face_colors(data["face_lod"]), 3, axis=0)
    a = me.color_attributes.get(ATTR)
    if a is None:
        a = me.color_attributes.new(ATTR, 'BYTE_COLOR', 'CORNER')
    a.data.foreach_set("color", np.ascontiguousarray(col, np.float32).ravel())
    me.color_attributes.active_color = a
    me.color_attributes.render_color_index = me.color_attributes.find(ATTR)


def _views():
    wm = bpy.context.window_manager
    for win in wm.windows if wm else ():
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D':
                        yield sp, area


def set_view(scene, view):
    """Switch the debug view: rewrite what VGEO drew, point Solid shading at the attribute."""
    global mode
    from . import instances, stream
    if view == mode:
        return
    mode = view
    stream.invalidate_all()                       # chunks are extracted again, with or without colors
    for inst in instances.instancers(scene):
        if inst.vgeo_inst.source is None:
            continue
        try:
            instances.build_levels(inst, inst.vgeo_inst.source)
            if inst.vgeo_inst.atoms is not None:
                instances.rebuild_atoms(inst)
        except RuntimeError as e:
            print("VGEO debug view:", e)
    for sp, area in _views():
        key = sp.as_pointer()
        sh = sp.shading
        if view != 'NONE':
            _shading.setdefault(key, (sh.type, sh.color_type))
            sh.type = 'SOLID'
            sh.color_type = 'VERTEX'   # Solid: show the active color attribute
        elif key in _shading:
            sh.type, sh.color_type = _shading.pop(key)
        area.tag_redraw()


def _on_debug(self, context):
    set_view(context.scene, self.vgeo_debug)


# ---------------------------------------------------------------- stats overlay

def scene_stats(scene):
    """Triangles in the scene's sources, triangles drawn, and how."""
    from . import instances, stream
    src = drawn = copies = inst_n = streamed = 0
    for inst in instances.instancers(scene):
        p = inst.vgeo_inst.source
        rt = stream._runtimes.get(p.vgeo.uid) if p is not None else None
        n = len(inst.data.vertices)
        per = p.vgeo.source_triangles if p is not None else 0
        if not per and rt is not None and rt.asset is not None:
            per = rt.asset.info["source_triangles"]
        src += per * n
        copies += n
        drawn += inst.vgeo_inst.shown_triangles
        if getattr(inst.vgeo_inst, "use_atoms", False):
            inst_n += sum(len(o.data.vertices) for o in instances.atom_clouds(inst).values())
        streamed += sum(1 for o in instances.slots(inst) if o.vgeo.slot_index >= 0)
    for ob in stream.proxies(scene):
        if ob.vgeo.slot_of:
            continue
        rt = stream._runtimes.get(ob.vgeo.uid)
        src += ob.vgeo.source_triangles
        if rt is not None and rt.asset is not None:
            drawn += rt.triangles
    return {"source": src, "drawn": drawn, "copies": copies, "atom_points": inst_n, "streamed": streamed}


def _fmt(n):
    for unit, d in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= d:
            return f"{n / d:.2f}{unit}"
    return str(n)


def _draw_overlay():
    import blf
    scene = bpy.context.scene
    if scene is None or not getattr(scene, "vgeo_stats", False):
        return
    st = scene_stats(scene)
    if not st["source"]:
        return
    share = 100.0 * st["drawn"] / st["source"] if st["source"] else 0.0
    lines = [f"VGEO  scene {_fmt(st['source'])} triangles",
             f"drawn {_fmt(st['drawn'])} ({share:.4f} %)"]
    if st["copies"]:
        lines.append(f"{st['copies']:,} copies, {st['atom_points']:,} atom instances, {st['streamed']} streamed")
    if mode != 'NONE':
        lines.append(f"debug view: {mode.title()}")
    font = 0
    blf.size(font, 13)
    y = 60
    for line in reversed(lines):
        blf.position(font, 20, y, 0)
        blf.color(font, 1, 1, 1, 0.9)
        blf.enable(font, blf.SHADOW)
        blf.shadow(font, 3, 0, 0, 0, 0.8)
        blf.draw(font, line)
        y += 18


def register():
    global _overlay
    bpy.types.Scene.vgeo_debug = bpy.props.EnumProperty(name="Debug View", items=MODES, default='NONE',
                                                        update=_on_debug)
    bpy.types.Scene.vgeo_stats = bpy.props.BoolProperty(name="Stats Overlay", default=False,
                                                        description="Scene and drawn triangles in the 3D view")
    if _overlay is None:
        _overlay = bpy.types.SpaceView3D.draw_handler_add(_draw_overlay, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _overlay
    if _overlay is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_overlay, 'WINDOW')
        _overlay = None
    del bpy.types.Scene.vgeo_stats
    del bpy.types.Scene.vgeo_debug
