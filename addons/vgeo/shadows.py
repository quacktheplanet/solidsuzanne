"""Sun shadows for Live Draw: a cascaded depth map from the scene's sun, sampled by the Live Draw shader.

- One sun casts shadows: the first visible sun whose Shadow toggle is on (light.use_shadow), and only while
  the scene's Live Draw Shadows toggle is on. Point lights stay unshadowed.
- Up to three cascades (one depth-texture layer each, 2048 texels square) split the part of the view that
  holds VGEO geometry, so near shadows stay sharp and far ones still exist. The light-space depth range
  covers the whole scene's bounds, so casters outside the view still shadow what is in it.
- A cascade is rendered again only when its key changes: its box (padded, kept while the view stays inside
  it, else snapped to a coarse grid and a quarter-octave size step), the sun's direction, or the casters
  (a cut swapping in, a shared level landing, placements moving). A still view costs no shadow rendering.
- Casters: a proxy draws its current cut (the same triangles the view shows, so it shadows itself without
  mismatch); an instancer draws every placement in the cascade's box with one shared level, the coarsest
  whose error stays under a shadow texel (the near copies' own cuts are not used: copies are cheaper at a
  level, and the shadow map can't resolve more).
- Ordinary meshes (not virtualized; livedraw's receivers) cast into a second set of layers, re-rendered
  only when they move or change. Virtualized surfaces test both sets; ordinary meshes test only the
  virtualized one (Blender already shadows its own meshes with each other), and are darkened by a second
  pass over Blender's image (livedraw._draw_receivers).
- Sampling (livedraw's fragment shader): 4x4 texel fetches weighted into a 3-texel tent (PCF), a depth
  bias of one texel plus a slope-scaled part, and a normal offset that grows with the angle to the light.
"""
import math
import time

import gpu
import numpy as np
from mathutils import Matrix, Vector

RES = 2048               # texels per cascade side
MAX_CASCADES = 3
PAD = 1.15               # a new box is this much larger than the area it must cover
KEEP = 1.7               # a box is kept while the area it must cover is at least 1/KEEP of it
ERR_TEXELS = 1.0         # a shared level may be off by this many shadow texels
MAX_TRIS = 8_000_000     # per cascade: instanced copies go coarser rather than draw more
SPLIT_LAMBDA = 0.95      # cascade splits: 1 = logarithmic, 0 = uniform
MAX_MAPS = 4             # 3D views with a shadow map of their own (48 MB each; a quad view has 4)

_maps = {}               # region_3d pointer -> ShadowMap
_shader = None
_none = {}
sync_timing = False      # benchmarks: wait for the GPU after each cascade, so render_ms is its real cost
stats = {"renders": 0, "render_ms": [], "update_ms": [], "cascades": 0, "tris": 0, "levels": []}


_VERT = """
void main() {
  int row = inst_base + gl_InstanceID;
  mat4 M = mat4(texelFetch(inst_tex, ivec2(0, row), 0), texelFetch(inst_tex, ivec2(1, row), 0),
                texelFetch(inst_tex, ivec2(2, row), 0), texelFetch(inst_tex, ivec2(3, row), 0));
  gl_Position = viewproj * (M * vec4(pos, 1.0));
  gl_Position.z += depth_push;   /* a coarse level is pushed away from the light by its error */
}
"""

_FRAG = """
void main() {
}
"""


def depth_shader():
    global _shader
    if _shader is None:
        ci = gpu.types.GPUShaderCreateInfo()
        ci.vertex_in(0, 'VEC3', "pos")
        ci.push_constant('MAT4', "viewproj")
        ci.push_constant('INT', "inst_base")
        ci.push_constant('FLOAT', "depth_push")
        ci.sampler(0, 'FLOAT_2D', "inst_tex")
        ci.vertex_source(_VERT)
        ci.fragment_source(_FRAG)
        _shader = gpu.shader.create_from_info(ci)
    return _shader


def _ubo_data(mats, info, bias):
    data = []
    for c in range(MAX_CASCADES):
        m = mats[c] if c < len(mats) else None
        if m is None:
            data += [0.0] * 16
        else:   # column-major
            data += [m[r][k] for k in range(4) for r in range(4)]
    data += list(info)
    for c in range(MAX_CASCADES):
        data += list(bias[c]) if c < len(bias) else [0.0] * 4
    return gpu.types.GPUUniformBuf(gpu.types.Buffer('FLOAT', len(data), data))


def none_bindings():
    """(ubo, texture) that switch shadows off in the shader."""
    if not _none:
        _none["ubo"] = _ubo_data([], (-1.0, 0.0, 1.0, 0.0), [])
        _none["tex"] = gpu.types.GPUTexture((1, 1), layers=1, format='DEPTH_COMPONENT32F')
        fb = gpu.types.GPUFrameBuffer(depth_slot={"texture": _none["tex"], "layer": 0})
        with fb.bind():
            fb.clear(depth=1.0)
    return _none["ubo"], _none["tex"]


class ShadowMap:
    """Layers 0..2: the cascades of virtualized casters; 3..5: the same cascades of ordinary meshes."""

    def __init__(self):
        self.tex = gpu.types.GPUTexture((RES, RES), layers=2 * MAX_CASCADES, format='DEPTH_COMPONENT32F')
        self.fbs = [gpu.types.GPUFrameBuffer(depth_slot={"texture": self.tex, "layer": c})
                    for c in range(2 * MAX_CASCADES)]
        self.boxes = [None] * MAX_CASCADES      # (cx, cy, size)
        self.keys = [None] * MAX_CASCADES
        self.keys_b = [None] * MAX_CASCADES     # the ordinary meshes' layer of each cascade
        self.mats = [None] * MAX_CASCADES       # world -> (u, v, depth)
        self.bias = [(0.0, 0.0, 0.0, 0.0)] * MAX_CASCADES
        self.keep = [None] * MAX_CASCADES       # matrix textures in use by a cascade's last render
        self.count = 0
        self.ubo = None
        self.ubo_key = None
        self.used = 0.0


def free_all():
    _maps.clear()


def _box8(a, b):
    return np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])], np.float64)


def _basis(d):
    d = Vector(d).normalized()
    up = Vector((0.0, 0.0, 1.0)) if abs(d.z) < 0.99 else Vector((0.0, 1.0, 0.0))
    r = up.cross(d).normalized()
    u = d.cross(r)
    return r, u, d


def _corners(view, window, persp, d0, d1):
    """World corners of the view frustum between view depths d0 and d1."""
    inv_w = window.inverted()
    inv_v = view.inverted()
    out = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            v = inv_w @ Vector((x, y, -1.0, 1.0))
            v = v.xyz / v.w
            for t in (d0, d1):
                if persp:
                    p = v * (t / max(-v.z, 1e-9))
                else:
                    p = Vector((v.x, v.y, -t))
                out.append(inv_v @ p)
    return out


def _splits(d0, d1, n):
    out = [d0]
    near = max(d0, d1 / 2000.0)      # a tiny clip start would give the first cascade nothing
    for i in range(1, n):
        f = i / n
        log = near * (d1 / near) ** f
        out.append(SPLIT_LAMBDA * log + (1.0 - SPLIT_LAMBDA) * (d0 + (d1 - d0) * f))
    out.append(d1)
    return out


def _box(prev, x0, x1, y0, y1):
    """A square light-space box (cx, cy, size) covering [x0, x1] x [y0, y1]: the previous one while it still
    covers it and isn't much too large, else a new one on a coarse grid (so small moves don't re-render)."""
    need = max(x1 - x0, y1 - y0, 1e-6)
    if prev is not None:
        cx, cy, s = prev
        h = s * 0.5
        if (cx - h <= x0 and x1 <= cx + h and cy - h <= y0 and y1 <= cy + h and s <= need * PAD * KEEP):
            return prev
    s = 2.0 ** (math.ceil(4.0 * math.log2(need * PAD)) / 4.0)
    while True:
        g = s / 8.0
        cx = round((x0 + x1) * 0.5 / g) * g
        cy = round((y0 + y1) * 0.5 / g) * g
        h = s * 0.5
        if cx - h <= x0 and x1 <= cx + h and cy - h <= y0 and y1 <= cy + h:
            return (cx, cy, s)
        s *= 2.0 ** 0.25


def _zrange(lo, hi):
    rng = max(hi - lo, 1e-3)
    step = 2.0 ** math.floor(math.log2(rng / 8.0))
    return math.floor((lo - rng * 0.01) / step) * step, math.ceil((hi + rng * 0.01) / step) * step


def _light_matrix(r, u, d, box, z0, z1):
    """World -> clip space of the light's orthographic view (GL conventions: depth -1 at the light)."""
    cx, cy, s = box
    V = Matrix((tuple(r) + (0.0,), tuple(u) + (0.0,), tuple(d) + (0.0,), (0.0, 0.0, 0.0, 1.0)))
    zr = z1 - z0
    P = Matrix(((2.0 / s, 0.0, 0.0, -2.0 * cx / s),
                (0.0, 2.0 / s, 0.0, -2.0 * cy / s),
                (0.0, 0.0, -2.0 / zr, (z1 + z0) / zr),
                (0.0, 0.0, 0.0, 1.0)))
    return P @ V


_TO_TEX = Matrix(((0.5, 0.0, 0.0, 0.5), (0.0, 0.5, 0.0, 0.5), (0.0, 0.0, 0.5, 0.5), (0.0, 0.0, 0.0, 1.0)))


def _cascade_items(casters, r, u, box, texel, zr, ls_wanted):
    """What a cascade draws: [(matrix texture, row0, count, batch, tris, depth push)], plus the textures to
    keep alive. A shared level stands in for finer geometry (a copy's own cut, or a finer level) in the
    view: where that surface lies under the coarse one by up to the level's error, the coarse caster would
    shadow it, so its depth is pushed away from the light by twice that error (a fraction of a texel)."""
    from . import livedraw
    cx, cy, s = box
    h = s * 0.5
    x0, x1, y0, y1 = cx - h, cx + h, cy - h, cy + h
    rv = np.array(tuple(r), np.float64)
    uv = np.array(tuple(u), np.float64)
    items, keep = [], []
    for c in casters:
        if c["kind"] == "proxy":
            pts = c["corners"]
            px, py = pts @ rv, pts @ uv
            if px.max() < x0 or px.min() > x1 or py.max() < y0 or py.min() > y1:
                continue
            for _m, batch, tris in c["entries"]:
                items.append((c["tex"], 0, 1, batch, tris, 0.0))
            continue
        world, radius = c["world"], c["radius"]
        if not len(world):
            continue
        pts = world[:, :3, 3].astype(np.float64)
        px, py = pts @ rv, pts @ uv
        mask = (px + radius >= x0) & (px - radius <= x1) & (py + radius >= y0) & (py - radius <= y1)
        n = int(mask.sum())
        if not n:
            continue
        ls, errors, tris = c["ls"], c["errors"], c["tris"]
        allowed = texel * ERR_TEXELS / max(c["max_scale"], 1e-9)       # in asset units
        lv = int(np.searchsorted(errors, allowed, side="right") - 1)
        lv = ls.clamp(min(max(lv, 0), len(errors) - 1))
        while lv < len(errors) - 1 and n * int(tris[lv]) > MAX_TRIS:
            lv += 1
        got = ls.available(lv)
        if got != lv:
            ls_wanted.append(lv)
        if got is None:
            continue
        tex = livedraw.matrix_texture(world[mask])
        keep.append(tex)
        push = 2.0 * 2.0 * float(errors[got]) * c["max_scale"] / zr     # clip-space z spans 2 over zr
        for _m, batch, t in ls.batches.get(got, []):
            items.append((tex, 0, n, batch, t, push))
        stats["levels"].append((round(texel, 4), lv, got, n, round(float(errors[got]) * c["max_scale"], 4)))
    return items, keep


def _ordinary_items(receivers, r, u, box):
    """The ordinary meshes (Live Draw's receivers) that touch a cascade's box, as casters."""
    from . import livedraw
    cx, cy, s = box
    h = s * 0.5
    rv = np.array(tuple(r), np.float64)
    uv = np.array(tuple(u), np.float64)
    items = []
    for ob, rc in receivers:
        try:
            mw = ob.matrix_world
        except ReferenceError:
            continue
        pts = (np.c_[rc.corners, np.ones(8)] @ np.array(mw, np.float64).T)[:, :3]
        px, py = pts @ rv, pts @ uv
        if px.max() < cx - h or px.min() > cx + h or py.max() < cy - h or py.min() > cy + h:
            continue
        tex = livedraw._one_matrix(("recv", ob.name), mw)
        for _m, batch, tris in rc.batches:
            items.append((tex, 0, 1, batch, tris, 0.0))
    return items


def _render(sm, c, mat, items):
    sh = depth_shader()
    fb = sm.fbs[c]
    t0 = time.perf_counter()
    tris = 0
    with fb.bind():
        fb.clear(depth=1.0)
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        gpu.state.face_culling_set('NONE')
        sh.bind()
        sh.uniform_float("viewproj", mat)
        last_tex = last_row = last_push = None
        for tex, row0, count, batch, t, push in items:
            if push != last_push:
                sh.uniform_float("depth_push", push)
                last_push = push
            if tex is not last_tex:
                sh.uniform_sampler("inst_tex", tex)
                last_tex = tex
            if row0 != last_row:
                sh.uniform_int("inst_base", row0)
                last_row = row0
            if count == 1:
                batch.draw(sh)
            else:
                batch.draw_instanced(sh, instance_start=0, instance_count=count)
            tris += t * count
        if sync_timing:
            fb.read_depth(0, 0, 1, 1)
    stats["renders"] += 1
    stats["render_ms"].append((time.perf_counter() - t0) * 1000.0)
    if len(stats["render_ms"]) > 240:
        del stats["render_ms"][:120]
    return tris


def update(rv3d, space, light, frame):
    """Bring this view's shadow map up to date. light: (shader light index, direction to the light) or None.
    Returns (ubo, texture) for the Live Draw shader, or None when nothing casts or receives shadows."""
    bounds = frame.get("bounds")
    casters = frame.get("casters") or []
    if light is None or bounds is None or not casters:
        return None
    t_start = time.perf_counter()
    idx, ldir = light
    key_view = rv3d.as_pointer()
    sm = _maps.get(key_view)
    if sm is None:
        if len(_maps) >= MAX_MAPS:
            _maps.pop(min(_maps, key=lambda k: _maps[k].used))
        sm = _maps[key_view] = ShadowMap()
    sm.used = t_start
    r, u, d = _basis(ldir)
    lo, hi = bounds
    rb = frame.get("receiver_bounds")
    if rb is not None:            # ordinary meshes receive too: they extend the depths, not the casters' area
        lo, hi = np.minimum(lo, rb[0]), np.maximum(hi, rb[1])
    cc = _box8(*bounds)
    bc = _box8(lo, hi)
    rv, uv, dv = (np.array(tuple(a), np.float64) for a in (r, u, d))
    sx, sy, sz = cc @ rv, cc @ uv, bc @ dv
    z0, z1 = _zrange(float(sz.min()), float(sz.max()))
    # the view depths that hold geometry
    view, window = rv3d.view_matrix, rv3d.window_matrix
    persp = rv3d.is_perspective
    vm = np.array(view, np.float64)
    depth = -(bc @ vm[2, :3].T + vm[2, 3])
    if persp:
        d0 = max(float(space.clip_start), float(depth.min()))
        d1 = min(float(space.clip_end), float(depth.max()))
    else:
        d0, d1 = float(depth.min()), float(depth.max())
    if d1 <= d0:
        return None
    if persp:
        ratio = d1 / max(d0, 1e-6)
        n = 1 if ratio < 6.0 else (2 if ratio < 40.0 else 3)
    else:
        n = 1
    n = min(n, MAX_CASCADES)
    splits = _splits(d0, d1, n)
    ldkey = tuple(round(x, 4) for x in d)
    sig = frame.get("shadow_sig")
    receivers = frame.get("receivers") or []
    rsig = frame.get("receiver_sig") if receivers else None
    wanted = []
    del stats["levels"][:-24]
    for c in range(n):
        pts = _corners(view, window, persp, splits[c], splits[c + 1])
        px = [p.dot(r) for p in pts]
        py = [p.dot(u) for p in pts]
        x0, x1 = max(min(px), float(sx.min())), min(max(px), float(sx.max()))
        y0, y1 = max(min(py), float(sy.min())), min(max(py), float(sy.max()))
        if x1 < x0 or y1 < y0:            # nothing of the scene in this slice: an empty, tiny box
            x0 = x1 = (x0 + x1) * 0.5
            y0 = y1 = (y0 + y1) * 0.5
        box = _box(sm.boxes[c] if sm.keys[c] is not None and sm.keys[c][1] == ldkey else None, x0, x1, y0, y1)
        sm.boxes[c] = box
        key = (box, ldkey, z0, z1, sig)
        key_b = (box, ldkey, z0, z1, rsig) if receivers else None
        if key == sm.keys[c] and key_b == sm.keys_b[c]:
            continue
        mat = _light_matrix(r, u, d, box, z0, z1)
        texel = box[2] / RES
        if key != sm.keys[c]:
            items, keep = _cascade_items(casters, r, u, box, texel, z1 - z0, wanted)
            stats["tris"] = _render(sm, c, mat, items)
            sm.keep[c] = keep
            sm.keys[c] = key
        if key_b is not None and key_b != sm.keys_b[c]:
            _render(sm, MAX_CASCADES + c, mat, _ordinary_items(receivers, r, u, box))
        sm.keys_b[c] = key_b
        sm.mats[c] = _TO_TEX @ mat
        sm.bias[c] = (texel, 1.0 / (z1 - z0), 0.0, 0.0)
    for c in range(n, MAX_CASCADES):
        sm.keys[c] = sm.keys_b[c] = None
        sm.keep[c] = None
    sm.count = n
    stats["cascades"] = n
    ukey = (tuple(sm.keys[:n]), tuple(sm.keys_b[:n]), idx)
    if ukey != sm.ubo_key:
        sm.ubo = _ubo_data(sm.mats[:n], (float(idx), float(n), float(RES), 1.0 if receivers else 0.0),
                           sm.bias[:n])
        sm.ubo_key = ukey
    if wanted:
        frame["busy"] = True               # the live tick builds the wanted levels; the map redraws then
    stats["update_ms"].append((time.perf_counter() - t_start) * 1000.0)
    if len(stats["update_ms"]) > 240:
        del stats["update_ms"][:120]
    return sm.ubo, sm.tex
