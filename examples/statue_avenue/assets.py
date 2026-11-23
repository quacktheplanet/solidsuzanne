"""Procedural sculpted assets for the Solid Suzanne demo: a carved stone Suzanne monolith on a plinth,
and a paved floor patch. Detail is real geometry (numpy displacement on dense meshes); a glow mask for
the carved sci-fi channels is stored in the UV map's U coordinate so it survives virtualization.
"""
import math

import bmesh
import bpy
import numpy as np

# ---------------------------------------------------------------- numpy noise
_rng = np.random.default_rng(11)
_PERM = np.concatenate([_rng.permutation(256)] * 2)
_G3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float64)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
    xf, yf, zf = x - xi, y - yi, z - zi
    xi &= 255; yi &= 255; zi &= 255
    u, v, w = _fade(xf), _fade(yf), _fade(zf)

    def g(ix, iy, iz, dx, dy, dz):
        gr = _G3[_PERM[_PERM[_PERM[ix] + iy] + iz] % 12]
        return gr[:, 0] * dx + gr[:, 1] * dy + gr[:, 2] * dz

    n000 = g(xi, yi, zi, xf, yf, zf); n100 = g(xi + 1, yi, zi, xf - 1, yf, zf)
    n010 = g(xi, yi + 1, zi, xf, yf - 1, zf); n110 = g(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf)
    n001 = g(xi, yi, zi + 1, xf, yf, zf - 1); n101 = g(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1)
    n011 = g(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1); n111 = g(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1)
    x1 = n000 + u * (n100 - n000); x2 = n010 + u * (n110 - n010)
    x3 = n001 + u * (n101 - n001); x4 = n011 + u * (n111 - n011)
    y1 = x1 + v * (x2 - x1); y2 = x3 + v * (x4 - x3)
    return y1 + w * (y2 - y1)


def fbm(p, octaves, freq, gain=0.5, off=0.0):
    s, a, f = 0.0, 1.0, freq
    for i in range(octaves):
        s = s + a * perlin(p * f + off + 17.13 * i)
        a *= gain; f *= 2.03
    return s


def voronoi_edge(p, scale, seed=0):
    """(F2 - F1) distance for a jittered 3D grid: ~0 on cell borders (crack network)."""
    q = p * scale
    c = np.floor(q).astype(np.int64)
    f1 = np.full(len(p), 9.0); f2 = np.full(len(p), 9.0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cc = c + np.array([dx, dy, dz])
                h = (cc[:, 0] * 73856093) ^ (cc[:, 1] * 19349663) ^ (cc[:, 2] * 83492791) ^ (seed * 2654435761)
                h = h & 0xFFFFFFFF
                j = np.stack([((h * k) % 10007) / 10007.0 for k in (1, 7, 13)], 1)
                d = np.linalg.norm(cc + j - q, axis=1)
                f2 = np.where(d < f1, f1, np.minimum(f2, d))
                f1 = np.minimum(f1, d)
    return (f2 - f1) / scale


# ---------------------------------------------------------------- helpers
def _mesh_arrays(me):
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
    no = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("normal", no)
    return co.reshape(-1, 3), no.reshape(-1, 3)


def _apply_modifiers(ob):
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.view_layer.objects:
        o.select_set(o == ob)
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def _write_glow_uv(me, glow_per_vertex):
    uv = me.uv_layers.new(name="UVMap") if not me.uv_layers else me.uv_layers[0]
    loops_v = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", loops_v)
    g = glow_per_vertex[loops_v]
    arr = np.zeros((len(me.loops), 2), np.float32); arr[:, 0] = g
    uv.data.foreach_set("uv", arr.ravel())


# ---------------------------------------------------------------- monolith
def make_monolith(name="Monolith", subdiv=5, plinth_cuts=180):
    """Suzanne bust (~6 m) on an engraved plinth. Origin at the plinth base."""
    # head
    bpy.ops.mesh.primitive_monkey_add(size=2.0)
    head = bpy.context.object
    sub = head.modifiers.new("sub", "SUBSURF"); sub.levels = subdiv; sub.render_levels = subdiv
    _apply_modifiers(head)
    me = head.data
    co, no = _mesh_arrays(me)
    p = co.copy()
    # eroded stone: broad weathering, chisel facets, crack network, grain
    d = 0.010 * fbm(p, 4, 2.2)
    d += 0.006 * fbm(p, 3, 9.0, off=3.1)
    e = voronoi_edge(p, 4.5, 1)
    crack_mask = np.clip(fbm(p, 2, 1.3, off=8.0) * 2.2 + 0.25, 0, 1)
    d -= 0.020 * np.exp(-(e / 0.012) ** 2) * crack_mask
    d += 0.0016 * fbm(p, 2, 60.0, off=5.0)
    # sci-fi carved channels: rings around the skull and vertical segments; these glow
    ang = np.arctan2(p[:, 0], p[:, 1] + 0.2)
    glow = np.zeros(len(p))
    for zc, w in ((0.62, 0.016), (0.80, 0.010), (-0.30, 0.012)):
        band = np.exp(-((p[:, 2] - zc) / w) ** 2)
        seg = (np.sin(ang * 14 + zc * 9) > -0.55).astype(np.float64)
        glow = np.maximum(glow, band * seg)
    vert = np.exp(-((np.sin(ang * 7 + 0.4)) / 0.035) ** 2) * ((p[:, 2] > 0.62) & (p[:, 2] < 0.80))
    glow = np.maximum(glow, vert)
    d -= 0.018 * glow
    co = co + no * d[:, None]
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    _write_glow_uv(me, np.clip(glow, 0, 1).astype(np.float32))
    head.scale = (2.6, 2.6, 2.6)
    head.location = (0, 0, 4.0 + 2.6 * 0.95)
    head.rotation_euler = (0.0, 0.0, 0.0)

    # plinth: bevelled box, densely subdivided, engraved panels + chipped edges
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    pl = bpy.context.object
    pl.scale = (3.6, 3.6, 4.0)
    bpy.ops.object.transform_apply(scale=True)
    bv = pl.modifiers.new("bev", "BEVEL"); bv.width = 0.18; bv.segments = 4
    _apply_modifiers(pl)
    bm = bmesh.new(); bm.from_mesh(pl.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(pl.data); bm.free()
    rem = pl.modifiers.new("rem", "REMESH"); rem.mode = 'VOXEL'; rem.voxel_size = 3.6 / plinth_cuts
    _apply_modifiers(pl)
    me2 = pl.data
    co2, no2 = _mesh_arrays(me2)
    q = co2.copy()
    d2 = 0.010 * fbm(q, 3, 0.9) + 0.003 * fbm(q, 3, 6.0, off=2.0)
    # panel grooves: grid lines on each face (in the face plane), plus a glowing glyph band
    gx = np.abs(((q[:, 0] + 0.15) / 0.9) % 1.0 - 0.5); gy = np.abs(((q[:, 1] + 0.15) / 0.9) % 1.0 - 0.5)
    gz = np.abs((q[:, 2] / 0.8) % 1.0 - 0.5)
    lines = np.maximum.reduce([np.exp(-((0.5 - gx) / 0.012) ** 2), np.exp(-((0.5 - gy) / 0.012) ** 2), np.exp(-((0.5 - gz) / 0.012) ** 2)])
    d2 -= 0.025 * lines
    glyph = np.exp(-((q[:, 2] - 1.4) / 0.05) ** 2) * (np.sin((q[:, 0] + q[:, 1]) * 11.0) > 0.0)
    glow2 = np.clip(glyph, 0, 1)
    d2 -= 0.03 * glow2
    # chipped edges near the box edges
    edge = (np.abs(q[:, 0]) > 1.6) & (np.abs(q[:, 1]) > 1.6) | (np.abs(q[:, 2]) > 1.85)
    d2 -= edge * np.clip(fbm(q, 3, 2.5, off=4.0) * 0.08 + 0.02, 0, 0.09)
    co2 = co2 + no2 * d2[:, None]
    me2.vertices.foreach_set("co", co2.ravel())
    me2.update()
    _write_glow_uv(me2, glow2.astype(np.float32))
    pl.location = (0, 0, 2.0)

    # join into one object, origin at the base
    for o in bpy.context.view_layer.objects:
        o.select_set(o in (head, pl))
    bpy.context.view_layer.objects.active = pl
    bpy.ops.object.join()
    ob = bpy.context.object
    ob.name = name
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.shade_smooth()
    return ob


def make_floor_patch(name="Floor", size=24.0, n=900, tile=2.0):
    """Square paving patch (origin at centre), tiles with chamfered grooves, cracks, slight tilt per tile."""
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=n, y_subdivisions=n, size=size)
    ob = bpy.context.object
    ob.name = name
    me = ob.data
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    x, y = co[:, 0], co[:, 1]
    tx, ty = np.floor(x / tile), np.floor(y / tile)
    fx, fy = x / tile - tx, y / tile - ty
    # per-tile tilt/height from a hash
    h = ((tx.astype(np.int64) * 73856093) ^ (ty.astype(np.int64) * 19349663)) & 0xFFFF
    hz = (h % 97) / 97.0 - 0.5
    tilt_x = ((h // 97) % 89) / 89.0 - 0.5
    tilt_y = ((h // 8633) % 7) / 7.0 - 0.5
    z = 0.025 * hz + 0.02 * tilt_x * (fx - 0.5) + 0.02 * tilt_y * (fy - 0.5)
    # chamfered groove between tiles
    bx = np.minimum(fx, 1 - fx); by = np.minimum(fy, 1 - fy)
    b = np.minimum(bx, by) * tile
    z -= 0.06 * np.clip(1 - b / 0.05, 0, 1) ** 1.5
    p3 = np.stack([x, y, np.zeros_like(x)], 1)
    e = voronoi_edge(p3, 0.9, 3)
    crack_mask = np.clip(fbm(p3, 2, 0.15, off=2.0) * 2.5, 0, 1)
    z -= 0.018 * np.exp(-(e / 0.010) ** 2) * crack_mask
    z += 0.004 * fbm(p3, 3, 3.0, off=6.0) + 0.0012 * fbm(p3, 2, 25.0)
    # inlaid glowing lines every 4 tiles (sci-fi floor circuitry)
    gl = (np.exp(-((fx - 0.5) / 0.022) ** 2) * (tx % 8 == 0) + np.exp(-((fy - 0.5) / 0.022) ** 2) * (ty % 12 == 0))
    gl = np.clip(gl, 0, 1)
    z -= 0.015 * gl
    co[:, 2] = z
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    _write_glow_uv(me, gl.astype(np.float32))
    bpy.ops.object.shade_smooth()
    return ob
