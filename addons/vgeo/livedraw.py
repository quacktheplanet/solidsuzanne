"""VGEO Live Draw: virtualized assets drawn straight from GPU buffers in Solid and Material Preview views.

The Blender-mesh path (stream.py) feeds cuts to EEVEE and Cycles as ordinary meshes; every change makes
Blender rebuild that mesh's buffers, and a 1,000-copy field draws whole-rock levels per copy. Live Draw
keeps Blender's renderers for final renders (and Rendered viewports) and draws the viewport itself:

- every vertex of an asset is uploaded once (one GPUVertBuf per .vgeo); the cluster DAG indexes those
  vertices globally, so every cut and every instance level is just index data;
- a cut keeps one index buffer per group of chunks and material; a view change re-selects (native, ~1 ms)
  and rebuilds only the groups whose chunks changed, double-buffered so a cut never shows half old, half
  new (no cracks), and time-budgeted per tick;
- an instancer's error levels are shared index buffers; placements are drawn instanced per level, their
  matrices in a float texture; the nearest heavy copies get a view-dependent cut of their own;
- shading reads each material's Principled BSDF (base colour, roughness, normal map via a cotangent frame,
  image or value), lit by the scene's lights and the world colour, tone-mapped like the scene's view
  transform (AgX approximated), depth-tested against the viewport so it mixes with other objects.

Final renders and Rendered viewports keep the Blender-object path untouched; Live Draw switches itself
off while any 3D view is in Rendered shading.
"""
import math
import time

import bpy
import gpu
import numpy as np

from . import native, shadows

GROUP_CHUNKS = 16        # chunks per index buffer: fewer draws, still small rebuilds
TICK_BUDGET = 0.004      # seconds of index-buffer building per tick
RESELECT = 0.1          # seconds between re-selections of one cut while the view keeps moving
MIN_SLICE = 0.001        # every build step gets at least this, whatever the bookkeeping before it cost
NEAR_COPIES = 32         # nearest heavy placements that get a view-dependent cut of their own
LEVEL_ERROR_SCALE = 2.0  # whole-asset levels draw their hidden back too: at twice the pixel error they cost
                         # about what a view-dependent cut costs at 1 px (near copies keep the real error)
NEAR_MIN_TRIS = 150_000  # a whole-asset level above this costs more than the copy's own cut (~200k)
MAX_LIGHTS = 4

_assets = {}             # .vgeo path -> GpuAsset
_cuts = {}               # key -> Cut (proxies: uid; near copies: (inst uid, slot k))
_levels = {}             # instancer uid -> LevelSet
_state = {"active": False, "handler": None, "drawn": 0, "frame_ms": [], "stats": {}, "by": {}, "landed": 0}
_shader = None
_textures = {}           # image name -> GPUTexture
_dummy = {}


# ---------------------------------------------------------------- settings

def active(context=None):
    """Live Draw runs when the scene asks for it and no 3D view renders (Rendered shading keeps the
    Blender path: EEVEE/Cycles need the real meshes)."""
    ctx = context or bpy.context
    scene = getattr(ctx, "scene", None)
    if scene is None or not getattr(scene, "vgeo_live_draw", False) or bpy.app.background:
        return False
    wm = ctx.window_manager
    seen = False
    for win in wm.windows:
        if win.screen is None:
            continue
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                seen = True
                if area.spaces.active.shading.type == 'RENDERED':
                    return False
    return seen


# ---------------------------------------------------------------- shader

_TYPEDEF = """
struct LiveMaterial {
  vec4 base_color;
  vec4 params;      /* x roughness, y metallic, z normal strength, w exposure */
  vec4 flags;       /* x base tex, y rough tex, z normal tex, w tone map (1 = AgX) */
  vec4 uv_map;      /* xy scale, zw offset */
  vec4 ambient;     /* rgb world colour * strength */
  vec4 light_vec[4];  /* xyz: direction to the light (sun) or position (point); w: 0 none, 1 sun, 2 point */
  vec4 light_col[4];  /* rgb: sun irradiance, or point power / (4 pi) */
};
struct LiveShadow {
  mat4 mat[3];      /* per cascade: world -> shadow map (xy texture coordinates, z depth, 0 at the light) */
  vec4 info;        /* x: index of the light that casts shadows (-1: none), y: cascades, z: map size (texels) */
  vec4 bias[3];     /* per cascade: x texel size in world units, y depth units per world unit */
};
"""

_VERT = """
void main() {
  int row = inst_base + gl_InstanceID;
  mat4 M = mat4(texelFetch(inst_tex, ivec2(0, row), 0), texelFetch(inst_tex, ivec2(1, row), 0),
                texelFetch(inst_tex, ivec2(2, row), 0), texelFetch(inst_tex, ivec2(3, row), 0));
  vec4 wp = M * vec4(pos, 1.0);
  wpos = wp.xyz;
  wnrm = mat3(M) * nrm;
  vuv = uv;
  gl_Position = viewproj * wp;
}
"""

_FRAG = """
vec3 agx_contrast(vec3 x) {
  vec3 x2 = x * x; vec3 x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx_display_linear(vec3 v) {
  const mat3 m = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                      0.0784335999999992, 0.878468636469772, 0.0784336,
                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  const mat3 mi = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  const float lo = -12.47393, hi = 4.026069;
  v = m * max(v, vec3(1e-10));
  v = clamp(log2(v), lo, hi);
  v = (v - lo) / (hi - lo);
  v = agx_contrast(v);
  v = mi * v;
  /* back to linear: the viewport's overlay buffer is sRGB-encoded on display */
  return pow(max(v, vec3(0.0)), vec3(2.2));
}
float ggx(float nh, float a) { float a2 = a * a; float d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (3.14159265 * d * d); }
float smith(float nv, float nl, float a) {
  float k = a * 0.5; return (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
}
/* Sun shadow: 4x4 texel fetches weighted into a 3-texel tent (PCF) in the first cascade that holds the
   point. Normal offset and slope-scaled depth bias in shadow texels, so neither acne nor peter-panning. */
float sun_shadow(vec3 P0, vec3 Ng, vec3 L) {
  int n = int(S.info.y);
  float res = S.info.z;
  float nl = clamp(dot(Ng, L), 0.0, 1.0);
  float sinl = sqrt(max(1.0 - nl * nl, 0.0));
  float tanl = min(sinl / max(nl, 1e-3), 5.0);
  for (int c = 0; c < 3; c++) {
    if (c >= n) break;
    float texel = S.bias[c].x;
    vec4 s = S.mat[c] * vec4(P0 + Ng * (texel * 1.5 * sinl), 1.0);
    float m = 2.5 / res;
    if (s.x < m || s.y < m || s.x > 1.0 - m || s.y > 1.0 - m) continue;
    if (s.z >= 1.0) return 1.0;
    float ref = s.z - S.bias[c].y * texel * (1.0 + 1.5 * tanl);
    vec2 t = s.xy * res - 0.5;
    vec2 f = fract(t);
    ivec2 i0 = ivec2(floor(t)) - 1;
    float lit = 0.0;
    for (int j = 0; j < 4; j++) {
      float wy = (j == 0) ? 1.0 - f.y : ((j == 3) ? f.y : 1.0);
      for (int i = 0; i < 4; i++) {
        float wx = (i == 0) ? 1.0 - f.x : ((i == 3) ? f.x : 1.0);
        float d = texelFetch(shadow_tex, ivec3(i0 + ivec2(i, j), c), 0).r;
        lit += wx * wy * step(ref, d);
      }
    }
    return lit / 9.0;
  }
  return 1.0;
}
void main() {
  vec2 tuv = vuv * P.uv_map.xy + P.uv_map.zw;
  vec3 N = normalize(wnrm);
  if (!gl_FrontFacing) N = -N;
  vec3 albedo = P.base_color.rgb;
  if (P.flags.x > 0.5) albedo *= texture(base_tex, tuv).rgb;
  float rough = P.params.x;
  if (P.flags.y > 0.5) rough = texture(rough_tex, tuv).r;
  rough = clamp(rough, 0.04, 1.0);
  if (P.flags.z > 0.5) {
    vec3 tn = texture(normal_tex, tuv).xyz * 2.0 - 1.0;
    vec3 dp1 = dFdx(wpos), dp2 = dFdy(wpos);
    vec2 duv1 = dFdx(tuv), duv2 = dFdy(tuv);
    vec3 dp2perp = cross(dp2, N), dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float im = inversesqrt(max(max(dot(T, T), dot(B, B)), 1e-20));
    vec3 Nm = normalize(mat3(T * im, B * im, N) * tn);
    N = normalize(mix(N, Nm, P.params.z));
  }
  vec3 V = normalize(cam_pos.xyz - wpos);
  int shadow_light = int(S.info.x);
  vec3 Ng = cross(dFdx(wpos), dFdy(wpos));
  Ng = dot(Ng, Ng) > 1e-24 ? normalize(Ng) : N;
  if (dot(Ng, V) < 0.0) Ng = -Ng;
  float nv = max(dot(N, V), 1e-4);
  float metal = P.params.y;
  vec3 F0 = mix(vec3(0.04), albedo, metal);
  vec3 diff = albedo * (1.0 - metal);
  float a = rough * rough;
  vec3 col = diff * P.ambient.rgb + F0 * P.ambient.rgb * 0.5;
  for (int i = 0; i < 4; i++) {
    float kind = P.light_vec[i].w;
    if (kind < 0.5) continue;
    vec3 L; vec3 E;
    if (kind < 1.5) { L = normalize(P.light_vec[i].xyz); E = P.light_col[i].rgb; }
    else { vec3 d = P.light_vec[i].xyz - wpos; float r2 = max(dot(d, d), 1e-4); L = d * inversesqrt(r2); E = P.light_col[i].rgb / r2; }
    float nl = dot(N, L);
    if (nl <= 0.0) continue;
    vec3 H = normalize(L + V);
    float nh = max(dot(N, H), 0.0);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    vec3 spec = F * ggx(nh, a) * smith(nv, nl, a) / max(4.0 * nv * nl, 1e-4);
    if (i == shadow_light) E *= sun_shadow(wpos, Ng, L);
    col += (diff / 3.14159265 + spec) * E * nl;
  }
  col *= P.params.w;
  if (P.flags.w > 0.5) col = agx_display_linear(col);
  frag = vec4(col, 1.0);
}
"""


def shader():
    global _shader
    if _shader is not None:
        return _shader
    ci = gpu.types.GPUShaderCreateInfo()
    ci.typedef_source(_TYPEDEF)
    ci.vertex_in(0, 'VEC3', "pos")
    ci.vertex_in(1, 'VEC3', "nrm")
    ci.vertex_in(2, 'VEC2', "uv")
    iface = gpu.types.GPUStageInterfaceInfo("vgeo_live_iface")
    iface.smooth('VEC3', "wpos")
    iface.smooth('VEC3', "wnrm")
    iface.smooth('VEC2', "vuv")
    ci.vertex_out(iface)
    ci.push_constant('MAT4', "viewproj")
    ci.push_constant('VEC4', "cam_pos")
    ci.push_constant('INT', "inst_base")
    ci.uniform_buf(0, "LiveMaterial", "P")
    ci.uniform_buf(1, "LiveShadow", "S")
    ci.sampler(0, 'FLOAT_2D', "inst_tex")
    ci.sampler(1, 'FLOAT_2D', "base_tex")
    ci.sampler(2, 'FLOAT_2D', "rough_tex")
    ci.sampler(3, 'FLOAT_2D', "normal_tex")
    ci.sampler(4, 'DEPTH_2D_ARRAY', "shadow_tex")
    ci.fragment_out(0, 'VEC4', "frag")
    ci.vertex_source(_VERT)
    ci.fragment_source(_FRAG)
    _shader = gpu.shader.create_from_info(ci)
    return _shader


def _dummy_tex(kind):
    t = _dummy.get(kind)
    if t is None:
        v = {"white": [1.0, 1.0, 1.0, 1.0], "normal": [0.5, 0.5, 1.0, 1.0]}[kind]
        t = gpu.types.GPUTexture((1, 1), format='RGBA16F', data=gpu.types.Buffer('FLOAT', 4, v))
        _dummy[kind] = t
    return t


def _image_tex(img):
    if img is None:
        return None
    t = _textures.get(img.name)
    if t is None:
        try:
            t = gpu.texture.from_image(img)
        except Exception:
            t = None
        _textures[img.name] = t
    return t


def matrix_texture(mats):
    """(n, 4, 4) world matrices -> a 4 x n RGBA32F texture of their columns (what the vertex shader reads)."""
    n = len(mats)
    cols = np.ascontiguousarray(np.transpose(np.asarray(mats, dtype=np.float32), (0, 2, 1))).reshape(-1)
    return gpu.types.GPUTexture((4, max(1, n)), format='RGBA32F',
                                data=gpu.types.Buffer('FLOAT', cols.size, cols.tolist()))


# ---------------------------------------------------------------- materials

def _upstream_image(sock, depth=0):
    """The image texture feeding a socket (through a few nodes), or None."""
    if sock is None or not sock.is_linked or depth > 6:
        return None, None
    node = sock.links[0].from_node
    if node.type == 'TEX_IMAGE':
        return node.image, node
    for inp in node.inputs:
        img, n = _upstream_image(inp, depth + 1)
        if img is not None:
            return img, n
    return None, None


def _mapping_of(img_node):
    if img_node is None:
        return (1.0, 1.0, 0.0, 0.0)
    v = img_node.inputs.get("Vector")
    if v is None or not v.is_linked:
        return (1.0, 1.0, 0.0, 0.0)
    m = v.links[0].from_node
    if m.type != 'MAPPING':
        return (1.0, 1.0, 0.0, 0.0)
    try:
        s = m.inputs["Scale"].default_value
        loc = m.inputs["Location"].default_value
        return (float(s[0]), float(s[1]), float(loc[0]), float(loc[1]))
    except (KeyError, IndexError):
        return (1.0, 1.0, 0.0, 0.0)


def material_info(mat):
    """What the shader needs from a material: values, textures, UV mapping."""
    info = {"base": (0.8, 0.8, 0.8, 1.0), "rough": 0.5, "metal": 0.0, "nstrength": 0.0,
            "base_tex": None, "rough_tex": None, "normal_tex": None, "uv": (1.0, 1.0, 0.0, 0.0),
            "cull": bool(getattr(mat, "use_backface_culling", False))}
    if mat is None:
        return info
    info["base"] = tuple(mat.diffuse_color)
    info["rough"] = float(getattr(mat, "roughness", 0.5))
    nt = mat.node_tree if getattr(mat, "use_nodes", False) else None
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None) if nt else None
    if bsdf is None:
        return info
    bc = bsdf.inputs["Base Color"]
    img, node = _upstream_image(bc)
    if img is not None:
        info["base_tex"], info["base"], info["uv"] = img, (1.0, 1.0, 1.0, 1.0), _mapping_of(node)
    else:
        info["base"] = tuple(bc.default_value)
    r = bsdf.inputs["Roughness"]
    img, node = _upstream_image(r)
    if img is not None:
        info["rough_tex"] = img
    else:
        info["rough"] = float(r.default_value)
    info["metal"] = float(bsdf.inputs["Metallic"].default_value) if not bsdf.inputs["Metallic"].is_linked else 0.0
    n = bsdf.inputs.get("Normal")
    if n is not None and n.is_linked and n.links[0].from_node.type == 'NORMAL_MAP':
        nm = n.links[0].from_node
        img, node = _upstream_image(nm.inputs["Color"])
        if img is not None:
            info["normal_tex"] = img
            info["nstrength"] = float(nm.inputs["Strength"].default_value)
    return info


def _scene_lighting(scene):
    """(ambient rgb, [(kind, vec, rgb)], shadow) from the scene's lights and world background; shadow is
    (index, direction to the light) of the first sun whose Shadow toggle is on, or None."""
    amb = (0.05, 0.05, 0.05)
    w = scene.world
    if w is not None:
        if w.use_nodes and w.node_tree:
            bg = next((n for n in w.node_tree.nodes if n.type == 'BACKGROUND'), None)
            if bg is not None:
                c, s = bg.inputs[0].default_value, bg.inputs[1].default_value
                amb = (c[0] * s, c[1] * s, c[2] * s)
        else:
            amb = tuple(w.color)
    lights = []
    shadow = None
    for ob in scene.objects:
        if ob.type != 'LIGHT' or len(lights) >= MAX_LIGHTS:
            continue
        try:
            if not ob.visible_get():
                continue
        except RuntimeError:
            continue
        L = ob.data
        col = tuple(c * L.energy for c in L.color)
        mw = ob.matrix_world
        if L.type == 'SUN':
            d = (mw.to_3x3() @ __import__("mathutils").Vector((0, 0, 1))).normalized()
            if shadow is None and L.use_shadow:
                shadow = (len(lights), tuple(d))
            lights.append((1.0, tuple(d), col))
        else:
            k = 1.0 / (4.0 * math.pi * math.pi)   # W -> radiance-ish, like Blender's point lights
            lights.append((2.0, tuple(mw.translation), tuple(c * k for c in col)))
    if not lights:   # no lights: a soft key light so shapes read
        lights.append((1.0, (0.4, -0.3, 0.85), (2.5, 2.5, 2.5)))
    return amb, lights, shadow


def _material_ubo(info, scene):
    amb, lights, _shadow = _scene_lighting(scene)
    vs = scene.view_settings
    tonemap = 1.0 if vs.view_transform in ('AgX', 'Filmic', 'Khronos PBR Neutral') else 0.0
    data = []
    data += list(info["base"][:4]) + [0.0] * (4 - len(info["base"][:4]))
    data += [info["rough"], info["metal"], info["nstrength"], 2.0 ** vs.exposure]
    data += [1.0 if info["base_tex"] is not None else 0.0, 1.0 if info["rough_tex"] is not None else 0.0,
             1.0 if info["normal_tex"] is not None else 0.0, tonemap]
    data += list(info["uv"])
    data += list(amb) + [0.0]
    vecs, cols = [], []
    for i in range(MAX_LIGHTS):
        if i < len(lights):
            kind, v, c = lights[i]
            vecs += list(v) + [kind]
            cols += list(c) + [0.0]
        else:
            vecs += [0.0, 0.0, 0.0, 0.0]
            cols += [0.0, 0.0, 0.0, 0.0]
    data += vecs + cols
    return gpu.types.GPUUniformBuf(gpu.types.Buffer('FLOAT', len(data), data))


# ---------------------------------------------------------------- assets and cuts

class GpuAsset:
    """Every vertex of one .vgeo on the GPU, plus its materials."""

    def __init__(self, path):
        self.path = path
        self.asset = native.Asset(path)
        pos, nrm, uv, _vmat = self.asset.vertex_arrays()
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt.attr_add(id="nrm", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt.attr_add(id="uv", comp_type='F32', len=2, fetch_mode='FLOAT')
        self.vbo = gpu.types.GPUVertBuf(fmt, len(pos))
        self.vbo.attr_fill("pos", pos)
        self.vbo.attr_fill("nrm", nrm)
        self.vbo.attr_fill("uv", uv if uv is not None else np.zeros((len(pos), 2), np.float32))
        self.vertex_count = len(pos)
        self.material_names = list(self.asset.material_names) or [""]
        self.chunk_count = self.asset.chunk_count
        self.ubos = {}       # material index -> (signature, ubo, info)

    def batch(self, indices):
        ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=indices.reshape(-1, 3))
        return gpu.types.GPUBatch(type='TRIS', buf=self.vbo, elem=ibo)

    def material(self, m, scene, materials, light_sig=None):
        mat = materials[m] if m < len(materials) else None
        sig = (mat.name if mat else None, light_sig if light_sig is not None else _light_sig(scene))
        cur = self.ubos.get(m)
        if cur is None or cur[0] != sig:
            info = material_info(mat)
            cur = (sig, _material_ubo(info, scene), info)
            self.ubos[m] = cur
        return cur[1], cur[2]

    def close(self):
        self.asset.close()


def _light_sig(scene):
    parts = [scene.view_settings.view_transform, round(scene.view_settings.exposure, 3)]
    for ob in scene.objects:
        if ob.type == 'LIGHT':
            try:
                vis = ob.visible_get()
            except RuntimeError:
                vis = False
            parts.append((ob.name, vis, round(ob.data.energy, 4), tuple(round(c, 4) for c in ob.data.color),
                          tuple(round(x, 4) for row in ob.matrix_world for x in row), ob.data.type,
                          bool(getattr(ob.data, "use_shadow", False))))
    w = scene.world
    if w is not None and w.use_nodes and w.node_tree:
        bg = next((n for n in w.node_tree.nodes if n.type == 'BACKGROUND'), None)
        if bg is not None:
            parts.append(tuple(round(c, 4) for c in bg.inputs[0].default_value))
            parts.append(round(bg.inputs[1].default_value, 4))
    return hash(tuple(parts))


def gpu_asset(path):
    ga = _assets.get(path)
    if ga is None:
        ga = GpuAsset(path)
        _assets[path] = ga
    return ga


class Cut:
    """A view-dependent cut drawn from index buffers: one per group of chunks and material,
    double-buffered so the shown cut is always whole."""

    def __init__(self, ga):
        self.ga = ga
        self.handle = native.Asset(ga.path)        # its own selection state
        n = ga.chunk_count
        self.groups = (n + GROUP_CHUNKS - 1) // GROUP_CHUNKS
        self.shown = {}            # group -> [(material, batch, tris)]
        self.shown_sigs = np.zeros(n, np.uint64)
        self.pending = None        # (sigs, todo groups list, built dict)
        self.key = None
        self.triangles = 0
        self.ready = False         # something has landed
        self.version = 0           # bumped whenever a new cut swaps in

    def due(self):
        return self.pending is None and (not self.ready or
                                         time.perf_counter() - getattr(self, "_t", 0.0) >= getattr(self, "interval", RESELECT))

    def want(self, views, key):
        """Select for these views (skipped while a rebuild is in flight, when nothing moved, or within
        RESELECT of the last selection: flying, a cut refreshes ten times a second, not every tick)."""
        if self.pending is not None or key == self.key:
            return
        now = time.perf_counter()
        if self.ready and now - getattr(self, "_t", 0.0) < getattr(self, "interval", RESELECT):
            return
        self._t = now
        self.handle.select(views)
        sigs = self.handle.sigs.copy()
        changed = np.nonzero(sigs != self.shown_sigs)[0]
        self.key = key
        if not len(changed) and self.ready:
            return
        todo = sorted({int(c) // GROUP_CHUNKS for c in changed}) if self.ready else list(range(self.groups))
        self.pending = (sigs, todo, {})

    def step(self, deadline):
        """Build pending groups until the deadline; swap them in together when all are built.
        Returns True while work remains."""
        if self.pending is None:
            return False
        sigs, todo, built = self.pending
        nm = len(self.ga.material_names)
        while todo and time.perf_counter() < deadline:
            g = todo.pop(0)
            idx, offs = self.handle.range_indices(g * GROUP_CHUNKS, GROUP_CHUNKS, nm)
            entry = []
            for m in range(nm):
                if offs[m + 1] > offs[m]:
                    arr = idx[offs[m]:offs[m + 1]]
                    entry.append((m, self.ga.batch(arr), len(arr) // 3))
            built[g] = entry
        if todo:
            return True
        self.shown.update(built)
        self.shown_sigs = sigs
        self.pending = None
        self.ready = True
        self.version += 1
        _state["landed"] += 1
        self.triangles = sum(t for e in self.shown.values() for _m, _b, t in e)
        return False

    def reset(self):
        self.shown = {}
        self.shown_sigs[:] = 0
        self.pending = None
        self.key = None
        self.ready = False
        self.triangles = 0

    def close(self):
        self.handle.close()


LEVEL_MAX_TRIS = 1_000_000   # finer shared levels are never built: copies that close get their own cut


class LevelSet:
    """An instancer's error levels as shared index buffers (per material), built lazily, coarsest first and
    resumably across ticks (a level's chunks are gathered a few at a time, so no tick stalls)."""

    def __init__(self, ga, errors, tris):
        self.ga = ga
        self.errors = list(errors)
        self.tris = list(tris)
        self.handle = native.Asset(ga.path)
        self.batches = {}          # level -> [(material, batch, tris)]
        self.queue = []
        self.job = None            # (level, next chunk, parts)
        self.version = 0           # bumped whenever a level lands
        self.finest = next((i for i, t in enumerate(self.tris) if t <= LEVEL_MAX_TRIS), len(self.tris) - 1)

    def clamp(self, level):
        return max(int(level), self.finest)

    def available(self, level):
        """The level to draw for a wanted level: itself if built, else the nearest coarser one built (or
        the nearest finer one); queues the wanted level."""
        level = self.clamp(level)
        if level in self.batches:
            return level
        if level not in self.queue and (self.job is None or self.job[0] != level):
            self.queue.append(level)
            self.queue.sort(key=lambda lv: self.tris[lv])      # cheapest first
        for lv in range(level + 1, len(self.errors)):
            if lv in self.batches:
                return lv
        for lv in range(level - 1, -1, -1):
            if lv in self.batches:
                return lv
        return None

    def step(self, deadline):
        nm = len(self.ga.material_names)
        while time.perf_counter() < deadline:
            if self.job is None:
                if not self.queue:
                    return False
                lv = self.queue.pop(0)
                if lv in self.batches:
                    continue
                if lv == 0 or self.errors[lv] <= 0.0:
                    self.handle.select_level(0)
                else:
                    self.handle.select([native.make_view((0.0, 0.0, 0.0), 1.0, 0.01, float(self.errors[lv]),
                                                         ortho=True, ortho_height=1.0)])
                self.job = [lv, 0, [[] for _ in range(nm)]]
            lv, c, parts = self.job
            while c < self.ga.chunk_count and time.perf_counter() < deadline:
                idx, offs = self.handle.range_indices(c, GROUP_CHUNKS, nm)
                for m in range(nm):
                    if offs[m + 1] > offs[m]:
                        parts[m].append(idx[offs[m]:offs[m + 1]])
                c += GROUP_CHUNKS
            self.job[1] = c
            if c < self.ga.chunk_count:
                return True
            self.batches[lv] = [(m, self.ga.batch(np.concatenate(parts[m])), sum(len(p) for p in parts[m]) // 3)
                                for m in range(nm) if parts[m]]
            self.job = None
            self.version += 1
            _state["landed"] += 1
        return bool(self.queue) or self.job is not None

    def close(self):
        self.handle.close()


# ---------------------------------------------------------------- per-frame state built in the tick

class Drawable:
    """What the draw handler draws: (gpu asset, list of (matrix texture, rows, [(material, batch, tris)]))."""
    __slots__ = ("ga", "materials", "items", "kind")

    def __init__(self, ga, materials, kind="proxy"):
        self.ga = ga
        self.materials = materials
        self.items = []
        self.kind = kind


_frame = {"drawables": [], "mat_tex": {}}


def _placement_matrices(inst):
    """World matrices of every placement, vectorised (points, Euler XYZ rotation, uniform scale)."""
    from . import instances
    me = inst.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    rot = np.zeros((n, 3), np.float32)
    a = me.attributes.get(instances.ROT_ATTR)
    if a is not None and a.domain == 'POINT':
        r = np.empty(n * 3, np.float32)
        a.data.foreach_get("vector", r)
        rot = r.reshape(-1, 3)
    sc = np.ones(n, np.float32)
    a = me.attributes.get(instances.SCALE_ATTR)
    if a is not None and a.domain == 'POINT' and a.data_type == 'FLOAT':
        a.data.foreach_get("value", sc)
    cx, cy, cz = np.cos(rot[:, 0]), np.cos(rot[:, 1]), np.cos(rot[:, 2])
    sx, sy, sz = np.sin(rot[:, 0]), np.sin(rot[:, 1]), np.sin(rot[:, 2])
    R = np.empty((n, 3, 3), np.float32)        # Rz @ Ry @ Rx (Blender's XYZ Euler)
    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = sx * sy * cz - cx * sz
    R[:, 0, 2] = cx * sy * cz + sx * sz
    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = sx * sy * sz + cx * cz
    R[:, 1, 2] = cx * sy * sz - sx * cz
    R[:, 2, 0] = -sy
    R[:, 2, 1] = sx * cy
    R[:, 2, 2] = cx * cy
    M = np.zeros((n, 4, 4), np.float32)
    M[:, :3, :3] = R * sc[:, None, None]
    M[:, :3, 3] = co
    M[:, 3, 3] = 1.0
    W = np.array(inst.matrix_world, dtype=np.float32)
    return np.einsum("ij,njk->nik", W, M)


_mat_cache = {}


def _cached_matrices(inst):
    """Placement matrices, recomputed when the instancer moves or its points change (checked by count and
    a cheap sample, plus at most once a second)."""
    uid = inst.vgeo_inst.uid
    key = (tuple(round(x, 6) for row in inst.matrix_world for x in row), len(inst.data.vertices))
    cur = _mat_cache.get(uid)
    now = time.perf_counter()
    if cur is not None and cur[0] == key and now - cur[2] < 1.0:
        return cur[1]
    m = _placement_matrices(inst)
    _mat_cache[uid] = (key, m, now)
    return m


class InstState:
    def __init__(self):
        self.levels = None         # chosen per placement (with hysteresis)
        self.near = {}             # slot k -> placement index
        self.key = None
        self.tex = None
        self.rows = []             # (level, row start, count)
        self.matrices = None
        self.mat_sig = None
        self.world_obj = None
        self.world_hash = None


_inst_state = {}


def _choose_instances(inst, views, ga):
    """Levels per placement (refine at once, coarsen when two levels too fine), frustum culling, and the
    nearest heavy placements for own cuts."""
    from . import instances
    st = _inst_state.setdefault(inst.vgeo_inst.uid, InstState())
    vi = inst.vgeo_inst
    levels, budget = instances._choose(inst, views, vi.pixel_error * LEVEL_ERROR_SCALE, "COARSEN", 8.0)
    if st.levels is not None and len(st.levels) == len(levels):
        keep = (st.levels <= levels) & (st.levels >= levels - 1)
        levels = np.where(keep, st.levels, levels).astype(np.int32)
    st.levels = levels
    _errors, tris = instances._level_tables(inst)
    k = max(int(vi.stream_slots), NEAR_COPIES)
    heavy = np.nonzero(np.asarray(tris)[levels] >= NEAR_MIN_TRIS)[0]
    ranked = [int(i) for i in heavy[np.argsort(budget[heavy], kind="stable")]]
    # hysteresis: a placement that has a cut keeps it while it stays among the nearest 1.5 k
    keep_zone = set(ranked[:int(k * 1.5)])
    near = [i for i in st.near.values() if i in keep_zone][:k]
    for i in ranked:
        if len(near) >= k:
            break
        if i not in near:
            near.append(i)
    st.near = {}
    return st, levels, near


def _visible_mask(world_m, radius, views):
    """Placements whose bounding sphere touches any view's frustum."""
    from . import stream
    pts = world_m[:, :3, 3]
    vis = np.zeros(len(pts), bool)
    for view, window, _h, _p, _c in views:
        planes = np.array(stream._frustum_planes(window @ view), dtype=np.float64)
        vis |= np.all(pts @ planes[:, :3].T + planes[:, 3] >= -radius[:, None], axis=1)
    return vis


def tick(views, budget=TICK_BUDGET):
    """Choose cuts and levels for these views and build index buffers within the budget. Called from the
    live loop. Returns True while work remains."""
    from . import instances, stream
    vkey = (tuple(round(x, 5) for v in views for m in (v[0], v[1]) for row in m for x in row),
            _scene_key(bpy.context.scene))
    if vkey == _frame.get("vkey") and not _frame.get("busy", True):
        return False
    deadline = time.perf_counter() + budget
    drawables = []
    busy = False
    live_cuts = set()
    landed0 = _state["landed"]
    casters, sig_parts, lows, highs = [], [], [], []   # for the shadow pass
    scene = bpy.context.scene
    vl = bpy.context.view_layer
    for obj in stream.proxies(scene):
        if obj.vgeo.slot_of:
            continue
        try:
            if vl is not None and not obj.visible_get(view_layer=vl):
                continue
        except RuntimeError:
            continue
        try:
            ga = gpu_asset(stream.asset_path(obj))
        except Exception as e:
            print("VGEO live:", e)
            continue
        cut = _cuts.get(obj.vgeo.uid)
        if cut is None or cut.ga is not ga:
            cut = Cut(ga)
            _cuts[obj.vgeo.uid] = cut
        live_cuts.add(obj.vgeo.uid)
        v = obj.vgeo
        if not v.freeze:
            lv = stream.local_views(obj, views, v.pixel_error, v.offscreen, v.offscreen_scale)
            cut.want(lv, stream._key(obj, views, v.pixel_error, v.offscreen, v.offscreen_scale))
        busy |= cut.step(max(deadline, time.perf_counter() + MIN_SLICE))
        d = Drawable(ga, list(obj.data.materials))
        tex = _one_matrix(obj.vgeo.uid, obj.matrix_world)
        entries = [e for g in cut.shown.values() for e in g]
        d.items.append((tex, 0, 1, entries))
        drawables.append(d)
        corners = _box_corners(ga.asset.info, obj.matrix_world)
        lows.append(corners.min(0))
        highs.append(corners.max(0))
        casters.append({"kind": "proxy", "tex": tex, "entries": entries, "corners": corners})
        sig_parts.append((obj.vgeo.uid, cut.version, id(tex)))
    for inst in instances.instancers(scene):
        try:
            if vl is not None and not inst.visible_get(view_layer=vl):
                continue
        except RuntimeError:
            continue
        src = inst.vgeo_inst.source if hasattr(inst.vgeo_inst, "source") else None
        if src is None or not src.vgeo.uid:
            continue
        try:
            ga = gpu_asset(stream.asset_path(src))
        except Exception as e:
            print("VGEO live:", e)
            continue
        errors, tris = instances._level_tables(inst)
        ls = _levels.get(inst.vgeo_inst.uid)
        if ls is None or ls.ga is not ga or len(ls.errors) != len(errors):
            ls = LevelSet(ga, errors, tris)
            _levels[inst.vgeo_inst.uid] = ls
        if inst.vgeo_inst.freeze and inst.vgeo_inst.uid in _inst_state and _inst_state[inst.vgeo_inst.uid].levels is not None:
            st = _inst_state[inst.vgeo_inst.uid]
            levels, near = st.levels, list(st.near.values())
        else:
            st, levels, near = _choose_instances(inst, views, ga)
        world = _cached_matrices(inst)
        scale = np.linalg.norm(world[:, :3, 0], axis=1) if len(world) else np.zeros(0)
        radius = float(inst.get("vgeo_radius", 1.0)) * scale
        vis = _visible_mask(world, radius, views)
        if len(world):
            if st.world_obj is not world:
                st.world_obj, st.world_hash = world, hash(world.tobytes())
            pts = world[:, :3, 3]
            lows.append((pts - radius[:, None]).min(0))
            highs.append((pts + radius[:, None]).max(0))
            casters.append({"kind": "inst", "ls": ls, "world": world, "radius": radius, "errors": errors,
                            "tris": tris, "max_scale": float(scale.max())})
            sig_parts.append((inst.vgeo_inst.uid, ls.version, st.world_hash))
        # shared levels first (cheap and needed by most copies), then the near copies' own cuts
        busy |= ls.step(max(min(time.perf_counter() + budget * 0.5, deadline), time.perf_counter() + MIN_SLICE))
        # near copies: their own cut, drawn once it has landed
        taken = set()
        d_near = Drawable(ga, list(src.data.materials), "near")
        rr = _frame.get("rr", 0)
        _frame["rr"] = rr + 1
        for k, i in enumerate(near):
            key = (inst.vgeo_inst.uid, "p", i)        # a cut belongs to its placement: no restarts on reorder
            cut = _cuts.get(key)
            if cut is None or cut.ga is not ga:
                cut = Cut(ga)
                _cuts[key] = cut
            live_cuts.add(key)
            st.near[k] = i
            from mathutils import Matrix
            m = Matrix(world[i].tolist())
            cut.interval = RESELECT * (1 + k // 8)        # farther copies refresh less often
            if cut.due():                                  # views are only worked out when it will select
                lv = _local_views_for(m, views, inst.vgeo_inst.pixel_error)
                cut.want(lv, (i, vkey[0]))
            # one copy a tick is guaranteed a slice (round robin); the rest share what is left
            floor = time.perf_counter() + MIN_SLICE if (rr % max(1, len(near))) == k else 0.0
            busy |= cut.step(max(deadline, floor))
            if cut.ready:
                taken.add(i)
                d_near.items.append((_one_matrix(key, m), 0, 1, [e for g in cut.shown.values() for e in g]))
        for k in [k for k in st.near if k >= len(near)]:
            st.near.pop(k, None)
        # instanced levels for everything else in view
        order, rows = [], []
        start = 0
        for lv in sorted(set(int(x) for x in levels)):
            idx = [int(i) for i in np.nonzero((levels == lv) & vis)[0] if int(i) not in taken]
            if not idx:
                continue
            got = ls.available(lv)
            if got is None:
                continue
            order.extend(idx)
            rows.append((got, start, len(idx)))
            start += len(idx)
        busy |= ls.step(max(deadline, time.perf_counter() + MIN_SLICE / 2))
        sig = (tuple(order), world.tobytes() if len(order) else b"")
        if sig != st.mat_sig:
            st.tex = matrix_texture(world[order]) if order else None
            st.mat_sig = sig
        st.rows = rows
        d = Drawable(ga, list(src.data.materials), "levels")
        if st.tex is not None:
            for lvl, s0, cnt in rows:
                d.items.append((st.tex, s0, cnt, ls.batches.get(lvl, [])))
        drawables.append(d)
        drawables.append(d_near)
    for key in [k for k in _cuts if k not in live_cuts]:
        _cuts.pop(key).close()
    # a cut or level that landed after its drawable was put together is drawn by the next tick
    busy |= _state["landed"] != landed0
    _frame["drawables"] = drawables
    _frame["vkey"] = vkey
    _frame["busy"] = busy
    _frame["casters"] = casters
    _frame["bounds"] = (np.min(lows, 0), np.max(highs, 0)) if lows else None
    sig = hash(tuple(sig_parts))
    if _state["landed"] != landed0 or sig != _frame.get("shadow_sig"):
        _frame["shadow_sig"] = sig
        _tag_redraw()          # something new to show (a cut, a level, the shadow casters)
    return busy


def _tag_redraw():
    for win in bpy.context.window_manager.windows:
        if win.screen is None:
            continue
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _box_corners(info, mw):
    """World-space corners (8 x 3) of an asset's bounding box."""
    lo, hi = info["aabb_min"], info["aabb_max"]
    c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return (c @ np.array(mw, np.float64).T)[:, :3]


def _scene_key(scene):
    """What else changes what is drawn: proxies and instancers (transform, visibility, settings)."""
    from . import instances, stream
    parts = []
    for o in stream.proxies(scene) + instances.instancers(scene):
        try:
            vis = o.visible_get()
        except RuntimeError:
            vis = False
        parts.append((o.name, vis, tuple(round(x, 5) for row in o.matrix_world for x in row),
                      round(getattr(o.vgeo, "pixel_error", 0.0), 3), round(o.vgeo_inst.pixel_error, 3),
                      bool(o.vgeo.freeze), bool(o.vgeo_inst.freeze), len(o.data.vertices)))
    return hash(tuple(parts))


_one = {}


def _one_matrix(key, mw):
    m = np.array(mw, dtype=np.float32)
    cur = _one.get(key)
    if cur is not None and np.array_equal(cur[0], m):
        return cur[1]
    tex = matrix_texture(m[None])
    _one[key] = (m, tex)
    return tex


def _local_views(m, views, pixel_error):
    from . import stream

    class _O:   # stream.local_views wants an object with matrix_world
        pass
    o = _O()
    o.matrix_world = m
    return stream.local_views(o, views, pixel_error, "COARSEN", 8.0)


_local_views_for = _local_views


# ---------------------------------------------------------------- drawing

def _usable_depth(region):
    """With overlays hidden, Material Preview hands draw handlers a depth buffer of zeros (nothing would pass
    the depth test; CodeNodes met the same). Then start from an empty one; a real depth buffer is left alone."""
    try:
        fb = gpu.state.active_framebuffer_get()
        if fb.read_depth(1, max(0, region.height - 2), 1, 1).to_list()[0][0] == 0.0:
            fb.clear(depth=1.0)
    except Exception:
        pass


def _draw():
    ctx = bpy.context
    if not _state["active"]:
        return
    space = ctx.space_data
    if space is None or space.shading.type not in ('SOLID', 'MATERIAL'):
        return
    rv3d = ctx.region_data
    if rv3d is None:
        return
    t0 = time.perf_counter()
    sh = shader()
    scene = ctx.scene
    lsig = _frame.get("light_sig")
    if lsig is None or _frame.get("light_sig_t", 0.0) < time.perf_counter() - 0.25:
        lsig = _light_sig(scene)                 # lights rarely change: checked a few times a second
        if lsig != _frame.get("light_sig"):
            _frame["shadow_light"] = _scene_lighting(scene)[2]
        _frame["light_sig"], _frame["light_sig_t"] = lsig, time.perf_counter()
    shadow = None
    if getattr(scene, "vgeo_live_shadows", True):
        shadow = shadows.update(rv3d, space, _frame.get("shadow_light"), _frame)
    _state["shadowed"] = shadow is not None
    t_shadow = time.perf_counter()
    s_ubo, s_tex = shadow or shadows.none_bindings()
    if space.shading.type == 'MATERIAL':
        _usable_depth(ctx.region)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('NONE')
    sh.bind()
    sh.uniform_float("viewproj", rv3d.perspective_matrix)
    cam = rv3d.view_matrix.inverted().translation
    sh.uniform_float("cam_pos", (cam.x, cam.y, cam.z, 1.0))
    sh.uniform_block("S", s_ubo)
    sh.uniform_sampler("shadow_tex", s_tex)
    drawn = 0
    _state["by"] = {}
    bound = None
    for d in _frame["drawables"]:
        mats = d.materials
        per_mat = {}
        for tex, row0, count, entries in d.items:
            if not count:
                continue
            for m, batch, tris in entries:
                per_mat.setdefault(m, []).append((tex, row0, count, batch, tris))
        for m, draws in per_mat.items():
            ubo, info = d.ga.material(m, scene, mats, lsig)
            if bound is not ubo:
                sh.uniform_block("P", ubo)
                sh.uniform_sampler("base_tex", _image_tex(info["base_tex"]) or _dummy_tex("white"))
                sh.uniform_sampler("rough_tex", _image_tex(info["rough_tex"]) or _dummy_tex("white"))
                sh.uniform_sampler("normal_tex", _image_tex(info["normal_tex"]) or _dummy_tex("normal"))
                gpu.state.face_culling_set('BACK' if info["cull"] else 'NONE')
                bound = ubo
            last_tex = last_row = None
            for tex, row0, count, batch, tris in draws:
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
                drawn += tris * count
                _state["by"][d.kind] = _state["by"].get(d.kind, 0) + tris * count
    gpu.state.face_culling_set('NONE')
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set('NONE')
    _state["drawn"] = drawn
    ms = _state["frame_ms"]
    ms.append((time.perf_counter() - t0) * 1000.0)
    if len(ms) > 240:
        del ms[:120]
    sms = _state.setdefault("shadow_ms", [])
    sms.append((t_shadow - t0) * 1000.0)
    if len(sms) > 240:
        del sms[:120]


def drawn_triangles():
    return _state["drawn"]


def shadow_light_found(scene):
    """Is there a sun that casts Live Draw shadows (visible, Shadow on)?"""
    return _scene_lighting(scene)[2] is not None


def set_active(on):
    _state["active"] = bool(on)


def is_active():
    return _state["active"]


def free_all():
    for c in _cuts.values():
        c.close()
    _cuts.clear()
    for ls in _levels.values():
        ls.close()
    _levels.clear()
    for ga in _assets.values():
        ga.close()
    _assets.clear()
    _inst_state.clear()
    _one.clear()
    _textures.clear()
    shadows.free_all()
    _frame["drawables"] = []
    _frame["casters"] = []
    _frame["bounds"] = None


@bpy.app.handlers.persistent
def _on_depsgraph(_scene, depsgraph):
    """A light or the world changed: look at the lights again on the next redraw (otherwise every 0.25 s)."""
    for u in depsgraph.updates:
        i = u.id
        if isinstance(i, (bpy.types.Light, bpy.types.World)) or (isinstance(i, bpy.types.Object) and i.type == 'LIGHT'):
            _frame["light_sig_t"] = 0.0
            return


def register():
    if _state["handler"] is None and not bpy.app.background:
        _state["handler"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _state["handler"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handler"], 'WINDOW')
        _state["handler"] = None
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    free_all()
