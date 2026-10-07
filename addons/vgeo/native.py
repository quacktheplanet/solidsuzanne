"""ctypes binding for vgeo_stream (the C++ builder + runtime).

ctypes keeps the add-on independent of Blender's Python version: the same
DLL loads in 5.0 (Python 3.11) and 5.1 (Python 3.13).
"""

import ctypes
import os
import sys

import numpy as np

_lib = None
_lib_error = None

MAX_ERR = 1024


class BuildInput(ctypes.Structure):
    _fields_ = [
        ("tri_count", ctypes.c_uint32),
        ("positions", ctypes.c_void_p),
        ("normals", ctypes.c_void_p),
        ("uvs", ctypes.c_void_p),
        ("materials", ctypes.c_void_p),
        ("vertex_count", ctypes.c_uint32),
        ("indices", ctypes.c_void_p),
        ("material_count", ctypes.c_uint32),
        ("material_names", ctypes.POINTER(ctypes.c_char_p)),
        ("max_triangles", ctypes.c_uint32),
        ("target_chunks", ctypes.c_uint32),
        ("material_params", ctypes.c_void_p),
        # library version 3 (older libraries read only the fields above)
        ("extra_count", ctypes.c_uint32),
        ("extras", ctypes.c_void_p),
        ("extra_desc", ctypes.c_char_p),
    ]


class BuildStats(ctypes.Structure):
    _fields_ = [
        ("source_triangles", ctypes.c_uint32),
        ("vertices", ctypes.c_uint32),
        ("clusters", ctypes.c_uint32),
        ("groups", ctypes.c_uint32),
        ("chunks", ctypes.c_uint32),
        ("lod_levels", ctypes.c_uint32),
        ("coarsest_triangles", ctypes.c_uint32),
        ("file_bytes", ctypes.c_uint64),
        ("seconds", ctypes.c_double),
    ]


class Info(ctypes.Structure):
    _fields_ = [
        ("vertex_count", ctypes.c_uint32),
        ("cluster_count", ctypes.c_uint32),
        ("group_count", ctypes.c_uint32),
        ("chunk_count", ctypes.c_uint32),
        ("material_count", ctypes.c_uint32),
        ("lod_levels", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("source_triangles", ctypes.c_uint32),
        ("aabb_min", ctypes.c_float * 3),
        ("aabb_max", ctypes.c_float * 3),
        ("extra_count", ctypes.c_uint32),      # library version 3
    ]


class View(ctypes.Structure):
    _fields_ = [
        ("camera", ctypes.c_float * 3),
        ("proj", ctypes.c_float),
        ("znear", ctypes.c_float),
        ("threshold", ctypes.c_float),
        ("ortho", ctypes.c_int32),
        ("ortho_height", ctypes.c_float),
        ("frustum_mode", ctypes.c_int32),
        ("offscreen_scale", ctypes.c_float),
        ("planes", (ctypes.c_float * 4) * 6),
    ]


class CutStats(ctypes.Structure):
    _fields_ = [
        ("triangles", ctypes.c_uint64),
        ("clusters", ctypes.c_uint32),
        ("changed_chunks", ctypes.c_uint32),
    ]


class ChunkData(ctypes.Structure):
    _fields_ = [
        ("vertex_count", ctypes.c_uint32),
        ("tri_count", ctypes.c_uint32),
        ("positions", ctypes.POINTER(ctypes.c_float)),
        ("normals", ctypes.POINTER(ctypes.c_float)),
        ("uvs", ctypes.POINTER(ctypes.c_float)),
        ("corner_verts", ctypes.POINTER(ctypes.c_int32)),
        ("face_materials", ctypes.POINTER(ctypes.c_int32)),
        ("face_lod", ctypes.POINTER(ctypes.c_int32)),
        ("edge_count", ctypes.c_uint32),
        ("edge_verts", ctypes.POINTER(ctypes.c_int32)),
        ("corner_edges", ctypes.POINTER(ctypes.c_int32)),
        ("extra_count", ctypes.c_uint32),      # library version 3
        ("extras", ctypes.POINTER(ctypes.c_float)),
    ]


PROGRESS_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_float)


def _library_name():
    if sys.platform == "win32":
        return "vgeo_stream.dll"
    if sys.platform == "darwin":
        return "vgeo_stream.dylib"
    return "vgeo_stream.so"


def library_path():
    return os.path.join(os.path.dirname(__file__), "bin", _library_name())


def lib():
    """Load the library once; raises RuntimeError with a readable reason."""
    global _lib, _lib_error
    if _lib is not None:
        return _lib
    if _lib_error is not None:
        raise RuntimeError(_lib_error)
    path = library_path()
    try:
        L = ctypes.CDLL(path)
    except OSError as e:
        _lib_error = f"VGEO native library not loadable ({path}): {e}"
        raise RuntimeError(_lib_error)
    c = ctypes
    L.vgeo_version.restype = c.c_int
    L.vgeo_build.argtypes = [c.POINTER(BuildInput), c.c_char_p, PROGRESS_FN, c.c_void_p,
                             c.POINTER(BuildStats), c.c_char_p, c.c_int]
    L.vgeo_build.restype = c.c_int
    L.vgeo_open.argtypes = [c.c_char_p, c.c_char_p, c.c_int]
    L.vgeo_open.restype = c.c_void_p
    L.vgeo_close.argtypes = [c.c_void_p]
    L.vgeo_close.restype = None
    L.vgeo_get_info.argtypes = [c.c_void_p, c.POINTER(Info)]
    L.vgeo_material_name.argtypes = [c.c_void_p, c.c_uint32, c.c_char_p, c.c_int]
    L.vgeo_select.argtypes = [c.c_void_p, c.POINTER(View), c.c_int, c.c_void_p, c.POINTER(CutStats)]
    L.vgeo_select_level.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.POINTER(CutStats)]
    L.vgeo_extract.argtypes = [c.c_void_p, c.c_uint32, c.POINTER(ChunkData)]
    L.vgeo_export_web.argtypes = [c.c_void_p, c.c_char_p, c.POINTER(c.c_uint64), c.c_char_p, c.c_int]
    L.vgeo_export_web.restype = c.c_int
    L.vgeo_level_errors.argtypes = [c.c_void_p, c.c_void_p, c.c_int]
    L.vgeo_level_errors.restype = c.c_int
    if hasattr(L, "vgeo_prefetch"):  # library version 2+
        L.vgeo_prefetch.argtypes = [c.c_void_p, c.c_void_p, c.c_int]
        L.vgeo_prefetch.restype = c.c_int
        L.vgeo_memory.argtypes = [c.c_void_p, c.POINTER(c.c_uint64), c.POINTER(c.c_uint64)]
        L.vgeo_memory.restype = c.c_int
        L.vgeo_corrupt.argtypes = [c.c_void_p]
        L.vgeo_corrupt.restype = c.c_int
        L.vgeo_export_web_paged.argtypes = [c.c_void_p, c.c_char_p, c.c_uint32, c.POINTER(c.c_uint64),
                                            c.c_char_p, c.c_int]
        L.vgeo_export_web_paged.restype = c.c_int
    if hasattr(L, "vgeo_chunk_indices"):  # live drawing
        P = c.POINTER
        L.vgeo_vertex_arrays.argtypes = [c.c_void_p, P(P(c.c_float)), P(P(c.c_float)), P(P(c.c_float)),
                                         P(P(c.c_uint16)), P(c.c_uint32)]
        L.vgeo_vertex_arrays.restype = c.c_int
        L.vgeo_chunk_indices.argtypes = [c.c_void_p, c.c_uint32, P(P(c.c_uint32)), P(c.c_uint32),
                                         P(c.c_uint32), c.c_uint32]
        L.vgeo_chunk_indices.restype = c.c_int
        L.vgeo_range_indices.argtypes = [c.c_void_p, c.c_uint32, c.c_uint32, P(P(c.c_uint32)), P(c.c_uint32),
                                         P(c.c_uint32), c.c_uint32]
        L.vgeo_range_indices.restype = c.c_int
    if hasattr(L, "vgeo_extra_desc"):  # library version 3
        L.vgeo_extra_desc.argtypes = [c.c_void_p, c.c_char_p, c.c_int]
        L.vgeo_extra_desc.restype = c.c_int
    _lib = L
    return L


def supports_extras():
    """Library version 3+: more UV maps and colour/attribute layers travel with the geometry."""
    try:
        return hasattr(lib(), "vgeo_extra_desc")
    except RuntimeError:
        return False


def available():
    try:
        lib()
        return True
    except RuntimeError:
        return False


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p) if a is not None else None


def build(path, positions, normals=None, uvs=None, materials=None, material_names=(),
          max_triangles=128, target_chunks=0, progress=None, indices=None, material_params=None,
          extras=None, extra_desc=""):
    """Build a .vgeo.

    Without indices, positions/normals/uvs hold one row per triangle corner.
    With indices (tri_count*3 vertex indices) they hold one row per vertex.
    extras (same rows, any number of float columns) travel with the geometry into every cut; extra_desc
    (a string) is stored for the reader. Needs library version 3 (ignored by older ones).
    progress(stage, fraction) -> truthy to cancel. Runs without holding the
    GIL (ctypes releases it), so it can be called from a worker thread.
    """
    L = lib()
    positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
    rows = positions.shape[0]
    if indices is not None:
        indices = np.ascontiguousarray(indices, dtype=np.uint32).ravel()
        if indices.size == 0 or indices.size % 3:
            raise ValueError("indices must hold 3 entries per triangle")
        if int(indices.max()) >= rows:
            raise ValueError("index out of range")
        tri_count = indices.size // 3
    else:
        if rows == 0 or rows % 3:
            raise ValueError("positions must hold 3 corners per triangle")
        tri_count = rows // 3
    if normals is not None:
        normals = np.ascontiguousarray(normals, dtype=np.float32).reshape(rows, 3)
    if uvs is not None:
        uvs = np.ascontiguousarray(uvs, dtype=np.float32).reshape(rows, 2)
    if materials is not None:
        materials = np.ascontiguousarray(materials, dtype=np.uint16).reshape(tri_count)
    names = [str(n).encode("utf-8") for n in material_names]
    name_arr = (ctypes.c_char_p * max(1, len(names)))(*names)

    inp = BuildInput()
    inp.tri_count = tri_count
    inp.positions = _ptr(positions)
    inp.normals = _ptr(normals)
    inp.uvs = _ptr(uvs)
    inp.materials = _ptr(materials)
    inp.vertex_count = rows if indices is not None else 0
    inp.indices = _ptr(indices)
    inp.material_count = len(names)
    inp.material_names = name_arr
    inp.max_triangles = max_triangles
    inp.target_chunks = target_chunks
    if material_params is not None and len(names):
        material_params = np.ascontiguousarray(material_params, dtype=np.float32).reshape(len(names), 4)
    else:
        material_params = None
    inp.material_params = _ptr(material_params)
    if supports_extras():
        if extras is not None and np.size(extras):
            extras = np.ascontiguousarray(extras, dtype=np.float32).reshape(rows, -1)
            inp.extra_count = extras.shape[1]
            inp.extras = _ptr(extras)
        inp.extra_desc = (extra_desc or "").encode("utf-8")

    def _cb(_user, stage, frac):
        try:
            return 1 if (progress and progress(stage, frac)) else 0
        except Exception:
            return 0

    cb = PROGRESS_FN(_cb)
    stats = BuildStats()
    err = ctypes.create_string_buffer(MAX_ERR)
    rc = L.vgeo_build(ctypes.byref(inp), path.encode("utf-8"), cb, None, ctypes.byref(stats), err, MAX_ERR)
    if rc == 2:
        raise InterruptedError("build cancelled")
    if rc != 0:
        raise RuntimeError(err.value.decode("utf-8", "replace") or "vgeo_build failed")
    return {f: getattr(stats, f) for f, _t in BuildStats._fields_}


class Asset:
    """An opened .vgeo file plus its current cut."""

    def __init__(self, path):
        L = lib()
        err = ctypes.create_string_buffer(MAX_ERR)
        h = L.vgeo_open(path.encode("utf-8"), err, MAX_ERR)
        if not h:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        self._h = h
        self.path = path
        info = Info()
        L.vgeo_get_info(h, ctypes.byref(info))
        self.info = {f: (list(getattr(info, f)) if f.startswith("aabb") else getattr(info, f))
                     for f, _t in Info._fields_}
        self.chunk_count = info.chunk_count
        self.has_uvs = bool(info.flags & 2)
        self.extra_count = getattr(info, "extra_count", 0) if hasattr(L, "vgeo_extra_desc") else 0
        self.extra_desc = {}
        if hasattr(L, "vgeo_extra_desc"):
            n = L.vgeo_extra_desc(h, None, 0)
            dbuf = ctypes.create_string_buffer(n + 1)
            L.vgeo_extra_desc(h, dbuf, n + 1)
            try:
                import json
                self.extra_desc = json.loads(dbuf.value.decode("utf-8")) if n else {}
            except ValueError:
                self.extra_desc = {}
        buf = ctypes.create_string_buffer(512)
        self.material_names = []
        for i in range(info.material_count):
            L.vgeo_material_name(h, i, buf, 512)
            self.material_names.append(buf.value.decode("utf-8", "replace"))
        self.sigs = np.zeros(self.chunk_count, dtype=np.uint64)
        self.last = CutStats()

    def close(self):
        if self._h:
            lib().vgeo_close(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def select(self, views):
        arr = (View * len(views))(*views)
        stats = CutStats()
        if lib().vgeo_select(self._h, arr, len(views), _ptr(self.sigs), ctypes.byref(stats)) != 0:
            raise RuntimeError("vgeo_select failed")
        self.last = stats
        return self.sigs

    def prefetch(self, chunks):
        """Ask the OS to start reading what these chunks' current selection needs (a hint)."""
        L = lib()
        if not hasattr(L, "vgeo_prefetch") or len(chunks) == 0:
            return
        arr = np.ascontiguousarray(chunks, dtype=np.uint32)
        L.vgeo_prefetch(self._h, _ptr(arr), len(arr))

    def memory(self):
        """(bytes mapped from the file, bytes the handle allocated itself)."""
        L = lib()
        if not hasattr(L, "vgeo_memory"):
            return (0, os.path.getsize(self.path))
        mapped, heap = ctypes.c_uint64(0), ctypes.c_uint64(0)
        L.vgeo_memory(self._h, ctypes.byref(mapped), ctypes.byref(heap))
        return (mapped.value, heap.value)

    @property
    def corrupt(self):
        """True once a cluster with bad indices was found (it is skipped)."""
        L = lib()
        return bool(hasattr(L, "vgeo_corrupt") and L.vgeo_corrupt(self._h))

    def select_level(self, depth):
        stats = CutStats()
        lib().vgeo_select_level(self._h, int(depth), _ptr(self.sigs), ctypes.byref(stats))
        self.last = stats
        return self.sigs

    def level_errors(self):
        """Geometric error (object units) of showing the asset at each uniform level."""
        out = np.zeros(max(1, self.info["lod_levels"]), dtype=np.float32)
        n = lib().vgeo_level_errors(self._h, _ptr(out), len(out))
        return out[:n]

    def level_mesh_data(self, depth):
        """The whole asset at one uniform level, as one chunk-style dict (for instancing)."""
        self.select_level(depth)
        return self._whole_cut()

    def error_cut_mesh_data(self, max_error):
        """The cheapest crack-free cut of the whole asset whose error stays under max_error (asset units),
        as one chunk-style dict. Unlike a uniform DAG depth, it keeps coarse clusters wherever they are
        good enough, so the triangles go where the shape needs them."""
        self.select([make_view((0.0, 0.0, 0.0), 1.0, 0.01, float(max_error), ortho=True, ortho_height=1.0)])
        return self._whole_cut()

    def _whole_cut(self):
        parts = [d for d in (self.extract(c) for c in range(self.chunk_count)) if d]
        if not parts:
            return None
        out = {k: [] for k in ("positions", "normals", "uvs", "corner_verts", "face_materials", "face_lod",
                               "edge_verts", "corner_edges", "extras")}
        v_off = e_off = 0
        for d in parts:
            out["positions"].append(d["positions"])
            out["normals"].append(d["normals"])
            if d["uvs"] is not None:
                out["uvs"].append(d["uvs"])
            if d.get("extras") is not None:
                out["extras"].append(d["extras"])
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
        if len(out["extras"]) != len(parts):
            res["extras"] = None
        res["desc"] = self.extra_desc
        res["vertex_count"] = v_off
        res["edge_count"] = e_off
        res["tri_count"] = len(res["face_materials"])
        return res

    def export_web(self, path, paged=False, page_vertices=0):
        """Write the compact web variant (.vgeow); returns its size in bytes.

        paged: version 2, split into pages ordered coarse to fine, which the viewer loads with
        HTTP range requests as the view needs them (the viewer from this version reads both)."""
        L = lib()
        size = ctypes.c_uint64(0)
        err = ctypes.create_string_buffer(MAX_ERR)
        if paged and hasattr(L, "vgeo_export_web_paged"):
            rc = L.vgeo_export_web_paged(self._h, path.encode("utf-8"), int(page_vertices), ctypes.byref(size),
                                         err, MAX_ERR)
        else:
            rc = L.vgeo_export_web(self._h, path.encode("utf-8"), ctypes.byref(size), err, MAX_ERR)
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace") or "vgeo_export_web failed")
        return size.value

    def vertex_arrays(self):
        """Copies of the whole asset's vertex arrays (positions, normals, uvs or None, vmat), for uploading
        every vertex to the GPU once (live drawing)."""
        L = lib()
        c = ctypes
        pp, pn, pu = c.POINTER(c.c_float)(), c.POINTER(c.c_float)(), c.POINTER(c.c_float)()
        pm = c.POINTER(c.c_uint16)()
        n = c.c_uint32(0)
        if L.vgeo_vertex_arrays(self._h, c.byref(pp), c.byref(pn), c.byref(pu), c.byref(pm), c.byref(n)) != 0:
            raise RuntimeError("vgeo_vertex_arrays failed")
        nv = n.value

        def grab(ptr, count, dtype):
            out = np.empty(count, dtype=dtype)
            ctypes.memmove(out.ctypes.data, ctypes.cast(ptr, ctypes.c_void_p).value, out.nbytes)
            return out
        return (grab(pp, nv * 3, np.float32).reshape(nv, 3), grab(pn, nv * 3, np.float32).reshape(nv, 3),
                grab(pu, nv * 2, np.float32).reshape(nv, 2) if pu else None, grab(pm, nv, np.uint16))

    def range_indices(self, first, count, material_count):
        """chunk_indices for chunks [first, first + count) at once."""
        c = ctypes
        ptr = c.POINTER(c.c_uint32)()
        n = c.c_uint32(0)
        offs = (c.c_uint32 * (material_count + 1))()
        if lib().vgeo_range_indices(self._h, first, count, c.byref(ptr), c.byref(n), offs, material_count) != 0:
            raise RuntimeError("vgeo_range_indices failed")
        out = np.empty(n.value, np.uint32)
        if n.value:
            ctypes.memmove(out.ctypes.data, ctypes.cast(ptr, ctypes.c_void_p).value, out.nbytes)
        return out, np.array(offs[:], dtype=np.int64)

    def chunk_indices(self, chunk, material_count):
        """The chunk's selected triangles as global vertex indices (uint32, copied), grouped by material,
        and material_count + 1 offsets into them."""
        c = ctypes
        ptr = c.POINTER(c.c_uint32)()
        n = c.c_uint32(0)
        offs = (c.c_uint32 * (material_count + 1))()
        if lib().vgeo_chunk_indices(self._h, chunk, c.byref(ptr), c.byref(n), offs, material_count) != 0:
            raise RuntimeError("vgeo_chunk_indices failed")
        out = np.empty(n.value, np.uint32)
        if n.value:
            ctypes.memmove(out.ctypes.data, ctypes.cast(ptr, ctypes.c_void_p).value, out.nbytes)
        return out, np.array(offs[:], dtype=np.int64)

    def extract(self, chunk):
        """Return numpy copies of one chunk's selected geometry (None if empty)."""
        d = ChunkData()
        if lib().vgeo_extract(self._h, chunk, ctypes.byref(d)) != 0:
            raise RuntimeError("vgeo_extract failed")
        nv, nt = d.vertex_count, d.tri_count
        if nt == 0:
            return None

        def grab(ptr, n, dtype):
            # memmove into a fresh array: much cheaper than np.ctypeslib.as_array per call
            out = np.empty(n, dtype=dtype)
            ctypes.memmove(out.ctypes.data, ctypes.cast(ptr, ctypes.c_void_p).value, out.nbytes)
            return out

        return {
            "positions": grab(d.positions, nv * 3, np.float32),
            "normals": grab(d.normals, nv * 3, np.float32),
            "uvs": grab(d.uvs, nv * 2, np.float32) if d.uvs else None,
            "extras": (grab(d.extras, nv * self.extra_count, np.float32).reshape(nv, self.extra_count)
                       if self.extra_count and d.extras else None),
            "desc": self.extra_desc,
            "corner_verts": grab(d.corner_verts, nt * 3, np.int32),
            "face_materials": grab(d.face_materials, nt, np.int32),
            "face_lod": grab(d.face_lod, nt, np.int32),
            "edge_verts": grab(d.edge_verts, d.edge_count * 2, np.int32),
            "corner_edges": grab(d.corner_edges, nt * 3, np.int32),
            "vertex_count": nv,
            "tri_count": nt,
            "edge_count": d.edge_count,
        }


FRUSTUM_MODES = {"FULL": 0, "COARSEN": 1, "CULL": 2}


def make_view(camera, proj, znear, threshold, ortho=False, ortho_height=1.0, planes=None,
              mode="FULL", offscreen_scale=8.0):
    """mode: FULL (ignore frustum), COARSEN (off-screen at threshold*offscreen_scale), CULL."""
    v = View()
    v.camera[:] = [float(x) for x in camera]
    v.proj = float(proj)
    v.znear = float(znear)
    v.threshold = float(threshold)
    v.ortho = 1 if ortho else 0
    v.ortho_height = float(ortho_height)
    v.offscreen_scale = float(offscreen_scale)
    if planes is not None:
        v.frustum_mode = FRUSTUM_MODES[mode]
        for i in range(6):
            for j in range(4):
                v.planes[i][j] = float(planes[i][j])
    return v
