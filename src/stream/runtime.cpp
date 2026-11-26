// .vgeo v2 runtime: pick the DAG cut for the current views and hand it out
// chunk by chunk as small indexed meshes.

#include "vgeo_stream.h"
#include "format_v2.h"
#include "io_util.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// Prefetch hints run on one background thread per handle: PrefetchVirtualMemory can block
// for seconds when the pages are not in memory, which must never stall a caller's frame.
struct Prefetcher {
    std::thread worker;
    std::mutex m;
    std::condition_variable cv;
    std::vector<uint64_t> pending;   // latest request wins: older ones are for views already gone
    bool stop = false;
    const vgeo_io::MappedFile* map = nullptr;

    void request(const vgeo_io::MappedFile* file, std::vector<uint64_t>&& ranges) {
        {
            std::lock_guard<std::mutex> lk(m);
            map = file;
            pending = std::move(ranges);
            if (!worker.joinable()) worker = std::thread([this] { run(); });
        }
        cv.notify_one();
    }
    void run() {
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            cv.wait(lk, [this] { return stop || !pending.empty(); });
            if (stop) return;
            std::vector<uint64_t> r = std::move(pending);
            pending.clear();
            lk.unlock();
            map->prefetch(r.data(), r.size() / 2);
            lk.lock();
        }
    }
    ~Prefetcher() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_one();
        if (worker.joinable()) worker.join();
    }
};

// Per asset and level: the whole asset's triangles at that uniform level (local positions).
struct LevelTris {
    std::vector<float> pos;       // 9 floats per triangle
    float error = 0.f;
};

struct Asset {
    // the file is memory-mapped (read into `owned` only if mapping fails): pages load on first
    // touch, so opening costs nothing and memory is bounded by what the cuts actually use
    vgeo_io::MappedFile map;
    Prefetcher prefetcher;          // after `map`: stopped (joined) before the file is unmapped
    std::vector<uint8_t> owned;
    const uint8_t* base = nullptr;
    uint64_t size = 0;
    vgeo2::Header h;
    const float* positions = nullptr;
    const float* normals = nullptr;
    const float* uvs = nullptr;
    const uint16_t* vmat = nullptr;
    const uint32_t* indices = nullptr;
    const vgeo2::Cluster* clusters = nullptr;
    const vgeo2::Group* groups = nullptr;
    const vgeo2::Chunk* chunks = nullptr;
    const uint32_t* chunk_clusters = nullptr;
    std::vector<std::string> materials;

    // selection state
    std::vector<uint8_t> group_pass;
    std::vector<uint8_t> selected;

    // clusters are validated when first used (0 unchecked, 1 ok, 2 bad), not all at open:
    // a full scan would read the whole file
    std::vector<uint8_t> checked;
    std::vector<uint32_t> vmin, vmax;   // vertex range per checked cluster (for prefetch)
    bool corrupt = false;

    // extraction scratch: global vertex -> chunk-local index, a hash sized to the chunk
    // (per-vertex arrays would cost 8 bytes per vertex of the asset, per handle)
    std::vector<uint32_t> remap_key;
    std::vector<int32_t> remap_val;
    std::vector<float> out_pos, out_nrm, out_uv;
    std::vector<int32_t> out_corner, out_mat, out_lod;
    std::vector<int32_t> out_edges, out_corner_edge;
    std::vector<uint64_t> edge_keys;
    std::vector<int32_t> edge_ids;

    // cut atoms (built on first use): the clusters sharing a (group, refined) pair. The cut
    // rule tests only those two groups, so an atom's clusters are always selected together.
    std::vector<vgeo_atom> atoms;
    std::vector<uint32_t> atom_clusters;
    std::vector<uint32_t> extract_ids;
    // blocks: the atoms simplified from one parent group (2 or more), drawn as one piece when all of
    // them are in a copy's cut. Unit ids: atoms 0..A-1, blocks A..A+B-1.
    std::vector<vgeo_atom> blocks;      // cluster_offset/count index block_atoms
    std::vector<uint32_t> block_atoms;
    std::vector<int32_t> atom_block;    // per atom: its block, or -1

    // occluders: the whole asset at a uniform level, built on first use per level
    std::vector<LevelTris> level_cache;
    std::vector<uint64_t> level_tri_counts;
};

// Unique edges of an indexed triangle list, Blender style.
void build_edges(Asset& a, uint32_t vertex_count) {
    const size_t corners = a.out_corner.size();
    size_t cap = 16;
    while (cap < corners * 2) cap <<= 1;  // ~1.5 edges per triangle, keep load < 50%
    a.edge_keys.assign(cap, ~0ull);
    a.edge_ids.resize(cap);
    a.out_edges.clear();
    a.out_corner_edge.resize(corners);
    const uint64_t mask = cap - 1;
    (void)vertex_count;
    for (size_t t = 0; t < corners; t += 3) {
        for (int j = 0; j < 3; ++j) {
            uint32_t v0 = uint32_t(a.out_corner[t + j]);
            uint32_t v1 = uint32_t(a.out_corner[t + (j + 1) % 3]);
            uint64_t key = v0 < v1 ? (uint64_t(v0) << 32 | v1) : (uint64_t(v1) << 32 | v0);
            uint64_t h = (key * 0x9E3779B97F4A7C15ull) >> 20;
            size_t slot = size_t(h & mask);
            while (a.edge_keys[slot] != ~0ull && a.edge_keys[slot] != key) slot = (slot + 1) & mask;
            if (a.edge_keys[slot] == ~0ull) {
                a.edge_keys[slot] = key;
                a.edge_ids[slot] = int32_t(a.out_edges.size() / 2);
                a.out_edges.push_back(int32_t(key >> 32));
                a.out_edges.push_back(int32_t(key & 0xFFFFFFFFu));
            }
            a.out_corner_edge[t + j] = a.edge_ids[slot];
        }
    }
}

void set_err(char* err, int err_len, const std::string& msg) {
    if (err && err_len > 0) std::snprintf(err, size_t(err_len), "%s", msg.c_str());
}

bool section_ok(const Asset& a, uint64_t off, uint64_t bytes) {
    return off >= sizeof(vgeo2::Header) && off <= a.size && bytes <= a.size - off;
}

// Check a cluster's indices once; bad clusters are skipped (and the asset marked corrupt).
bool cluster_ok(Asset& a, uint32_t id) {
    uint8_t& st = a.checked[id];
    if (st) return st == 1;
    const vgeo2::Cluster& c = a.clusters[id];
    const uint32_t* idx = a.indices + c.index_offset;
    uint32_t lo = ~0u, hi = 0;
    bool ok = true;
    for (uint32_t k = 0; k < c.tri_count * 3; ++k) {
        uint32_t v = idx[k];
        if (v >= a.h.vertex_count) { ok = false; break; }
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    st = ok ? 1 : 2;
    if (ok) {
        a.vmin[id] = lo;
        a.vmax[id] = hi;
    } else {
        a.corrupt = true;
    }
    return ok;
}

inline bool in_frustum(const vgeo_view& v, const float* c, float r) {
    for (int p = 0; p < 6; ++p)
        if (v.planes[p][0] * c[0] + v.planes[p][1] * c[1] + v.planes[p][2] * c[2] + v.planes[p][3] < -r)
            return false;
    return true;
}

// projected error as a fraction of view height; FLT_MAX error never passes
inline bool group_passes(const vgeo2::Group& g, const vgeo_view* views, int view_count) {
    if (!(g.error < FLT_MAX)) return false;
    for (int i = 0; i < view_count; ++i) {
        const vgeo_view& v = views[i];
        float e;
        if (v.ortho) {
            e = g.error / std::max(v.ortho_height, 1e-20f);
        } else {
            float dx = g.center[0] - v.camera[0], dy = g.center[1] - v.camera[1], dz = g.center[2] - v.camera[2];
            float d = std::sqrt(dx * dx + dy * dy + dz * dz) - g.radius;
            e = g.error / std::max(d, v.znear) * (v.proj * 0.5f);
        }
        float t = v.threshold;
        if (v.frustum_mode == 1 && v.offscreen_scale > 1.f && !in_frustum(v, g.center, g.radius))
            t *= v.offscreen_scale;
        if (e > t) return false;
    }
    return true;
}

inline bool sphere_visible(const float* c, float r, const vgeo_view* views, int view_count) {
    for (int i = 0; i < view_count; ++i) {
        const vgeo_view& v = views[i];
        if (v.frustum_mode != 2 || in_frustum(v, c, r)) return true;  // kept by at least one view
    }
    return false;
}

inline uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

void signatures(Asset& a, uint64_t* chunk_sig, vgeo_cut_stats* stats) {
    uint64_t tris = 0;
    uint32_t count = 0;
    for (uint32_t c = 0; c < a.h.chunk_count; ++c) {
        const vgeo2::Chunk& ch = a.chunks[c];
        uint64_t sig = 0x243F6A8885A308D3ull;
        for (uint32_t k = 0; k < ch.cluster_count; ++k) {
            uint32_t id = a.chunk_clusters[ch.cluster_offset + k];
            if (a.selected[id]) {
                sig += mix(id);  // order independent
                tris += a.clusters[id].tri_count;
                ++count;
            }
        }
        if (chunk_sig) chunk_sig[c] = sig;
    }
    if (stats) {
        stats->triangles = tris;
        stats->clusters = count;
        stats->changed_chunks = 0;
    }
}

}  // namespace

extern "C" VGEO_API void* vgeo_open(const char* path_utf8, char* err, int err_len) {
    if (!path_utf8) { set_err(err, err_len, "no path"); return nullptr; }
    Asset* a = new Asset();
    if (a->map.open(path_utf8)) {
        a->base = a->map.data;
        a->size = a->map.size;
    } else {
        // mapping unavailable (or an empty file): read it whole
        FILE* f = vgeo_io::open_utf8(path_utf8, "rb");
        if (!f) { delete a; set_err(err, err_len, std::string("cannot open ") + path_utf8); return nullptr; }
        std::fseek(f, 0, SEEK_END);
        int64_t size = vgeo_io::tell(f);
        vgeo_io::seek(f, 0);
        if (size < int64_t(sizeof(vgeo2::Header))) {
            std::fclose(f); delete a;
            set_err(err, err_len, "file too small");
            return nullptr;
        }
        try {
            a->owned.resize(size_t(size));
        } catch (...) {
            std::fclose(f); delete a;
            set_err(err, err_len, "out of memory");
            return nullptr;
        }
        size_t got = std::fread(a->owned.data(), 1, size_t(size), f);
        std::fclose(f);
        if (got != size_t(size)) { delete a; set_err(err, err_len, "read failed"); return nullptr; }
        a->base = a->owned.data();
        a->size = uint64_t(size);
    }
    if (a->size < sizeof(vgeo2::Header)) { delete a; set_err(err, err_len, "file too small"); return nullptr; }

    std::memcpy(&a->h, a->base, sizeof(vgeo2::Header));
    const vgeo2::Header& h = a->h;
    if (std::memcmp(h.magic, vgeo2::kMagic, 8) != 0) { delete a; set_err(err, err_len, "not a VGEO v2 file"); return nullptr; }
    if (h.version != vgeo2::kVersion) { delete a; set_err(err, err_len, "unsupported VGEO version"); return nullptr; }
    bool ok = h.file_size <= a->size
        && section_ok(*a, h.off_positions, uint64_t(h.vertex_count) * 12)
        && section_ok(*a, h.off_normals, uint64_t(h.vertex_count) * 12)
        && (!(h.flags & vgeo2::kHasUVs) || section_ok(*a, h.off_uvs, uint64_t(h.vertex_count) * 8))
        && section_ok(*a, h.off_vmat, uint64_t(h.vertex_count) * 2)
        && section_ok(*a, h.off_indices, uint64_t(h.index_count) * 4)
        && section_ok(*a, h.off_clusters, uint64_t(h.cluster_count) * sizeof(vgeo2::Cluster))
        && section_ok(*a, h.off_groups, uint64_t(h.group_count) * sizeof(vgeo2::Group))
        && section_ok(*a, h.off_chunks, uint64_t(h.chunk_count) * sizeof(vgeo2::Chunk))
        && section_ok(*a, h.off_chunk_clusters, uint64_t(h.chunk_cluster_count) * 4)
        && section_ok(*a, h.off_materials, 4);
    if (!ok) { delete a; set_err(err, err_len, "corrupt file (section out of range)"); return nullptr; }

    const uint8_t* b = a->base;
    a->positions = reinterpret_cast<const float*>(b + h.off_positions);
    a->normals = reinterpret_cast<const float*>(b + h.off_normals);
    a->uvs = (h.flags & vgeo2::kHasUVs) ? reinterpret_cast<const float*>(b + h.off_uvs) : nullptr;
    a->vmat = reinterpret_cast<const uint16_t*>(b + h.off_vmat);
    a->indices = reinterpret_cast<const uint32_t*>(b + h.off_indices);
    a->clusters = reinterpret_cast<const vgeo2::Cluster*>(b + h.off_clusters);
    a->groups = reinterpret_cast<const vgeo2::Group*>(b + h.off_groups);
    a->chunks = reinterpret_cast<const vgeo2::Chunk*>(b + h.off_chunks);
    a->chunk_clusters = reinterpret_cast<const uint32_t*>(b + h.off_chunk_clusters);

    // validate every reference once so the hot paths can trust the data
    for (uint32_t i = 0; i < h.cluster_count && ok; ++i) {
        const vgeo2::Cluster& c = a->clusters[i];
        ok = uint64_t(c.index_offset) + uint64_t(c.tri_count) * 3 <= h.index_count
            && c.group >= 0 && uint32_t(c.group) < h.group_count
            && c.refined >= -1 && c.refined < int32_t(h.group_count)
            && c.chunk < h.chunk_count;
    }
    // indices are checked per cluster on first use (cluster_ok): scanning them all here would
    // read the whole file
    for (uint32_t i = 0; i < h.chunk_count && ok; ++i)
        ok = uint64_t(a->chunks[i].cluster_offset) + a->chunks[i].cluster_count <= h.chunk_cluster_count;
    for (uint32_t i = 0; i < h.chunk_cluster_count && ok; ++i) ok = a->chunk_clusters[i] < h.cluster_count;
    if (!ok) { delete a; set_err(err, err_len, "corrupt file (bad reference)"); return nullptr; }

    {
        const uint8_t* p = b + h.off_materials;
        const uint8_t* end = b + a->size;
        uint32_t n;
        std::memcpy(&n, p, 4);
        p += 4;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t len;
            if (end - p < 4) break;
            std::memcpy(&len, p, 4);
            p += 4;
            if (uint64_t(end - p) < len) break;
            a->materials.emplace_back(reinterpret_cast<const char*>(p), len);
            p += len;
        }
    }

    a->group_pass.assign(h.group_count, 0);
    a->selected.assign(h.cluster_count, 0);
    a->checked.assign(h.cluster_count, 0);
    a->vmin.assign(h.cluster_count, 0);
    a->vmax.assign(h.cluster_count, 0);
    return a;
}

extern "C" VGEO_API void vgeo_close(void* handle) { delete static_cast<Asset*>(handle); }

extern "C" VGEO_API int vgeo_get_info(void* handle, vgeo_info* info) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !info) return 1;
    info->vertex_count = a->h.vertex_count;
    info->cluster_count = a->h.cluster_count;
    info->group_count = a->h.group_count;
    info->chunk_count = a->h.chunk_count;
    info->material_count = uint32_t(a->materials.size());
    info->lod_levels = a->h.lod_levels;
    info->flags = a->h.flags;
    info->source_triangles = a->h.source_triangles;
    std::memcpy(info->aabb_min, a->h.aabb_min, sizeof(info->aabb_min));
    std::memcpy(info->aabb_max, a->h.aabb_max, sizeof(info->aabb_max));
    return 0;
}

extern "C" VGEO_API int vgeo_material_name(void* handle, uint32_t index, char* buf, int buf_len) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || index >= a->materials.size() || !buf || buf_len <= 0) return 1;
    std::snprintf(buf, size_t(buf_len), "%s", a->materials[index].c_str());
    return 0;
}

extern "C" VGEO_API int vgeo_select(void* handle, const vgeo_view* views, int view_count,
                                    uint64_t* chunk_sig, vgeo_cut_stats* stats) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !views || view_count <= 0) return 1;
    for (uint32_t g = 0; g < a->h.group_count; ++g)
        a->group_pass[g] = group_passes(a->groups[g], views, view_count) ? 1 : 0;
    for (uint32_t i = 0; i < a->h.cluster_count; ++i) {
        const vgeo2::Cluster& c = a->clusters[i];
        bool in_cut = !a->group_pass[c.group] && (c.refined < 0 || a->group_pass[c.refined]);
        a->selected[i] = (in_cut && sphere_visible(c.center, c.radius, views, view_count)) ? 1 : 0;
    }
    signatures(*a, chunk_sig, stats);
    return 0;
}

extern "C" VGEO_API int vgeo_select_level(void* handle, int depth, uint64_t* chunk_sig, vgeo_cut_stats* stats) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a) return 1;
    for (uint32_t i = 0; i < a->h.cluster_count; ++i) {
        const vgeo2::Cluster& c = a->clusters[i];
        bool terminal = !(a->groups[c.group].error < FLT_MAX);
        bool sel = depth < 0 ? terminal : (int(c.depth) == depth || (int(c.depth) < depth && terminal));
        a->selected[i] = sel ? 1 : 0;
    }
    signatures(*a, chunk_sig, stats);
    return 0;
}

extern "C" VGEO_API int vgeo_level_errors(void* handle, float* out, int max_levels) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !out || max_levels <= 0) return 0;
    const int n = std::min<int>(max_levels, int(a->h.lod_levels));
    std::vector<float> err(size_t(std::max(n, 1)), 0.f);
    // clusters at depth L were produced by simplifying their refined group (depth L-1);
    // that group's error bounds how far level L deviates from the source
    for (uint32_t i = 0; i < a->h.cluster_count; ++i) {
        const vgeo2::Cluster& c = a->clusters[i];
        if (c.refined < 0 || int(c.depth) >= n) continue;
        float e = a->groups[c.refined].error;
        if (e < FLT_MAX) err[c.depth] = std::max(err[c.depth], e);
    }
    for (int L = 1; L < n; ++L) err[L] = std::max(err[L], err[L - 1]);  // monotonic
    std::copy(err.begin(), err.begin() + n, out);
    return n;
}

namespace {

// Extract clusters `ids` (checked ones only) as one indexed mesh with Blender-style edges.
void extract_clusters(Asset* a, const uint32_t* ids, size_t n, vgeo_chunk_data* out) {
    a->out_pos.clear(); a->out_nrm.clear(); a->out_uv.clear();
    a->out_corner.clear(); a->out_mat.clear(); a->out_lod.clear();

    uint64_t corners = 0;
    for (size_t k = 0; k < n; ++k)
        if (cluster_ok(*a, ids[k])) corners += uint64_t(a->clusters[ids[k]].tri_count) * 3;
    size_t cap = 64;
    while (cap < corners * 2) cap <<= 1;   // load factor < 50% even if every corner is unique
    a->remap_key.assign(cap, ~0u);
    a->remap_val.resize(cap);
    const size_t mask = cap - 1;

    int32_t next = 0;
    for (size_t k = 0; k < n; ++k) {
        uint32_t id = ids[k];
        if (a->checked[id] != 1) continue;
        const vgeo2::Cluster& c = a->clusters[id];
        const uint32_t* idx = a->indices + c.index_offset;
        for (uint32_t t = 0; t < c.tri_count; ++t) {
            for (int j = 0; j < 3; ++j) {
                uint32_t v = idx[t * 3 + j];
                size_t slot = size_t((uint64_t(v) * 0x9E3779B97F4A7C15ull) >> 32) & mask;
                while (a->remap_key[slot] != ~0u && a->remap_key[slot] != v) slot = (slot + 1) & mask;
                if (a->remap_key[slot] == ~0u) {
                    a->remap_key[slot] = v;
                    a->remap_val[slot] = next++;
                    a->out_pos.insert(a->out_pos.end(), a->positions + size_t(v) * 3, a->positions + size_t(v) * 3 + 3);
                    a->out_nrm.insert(a->out_nrm.end(), a->normals + size_t(v) * 3, a->normals + size_t(v) * 3 + 3);
                    if (a->uvs) a->out_uv.insert(a->out_uv.end(), a->uvs + size_t(v) * 2, a->uvs + size_t(v) * 2 + 2);
                }
                a->out_corner.push_back(a->remap_val[slot]);
            }
            a->out_mat.push_back(int32_t(a->vmat[idx[t * 3]]));
            a->out_lod.push_back(int32_t(c.depth));
        }
    }
    build_edges(*a, uint32_t(next));
    out->edge_count = uint32_t(a->out_edges.size() / 2);
    out->edge_verts = a->out_edges.data();
    out->corner_edges = a->out_corner_edge.data();
    out->vertex_count = uint32_t(next);
    out->tri_count = uint32_t(a->out_mat.size());
    out->positions = a->out_pos.data();
    out->normals = a->out_nrm.data();
    out->uvs = a->uvs ? a->out_uv.data() : nullptr;
    out->corner_verts = a->out_corner.data();
    out->face_materials = a->out_mat.data();
    out->face_lod = a->out_lod.data();
}

}  // namespace

extern "C" VGEO_API int vgeo_extract(void* handle, uint32_t chunk, vgeo_chunk_data* out) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !out || chunk >= a->h.chunk_count) return 1;
    const vgeo2::Chunk& ch = a->chunks[chunk];
    a->extract_ids.clear();
    for (uint32_t k = 0; k < ch.cluster_count; ++k) {
        uint32_t id = a->chunk_clusters[ch.cluster_offset + k];
        if (a->selected[id]) a->extract_ids.push_back(id);
    }
    extract_clusters(a, a->extract_ids.data(), a->extract_ids.size(), out);
    return 0;
}

extern "C" VGEO_API int vgeo_prefetch(void* handle, const uint32_t* chunks, int count) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || (!chunks && count > 0)) return 1;
    if (a->owned.size() || count <= 0) return 0;   // nothing to fetch when the file is in memory
    // byte ranges of the selected clusters' indices, plus their vertex data once known
    std::vector<uint64_t> r;
    auto add = [&](uint64_t off, uint64_t bytes) {
        if (bytes == 0 || off >= a->size) return;
        bytes = std::min(bytes, a->size - off);
        if (!r.empty() && r[r.size() - 2] + r.back() + 4096 >= off && r[r.size() - 2] <= off) {
            uint64_t end = std::max(r[r.size() - 2] + r.back(), off + bytes);
            r.back() = end - r[r.size() - 2];
        } else {
            r.push_back(off);
            r.push_back(bytes);
        }
    };
    const vgeo2::Header& h = a->h;
    for (int i = 0; i < count; ++i) {
        if (chunks[i] >= h.chunk_count) continue;
        const vgeo2::Chunk& ch = a->chunks[chunks[i]];
        for (uint32_t k = 0; k < ch.cluster_count; ++k) {
            uint32_t id = a->chunk_clusters[ch.cluster_offset + k];
            if (!a->selected[id]) continue;
            const vgeo2::Cluster& c = a->clusters[id];
            add(h.off_indices + uint64_t(c.index_offset) * 4, uint64_t(c.tri_count) * 12);
            // vertex data once the cluster was seen, and only when its vertices sit close together
            // (a coarse cluster can span most of the file)
            if (a->checked[id] == 1 && uint64_t(a->vmax[id] - a->vmin[id]) < 65536) {
                uint64_t v0 = a->vmin[id], n = uint64_t(a->vmax[id]) - v0 + 1;
                add(h.off_positions + v0 * 12, n * 12);
                add(h.off_normals + v0 * 12, n * 12);
                if (a->uvs) add(h.off_uvs + v0 * 8, n * 8);
                add(h.off_vmat + v0 * 2, n * 2);
            }
        }
    }
    a->prefetcher.request(&a->map, std::move(r));
    return 0;
}

extern "C" VGEO_API int vgeo_memory(void* handle, uint64_t* mapped_bytes, uint64_t* heap_bytes) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a) return 1;
    uint64_t heap = a->owned.capacity() + a->group_pass.capacity() + a->selected.capacity()
        + a->checked.capacity() + (a->vmin.capacity() + a->vmax.capacity()) * 4
        + (a->remap_key.capacity() + a->remap_val.capacity()) * 4
        + (a->out_pos.capacity() + a->out_nrm.capacity() + a->out_uv.capacity()) * 4
        + (a->out_corner.capacity() + a->out_mat.capacity() + a->out_lod.capacity()) * 4
        + (a->out_edges.capacity() + a->out_corner_edge.capacity() + a->edge_ids.capacity()) * 4
        + a->edge_keys.capacity() * 8
        + (a->atoms.capacity() + a->blocks.capacity()) * sizeof(vgeo_atom)
        + (a->atom_clusters.capacity() + a->extract_ids.capacity() + a->block_atoms.capacity()
           + a->atom_block.capacity()) * 4;
    if (mapped_bytes) *mapped_bytes = a->owned.size() ? 0 : a->size;
    if (heap_bytes) *heap_bytes = heap;
    return 0;
}

extern "C" VGEO_API int vgeo_corrupt(void* handle) {
    Asset* a = static_cast<Asset*>(handle);
    return a && a->corrupt ? 1 : 0;
}

// ---------------------------------------------------------------- occlusion culling

namespace {

inline void xform(const float* m, const float* p, float* out) {
    for (int r = 0; r < 3; ++r) out[r] = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3];
}


// Linear view depth (distance along the view direction) per pixel of a small buffer; the occluders
// are coarse levels of the nearest copies, pushed away from the camera by their simplification
// error and eroded by a pixel, so the buffer only ever claims what the real surfaces hide.
struct Occlusion {
    int w = 0, h = 0;
    float vp[16] = {};            // world -> clip, row-major
    float cam[3] = {};
    float near_w = 1e-3f;
    std::vector<float> depth;     // w per pixel, +inf = nothing
    std::vector<std::vector<float>> hiz;   // max-depth pyramid, level 0 = eroded depth
    std::vector<int> hw, hh;
    uint64_t triangles = 0;
    bool ready = false;
    uint64_t copy_tris = 6000;    // per copy at most (a coarser level is still a valid occluder)
    bool erode = true;
    bool push = true;
};

std::mutex g_level_mutex;

const LevelTris& level_tris(Asset& a, int depth, std::vector<LevelTris>& cache) {
    std::lock_guard<std::mutex> lock(g_level_mutex);
    if (cache.empty()) cache.resize(std::max(1u, a.h.lod_levels));
    depth = std::min<int>(depth, int(cache.size()) - 1);
    LevelTris& L = cache[depth];
    if (!L.pos.empty()) return L;
    for (uint32_t i = 0; i < a.h.cluster_count; ++i) {
        const vgeo2::Cluster& c = a.clusters[i];
        bool terminal = !(a.groups[c.group].error < FLT_MAX);
        bool sel = int(c.depth) == depth || (int(c.depth) < depth && terminal);
        if (!sel || !cluster_ok(a, i)) continue;
        if (c.refined >= 0 && a.groups[c.refined].error < FLT_MAX)
            L.error = std::max(L.error, a.groups[c.refined].error);
        const uint32_t* idx = a.indices + c.index_offset;
        for (uint32_t t = 0; t < c.tri_count * 3; ++t) {
            const float* p = a.positions + size_t(idx[t]) * 3;
            L.pos.insert(L.pos.end(), p, p + 3);
        }
    }
    if (L.pos.empty()) L.pos.push_back(0.f);   // never rebuilt
    return L;
}

inline void to_clip(const float* m, const float* p, float* out) {
    for (int r = 0; r < 4; ++r) out[r] = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3];
}

void raster_tri(Occlusion& O, const float* a, const float* b, const float* c) {
    float ca[4], cb[4], cc[4];
    to_clip(O.vp, a, ca);
    to_clip(O.vp, b, cb);
    to_clip(O.vp, c, cc);
    if (ca[3] < O.near_w || cb[3] < O.near_w || cc[3] < O.near_w) return;   // crosses the near plane: skip
    // screen positions (pixel units, y down irrelevant: tests use the same mapping) and 1/w
    float sx[3], sy[3], iw[3];
    const float* cs[3] = {ca, cb, cc};
    for (int k = 0; k < 3; ++k) {
        iw[k] = 1.f / cs[k][3];
        sx[k] = (cs[k][0] * iw[k] * 0.5f + 0.5f) * O.w;
        sy[k] = (cs[k][1] * iw[k] * 0.5f + 0.5f) * O.h;
    }
    float area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
    if (std::fabs(area) < 1e-12f) return;
    int x0 = std::max(0, int(std::floor(std::min({sx[0], sx[1], sx[2]}))));
    int x1 = std::min(O.w - 1, int(std::ceil(std::max({sx[0], sx[1], sx[2]}))));
    int y0 = std::max(0, int(std::floor(std::min({sy[0], sy[1], sy[2]}))));
    int y1 = std::min(O.h - 1, int(std::ceil(std::max({sy[0], sy[1], sy[2]}))));
    if (x0 > x1 || y0 > y1) return;
    const float inv = 1.f / area;
    for (int y = y0; y <= y1; ++y) {
        const float py = y + 0.5f;
        for (int x = x0; x <= x1; ++x) {
            const float px = x + 0.5f;
            float w0 = ((sx[1] - px) * (sy[2] - py) - (sx[2] - px) * (sy[1] - py)) * inv;
            float w1 = ((sx[2] - px) * (sy[0] - py) - (sx[0] - px) * (sy[2] - py)) * inv;
            float w2 = 1.f - w0 - w1;
            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
            float d = 1.f / (w0 * iw[0] + w1 * iw[1] + w2 * iw[2]);   // perspective-correct view depth
            float& dst = O.depth[size_t(y) * O.w + x];
            if (d < dst) dst = d;
        }
    }
}

// Screen rect (pixels of level 0) and nearest view depth of a sphere; false if it reaches the camera
// or leaves the screen in a way the rect cannot express (then it counts as visible).
bool sphere_rect(const Occlusion& O, const float* c, float r, int& x0, int& y0, int& x1, int& y1, float& near_d) {
    float lo_x = FLT_MAX, lo_y = FLT_MAX, hi_x = -FLT_MAX, hi_y = -FLT_MAX;
    float cw[4];
    to_clip(O.vp, c, cw);
    // the view direction is the clip w row: distance along it of the sphere's nearest point
    float wlen = std::sqrt(O.vp[12] * O.vp[12] + O.vp[13] * O.vp[13] + O.vp[14] * O.vp[14]);
    near_d = cw[3] - r * wlen;
    if (near_d < O.near_w) return false;
    for (int k = 0; k < 8; ++k) {
        float p[3] = {c[0] + ((k & 1) ? r : -r), c[1] + ((k & 2) ? r : -r), c[2] + ((k & 4) ? r : -r)};
        float q[4];
        to_clip(O.vp, p, q);
        if (q[3] < O.near_w) return false;
        float x = (q[0] / q[3] * 0.5f + 0.5f) * O.w, y = (q[1] / q[3] * 0.5f + 0.5f) * O.h;
        lo_x = std::min(lo_x, x); hi_x = std::max(hi_x, x);
        lo_y = std::min(lo_y, y); hi_y = std::max(hi_y, y);
    }
    x0 = std::max(0, int(std::floor(lo_x)));
    y0 = std::max(0, int(std::floor(lo_y)));
    x1 = std::min(O.w - 1, int(std::floor(hi_x)));
    y1 = std::min(O.h - 1, int(std::floor(hi_y)));
    return x0 <= x1 && y0 <= y1;   // entirely off screen: not this test's business
}

// True when every occluder pixel the sphere covers is nearer than the sphere's nearest point.
bool sphere_occluded(const Occlusion& O, const float* c, float r) {
    if (!O.ready) return false;
    int x0, y0, x1, y1;
    float near_d;
    if (!sphere_rect(O, c, r, x0, y0, x1, y1, near_d)) return false;
    // pick the pyramid level where the rect spans at most 4 texels a side
    int lvl = 0;
    while (lvl + 1 < int(O.hiz.size()) && ((x1 >> lvl) - (x0 >> lvl) > 3 || (y1 >> lvl) - (y0 >> lvl) > 3)) ++lvl;
    const std::vector<float>& H = O.hiz[lvl];
    const int W = O.hw[lvl];
    for (int y = y0 >> lvl; y <= (y1 >> lvl); ++y)
        for (int x = x0 >> lvl; x <= (x1 >> lvl); ++x)
            if (!(H[size_t(y) * W + x] < near_d)) return false;
    return true;
}

}  // namespace

extern "C" VGEO_API void* vgeo_occlusion_new(int width, int height) {
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) return nullptr;
    Occlusion* O = new Occlusion();
    O->w = width;
    O->h = height;
    O->depth.assign(size_t(width) * height, FLT_MAX);
    return O;
}

extern "C" VGEO_API void vgeo_occlusion_free(void* occ) { delete static_cast<Occlusion*>(occ); }

// Experiments: triangles per occluder copy (default 6000), erosion and error push (default on).
extern "C" VGEO_API int vgeo_occlusion_config(void* occ, uint64_t copy_triangles, int erode, int push) {
    Occlusion* O = static_cast<Occlusion*>(occ);
    if (!O) return 1;
    if (copy_triangles) O->copy_tris = copy_triangles;
    O->erode = erode != 0;
    O->push = push != 0;
    return 0;
}

extern "C" VGEO_API int vgeo_occlusion_begin(void* occ, const float* view_proj, const float* camera, float znear) {
    Occlusion* O = static_cast<Occlusion*>(occ);
    if (!O || !view_proj || !camera) return 1;
    std::memcpy(O->vp, view_proj, sizeof(O->vp));
    std::memcpy(O->cam, camera, sizeof(O->cam));
    O->near_w = std::max(znear, 1e-6f);
    std::fill(O->depth.begin(), O->depth.end(), FLT_MAX);
    O->triangles = 0;
    O->ready = false;
    return 0;
}

extern "C" VGEO_API int vgeo_occlusion_add(void* occ, void* handle, const float* transforms, const float* scales,
                                           int count, float pixel_error, uint64_t max_triangles,
                                           uint64_t* out_triangles) {
    Occlusion* O = static_cast<Occlusion*>(occ);
    Asset* a = static_cast<Asset*>(handle);
    if (!O || !a || (count > 0 && (!transforms || !scales))) return 1;
    // errors per uniform level, coarsest that keeps a copy within pixel_error of the buffer
    std::vector<float> errs(std::max(1u, a->h.lod_levels), 0.f);
    vgeo_level_errors(a, errs.data(), int(errs.size()));
    const float proj_h = std::sqrt(O->vp[4] * O->vp[4] + O->vp[5] * O->vp[5] + O->vp[6] * O->vp[6]);
    float lo[3], hi[3], cl[3];
    for (int d = 0; d < 3; ++d) {
        lo[d] = a->h.aabb_min[d];
        hi[d] = a->h.aabb_max[d];
        cl[d] = 0.5f * (lo[d] + hi[d]);
    }
    float rl = 0.5f * std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) +
                                (hi[2] - lo[2]) * (hi[2] - lo[2]));
    const float wlen = std::sqrt(O->vp[12] * O->vp[12] + O->vp[13] * O->vp[13] + O->vp[14] * O->vp[14]);
    {
        std::lock_guard<std::mutex> lock(g_level_mutex);
        if (a->level_tri_counts.empty()) {
            a->level_tri_counts.assign(std::max(1u, a->h.lod_levels), 0);
            for (uint32_t i = 0; i < a->h.cluster_count; ++i) {
                const vgeo2::Cluster& c = a->clusters[i];
                bool terminal = !(a->groups[c.group].error < FLT_MAX);
                for (uint32_t L = 0; L < a->level_tri_counts.size(); ++L)
                    if (c.depth == L || (c.depth < L && terminal)) a->level_tri_counts[L] += c.tri_count;
            }
        }
    }
    for (int i = 0; i < count && (max_triangles == 0 || O->triangles < max_triangles); ++i) {
        const float* M = transforms + size_t(i) * 12;
        const float s = scales[i];
        float cw[3], q[4];
        xform(M, cl, cw);
        to_clip(O->vp, cw, q);
        float dist = (q[3] - rl * s * wlen) / std::max(wlen, 1e-20f);
        if (dist < O->near_w) dist = O->near_w;
        // error in buffer pixels of level L: err * s / dist * proj_h * h / 2
        int L = 0;
        for (int k = int(errs.size()) - 1; k >= 0; --k)
            if (errs[k] * s / dist * proj_h / std::max(wlen, 1e-20f) * O->h * 0.5f <= pixel_error) { L = k; break; }
        // too heavy at the level its error allows: not an occluder (a coarser level could bulge
        // sideways past the real silhouette, which pushing it back does not undo)
        if (L < int(a->level_tri_counts.size()) && a->level_tri_counts[L] > O->copy_tris) continue;
        const LevelTris& T = level_tris(*a, L, a->level_cache);
        const float push = O->push ? T.error * s : 0.f;   // move each vertex away from the camera by the level's error
        const size_t n = T.pos.size() / 9;
        for (size_t t = 0; t < n; ++t) {
            float v[3][3];
            for (int k = 0; k < 3; ++k) {
                xform(M, &T.pos[t * 9 + k * 3], v[k]);
                if (push > 0.f) {
                    float dx = v[k][0] - O->cam[0], dy = v[k][1] - O->cam[1], dz = v[k][2] - O->cam[2];
                    float len = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (len > 0.f) {
                        v[k][0] += dx / len * push; v[k][1] += dy / len * push; v[k][2] += dz / len * push;
                    }
                }
            }
            raster_tri(*O, v[0], v[1], v[2]);
        }
        O->triangles += n;
    }
    if (out_triangles) *out_triangles = O->triangles;
    return 0;
}

extern "C" VGEO_API int vgeo_occlusion_finish(void* occ) {
    Occlusion* O = static_cast<Occlusion*>(occ);
    if (!O) return 1;
    // erode the occluders by a pixel (3x3 max of depth): a pixel only counts as hidden when its
    // neighbours are too, which covers partially covered edge pixels
    O->hiz.clear(); O->hw.clear(); O->hh.clear();
    std::vector<float> e(O->depth.size());
    for (int y = 0; y < O->h; ++y)
        for (int x = 0; x < O->w; ++x) {
            float m = 0.f;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = x + dx, yy = y + dy;
                    if (!O->erode && (dx || dy)) continue;
                    float d = (xx < 0 || yy < 0 || xx >= O->w || yy >= O->h) ? FLT_MAX : O->depth[size_t(yy) * O->w + xx];
                    m = std::max(m, d);
                }
            e[size_t(y) * O->w + x] = m;
        }
    O->hiz.push_back(std::move(e)); O->hw.push_back(O->w); O->hh.push_back(O->h);
    while (O->hw.back() > 1 || O->hh.back() > 1) {
        const std::vector<float>& P = O->hiz.back();
        int pw = O->hw.back(), ph = O->hh.back();
        int nw = std::max(1, (pw + 1) / 2), nh = std::max(1, (ph + 1) / 2);
        std::vector<float> N(size_t(nw) * nh);
        for (int y = 0; y < nh; ++y)
            for (int x = 0; x < nw; ++x) {
                float m = 0.f;
                for (int k = 0; k < 4; ++k) {
                    int xx = std::min(pw - 1, x * 2 + (k & 1)), yy = std::min(ph - 1, y * 2 + (k >> 1));
                    m = std::max(m, P[size_t(yy) * pw + xx]);
                }
                N[size_t(y) * nw + x] = m;
            }
        O->hiz.push_back(std::move(N)); O->hw.push_back(nw); O->hh.push_back(nh);
    }
    O->ready = true;
    return 0;
}

extern "C" VGEO_API int vgeo_occlusion_test(void* occ, const float* centers, const float* radii, int count,
                                            uint8_t* out_occluded) {
    Occlusion* O = static_cast<Occlusion*>(occ);
    if (!O || (count > 0 && (!centers || !radii || !out_occluded))) return 1;
    for (int i = 0; i < count; ++i) out_occluded[i] = sphere_occluded(*O, centers + size_t(i) * 3, radii[i]) ? 1 : 0;
    return 0;
}

extern "C" VGEO_API int vgeo_occlusion_depth(void* occ, float* out) {
    Occlusion* O = static_cast<Occlusion*>(occ);
    if (!O || !out) return 1;
    const std::vector<float>& src = O->ready ? O->hiz[0] : O->depth;
    std::copy(src.begin(), src.end(), out);
    return 0;
}

// ---------------------------------------------------------------- cut atoms and many copies

namespace {

void build_atoms(Asset& a) {
    if (!a.atoms.empty() || a.h.cluster_count == 0) return;
    std::vector<uint32_t> order(a.h.cluster_count);
    for (uint32_t i = 0; i < a.h.cluster_count; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
        const vgeo2::Cluster& cx = a.clusters[x];
        const vgeo2::Cluster& cy = a.clusters[y];
        return cx.group != cy.group ? cx.group < cy.group : cx.refined < cy.refined;
    });
    a.atom_clusters = order;
    for (uint32_t k = 0; k < order.size();) {
        const vgeo2::Cluster& first = a.clusters[order[k]];
        uint32_t e = k;
        while (e < order.size() && a.clusters[order[e]].group == first.group &&
               a.clusters[order[e]].refined == first.refined)
            ++e;
        vgeo_atom at{};
        at.group = first.group;
        at.refined = first.refined;
        at.cluster_offset = k;
        at.cluster_count = e - k;
        at.depth = first.depth;
        // a sphere enclosing the clusters' spheres, centred on their bounding box
        float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t j = k; j < e; ++j) {
            const vgeo2::Cluster& c = a.clusters[order[j]];
            at.tri_count += c.tri_count;
            for (int d = 0; d < 3; ++d) {
                lo[d] = std::min(lo[d], c.center[d] - c.radius);
                hi[d] = std::max(hi[d], c.center[d] + c.radius);
            }
        }
        for (int d = 0; d < 3; ++d) at.center[d] = 0.5f * (lo[d] + hi[d]);
        float r = 0.f;
        for (uint32_t j = k; j < e; ++j) {
            const vgeo2::Cluster& c = a.clusters[order[j]];
            float dx = c.center[0] - at.center[0], dy = c.center[1] - at.center[1], dz = c.center[2] - at.center[2];
            r = std::max(r, std::sqrt(dx * dx + dy * dy + dz * dz) + c.radius);
        }
        at.radius = r * 1.0001f + 1e-7f;
        a.atoms.push_back(at);
        k = e;
    }
    // blocks by parent group
    const uint32_t na = uint32_t(a.atoms.size());
    a.atom_block.assign(na, -1);
    std::vector<uint32_t> by_parent(na);
    for (uint32_t i = 0; i < na; ++i) by_parent[i] = i;
    std::stable_sort(by_parent.begin(), by_parent.end(),
                     [&](uint32_t x, uint32_t y) { return a.atoms[x].refined < a.atoms[y].refined; });
    for (uint32_t k = 0; k < na;) {
        const int32_t r = a.atoms[by_parent[k]].refined;
        uint32_t e = k;
        while (e < na && a.atoms[by_parent[e]].refined == r) ++e;
        if (r >= 0 && e - k >= 2) {
            vgeo_atom b{};
            b.group = -1;
            b.refined = r;
            b.cluster_offset = uint32_t(a.block_atoms.size());
            b.cluster_count = e - k;
            b.depth = a.atoms[by_parent[k]].depth;
            float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (uint32_t j = k; j < e; ++j) {
                const vgeo_atom& at = a.atoms[by_parent[j]];
                a.block_atoms.push_back(by_parent[j]);
                a.atom_block[by_parent[j]] = int32_t(a.blocks.size());
                b.tri_count += at.tri_count;
                b.depth = std::min(b.depth, at.depth);
                for (int d = 0; d < 3; ++d) {
                    lo[d] = std::min(lo[d], at.center[d] - at.radius);
                    hi[d] = std::max(hi[d], at.center[d] + at.radius);
                }
            }
            for (int d = 0; d < 3; ++d) b.center[d] = 0.5f * (lo[d] + hi[d]);
            float rad = 0.f;
            for (uint32_t j = k; j < e; ++j) {
                const vgeo_atom& at = a.atoms[by_parent[j]];
                float dx = at.center[0] - b.center[0], dy = at.center[1] - b.center[1], dz = at.center[2] - b.center[2];
                rad = std::max(rad, std::sqrt(dx * dx + dy * dy + dz * dz) + at.radius);
            }
            b.radius = rad * 1.0001f + 1e-7f;
            a.blocks.push_back(b);
        }
        k = e;
    }
}

inline bool inside_any(const vgeo_view* views, int n, const float* c, float r) {
    for (int i = 0; i < n; ++i)
        if (in_frustum(views[i], c, r)) return true;
    return false;
}

// Can the shadow of a sphere (c, r), part of a copy of radius copy_r, reach any view?
bool shadow_reaches(const vgeo_view* views, int nv, const vgeo_shadow_light* lights, int nl,
                    const float* c, float r, float copy_r) {
    for (int l = 0; l < nl; ++l) {
        const vgeo_shadow_light& L = lights[l];
        for (int i = 0; i < nv; ++i) {
            const vgeo_view& v = views[i];
            if (L.kind == 0) {
                // the sphere swept along the light by `reach` must not lie fully outside one plane
                float reach = L.reach > 0.f ? L.reach : 2.f * copy_r / std::max(-L.dir[2], 0.1f);
                bool hit = true;
                for (int p = 0; p < 6 && hit; ++p) {
                    float d = v.planes[p][0] * c[0] + v.planes[p][1] * c[1] + v.planes[p][2] * c[2] + v.planes[p][3];
                    float nd = v.planes[p][0] * L.dir[0] + v.planes[p][1] * L.dir[1] + v.planes[p][2] * L.dir[2];
                    if (d + reach * std::max(nd, 0.f) < -r) hit = false;
                }
                if (hit) return true;
            } else {
                if (!in_frustum(v, L.pos, L.range)) continue;   // its light never reaches this view
                float dx = c[0] - L.pos[0], dy = c[1] - L.pos[1], dz = c[2] - L.pos[2];
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist - r > L.range) continue;
                if (L.half_angle < 1.5707964f && dist > r) {
                    float cosang = (dx * L.axis[0] + dy * L.axis[1] + dz * L.axis[2]) / dist;
                    float ang = std::acos(std::min(1.f, std::max(-1.f, cosang))) - std::asin(std::min(1.f, r / dist));
                    if (ang > L.half_angle) continue;
                }
                return true;
            }
        }
    }
    return false;
}

struct InstanceJob {
    const Asset* a;
    const vgeo_view* views;
    int nv;
    const float* transforms;
    const float* scales;
    const uint8_t* mask;
    int mode;
    float offscreen_scale;
    const vgeo_shadow_light* lights;
    int nl;
    float local_center[3];
    float local_radius;
    const Occlusion* occ;
    bool blocks;
};

// Shadows mode: may an occluded sphere still shadow something visible? Each sun's shadow volume
// (the sphere swept along the light by its reach) must be occluded too; a point or spot lamp keeps it
// when in range (its shadow can fall anywhere around it).
bool shadow_unoccluded(const InstanceJob& J, const float* c, float r, float copy_r) {
    for (int l = 0; l < J.nl; ++l) {
        const vgeo_shadow_light& L = J.lights[l];
        if (L.kind == 0) {
            float reach = L.reach > 0.f ? L.reach : 2.f * copy_r / std::max(-L.dir[2], 0.1f);
            float sc[3] = {c[0] + L.dir[0] * reach * 0.5f, c[1] + L.dir[1] * reach * 0.5f,
                           c[2] + L.dir[2] * reach * 0.5f};
            if (!sphere_occluded(*J.occ, sc, r + reach * 0.5f)) return true;
        } else {
            float dx = c[0] - L.pos[0], dy = c[1] - L.pos[1], dz = c[2] - L.pos[2];
            if (std::sqrt(dx * dx + dy * dy + dz * dz) - r <= L.range) return true;
        }
    }
    return false;
}

void select_range(const InstanceJob& J, int lo, int hi, std::vector<uint32_t>& pairs, uint64_t& tris,
                  uint32_t& drawn) {
    const Asset& a = *J.a;
    const int G = int(a.h.group_count);
    std::vector<uint8_t> pass(static_cast<size_t>(G));
    const bool use_blocks = J.blocks && !a.blocks.empty();
    std::vector<uint32_t> inc;                                   // atoms in this copy's cut
    std::vector<uint32_t> bcount(use_blocks ? a.blocks.size() : 0, 0);
    const bool cull = J.mode == VGEO_OFFSCREEN_CULL || J.mode == VGEO_OFFSCREEN_SHADOWS;
    const bool coarsen = (J.mode == VGEO_OFFSCREEN_COARSEN || J.mode == VGEO_OFFSCREEN_SHADOWS) &&
                         J.offscreen_scale > 1.f;
    const bool shadows = J.mode == VGEO_OFFSCREEN_SHADOWS && J.nl > 0;
    for (int i = lo; i < hi; ++i) {
        if (J.mask && !J.mask[i]) continue;
        const float* M = J.transforms + size_t(i) * 12;
        const float s = J.scales[i];
        float cc[3];
        xform(M, J.local_center, cc);
        const float cr = J.local_radius * s;
        if (cull && !inside_any(J.views, J.nv, cc, cr) &&
            !(shadows && shadow_reaches(J.views, J.nv, J.lights, J.nl, cc, cr, cr)))
            continue;
        // occlusion (one view): a hidden copy is dropped (Cull), kept coarsened when its shadow may
        // still show (Shadows), or coarsened (Coarsen)
        const bool hidden = J.occ && sphere_occluded(*J.occ, cc, cr);
        if (hidden && J.mode == VGEO_OFFSCREEN_CULL) continue;
        if (hidden && J.mode == VGEO_OFFSCREEN_SHADOWS && !shadow_unoccluded(J, cc, cr, cr)) continue;
        // a copy entirely inside a view has every group and atom inside it too (their spheres
        // contain geometry, which lies in the copy's sphere): no per-group or per-atom frustum test
        uint32_t fully_inside = 0;   // bit per view (first 32)
        for (int v = 0; v < J.nv && v < 32; ++v) {
            bool all = true;
            for (int p = 0; p < 6 && all; ++p) {
                const float* pl = J.views[v].planes[p];
                if (pl[0] * cc[0] + pl[1] * cc[1] + pl[2] * cc[2] + pl[3] < cr) all = false;
            }
            if (all) fully_inside |= 1u << v;
        }
        for (int g = 0; g < G; ++g) {
            const vgeo2::Group& gr = a.groups[g];
            if (!(gr.error < FLT_MAX)) { pass[g] = 0; continue; }
            float wc[3];
            xform(M, gr.center, wc);
            const float wr = gr.radius * s, werr = gr.error * s;
            bool ok = true;
            for (int v = 0; v < J.nv && ok; ++v) {
                const vgeo_view& vw = J.views[v];
                float e;
                if (vw.ortho) {
                    e = werr / std::max(vw.ortho_height, 1e-20f);
                } else {
                    float dx = wc[0] - vw.camera[0], dy = wc[1] - vw.camera[1], dz = wc[2] - vw.camera[2];
                    float d = std::sqrt(dx * dx + dy * dy + dz * dz) - wr;
                    e = werr / std::max(d, vw.znear) * (vw.proj * 0.5f);
                }
                float t = vw.threshold;
                if (hidden && J.offscreen_scale > 1.f)
                    t *= J.offscreen_scale;
                else if (coarsen && !(v < 32 && (fully_inside >> v) & 1u) && !in_frustum(vw, wc, wr))
                    t *= J.offscreen_scale;
                if (e > t) ok = false;
            }
            pass[g] = ok ? 1 : 0;
        }
        bool any = false;
        const uint32_t na = uint32_t(a.atoms.size());
        for (uint32_t k = 0; k < na; ++k) {
            const vgeo_atom& at = a.atoms[k];
            if (pass[at.group] || (at.refined >= 0 && !pass[at.refined])) continue;
            if (cull && !fully_inside) {
                float wc[3];
                xform(M, at.center, wc);
                const float wr = at.radius * s;
                if (!inside_any(J.views, J.nv, wc, wr) &&
                    !(shadows && shadow_reaches(J.views, J.nv, J.lights, J.nl, wc, wr, cr)))
                    continue;
            }
            if (cull && J.occ && !hidden) {
                float wc[3];
                xform(M, at.center, wc);
                const float wr = at.radius * s;
                if (sphere_occluded(*J.occ, wc, wr) && !(shadows && shadow_unoccluded(J, wc, wr, cr))) continue;
            }
            if (use_blocks) {
                inc.push_back(k);
                if (a.atom_block[k] >= 0) ++bcount[a.atom_block[k]];
                continue;
            }
            pairs.push_back(uint32_t(i));
            pairs.push_back(k);
            tris += at.tri_count;
            any = true;
        }
        if (use_blocks) {
            // a block whose atoms are all in the cut is drawn as one piece
            const uint32_t A = na;
            for (uint32_t k : inc) {
                const int32_t b = a.atom_block[k];
                if (b >= 0 && bcount[b] == a.blocks[b].cluster_count) {
                    pairs.push_back(uint32_t(i));
                    pairs.push_back(A + uint32_t(b));
                    tris += a.blocks[b].tri_count;
                    bcount[b] = 0xFFFFFFFFu;   // emitted
                } else if (b < 0 || bcount[b] != 0xFFFFFFFFu) {
                    pairs.push_back(uint32_t(i));
                    pairs.push_back(k);
                    tris += a.atoms[k].tri_count;
                }
            }
            for (uint32_t k : inc)
                if (a.atom_block[k] >= 0) bcount[a.atom_block[k]] = 0;
            any = !inc.empty();
            inc.clear();
        }
        if (any) ++drawn;
    }
}

}  // namespace

extern "C" VGEO_API int vgeo_atoms(void* handle, vgeo_atom* out, int max_atoms) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a) return -1;
    build_atoms(*a);
    const int n = int(a->atoms.size());
    if (out && max_atoms > 0) std::copy(a->atoms.begin(), a->atoms.begin() + std::min(n, max_atoms), out);
    return n;
}

extern "C" VGEO_API int vgeo_extract_atom(void* handle, uint32_t atom, vgeo_chunk_data* out) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !out) return 1;
    build_atoms(*a);
    const uint32_t A = uint32_t(a->atoms.size());
    if (atom < A) {
        const vgeo_atom& at = a->atoms[atom];
        extract_clusters(a, a->atom_clusters.data() + at.cluster_offset, at.cluster_count, out);
        return 0;
    }
    if (atom - A >= a->blocks.size()) return 1;
    const vgeo_atom& b = a->blocks[atom - A];
    a->extract_ids.clear();
    for (uint32_t j = 0; j < b.cluster_count; ++j) {
        const vgeo_atom& at = a->atoms[a->block_atoms[b.cluster_offset + j]];
        a->extract_ids.insert(a->extract_ids.end(), a->atom_clusters.begin() + at.cluster_offset,
                              a->atom_clusters.begin() + at.cluster_offset + at.cluster_count);
    }
    std::vector<uint32_t> ids(a->extract_ids);   // extract_clusters does not use extract_ids, but be safe
    extract_clusters(a, ids.data(), ids.size(), out);
    return 0;
}

extern "C" VGEO_API int vgeo_blocks(void* handle, vgeo_atom* out, int max_blocks) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a) return -1;
    build_atoms(*a);
    const int n = int(a->blocks.size());
    if (out && max_blocks > 0) std::copy(a->blocks.begin(), a->blocks.begin() + std::min(n, max_blocks), out);
    return n;
}

extern "C" VGEO_API int vgeo_select_instances(void* handle, const vgeo_view* views, int view_count,
                                              const float* transforms, const float* scales,
                                              const uint8_t* copy_mask, int copy_count, int mode,
                                              float offscreen_scale, const vgeo_shadow_light* lights,
                                              int light_count, const void* occlusion, int flags,
                                              uint32_t* out_pairs, uint32_t max_pairs,
                                              vgeo_instances_stats* stats) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !views || view_count <= 0 || copy_count < 0 || (copy_count > 0 && (!transforms || !scales)))
        return 1;
    build_atoms(*a);
    InstanceJob J{};
    J.a = a;
    J.views = views;
    J.nv = view_count;
    J.transforms = transforms;
    J.scales = scales;
    J.mask = copy_mask;
    J.mode = mode;
    J.offscreen_scale = offscreen_scale;
    J.lights = lights;
    J.nl = lights ? light_count : 0;
    J.blocks = (flags & VGEO_SELECT_BLOCKS) != 0;
    J.occ = (occlusion && static_cast<const Occlusion*>(occlusion)->ready && view_count == 1)
                ? static_cast<const Occlusion*>(occlusion) : nullptr;
    float r2 = 0.f;
    for (int d = 0; d < 3; ++d) {
        J.local_center[d] = 0.5f * (a->h.aabb_min[d] + a->h.aabb_max[d]);
        float h = 0.5f * (a->h.aabb_max[d] - a->h.aabb_min[d]);
        r2 += h * h;
    }
    J.local_radius = std::sqrt(r2) * 1.0001f;

    // copies split over threads; each fills its own list, joined in copy order
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int threads = int(std::min<unsigned>(hw, 16u));
    const uint64_t work = uint64_t(copy_count) * (a->h.group_count + a->atoms.size());
    if (work < 200000) threads = 1;
    threads = std::max(1, std::min(threads, copy_count));
    std::vector<std::vector<uint32_t>> pairs(static_cast<size_t>(threads));
    std::vector<uint64_t> tris(static_cast<size_t>(threads), 0);
    std::vector<uint32_t> drawn(static_cast<size_t>(threads), 0);
    if (threads == 1) {
        select_range(J, 0, copy_count, pairs[0], tris[0], drawn[0]);
    } else {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            int lo = int(int64_t(copy_count) * t / threads), hi = int(int64_t(copy_count) * (t + 1) / threads);
            pool.emplace_back([&, t, lo, hi] { select_range(J, lo, hi, pairs[t], tris[t], drawn[t]); });
        }
        for (auto& th : pool) th.join();
    }
    uint64_t total_pairs = 0, total_tris = 0;
    uint32_t total_drawn = 0;
    for (int t = 0; t < threads; ++t) {
        total_pairs += pairs[t].size() / 2;
        total_tris += tris[t];
        total_drawn += drawn[t];
    }
    if (out_pairs) {
        uint64_t at = 0;
        for (int t = 0; t < threads && at < max_pairs; ++t) {
            uint64_t n = std::min<uint64_t>(pairs[t].size() / 2, max_pairs - at);
            if (n) std::memcpy(out_pairs + at * 2, pairs[t].data(), size_t(n) * 8);
            at += n;
        }
    }
    if (stats) {
        stats->triangles = total_tris;
        stats->pairs = uint32_t(std::min<uint64_t>(total_pairs, 0xFFFFFFFFu));
        stats->copies_drawn = total_drawn;
    }
    return (out_pairs && total_pairs > max_pairs) ? 2 : 0;
}

// ---------------------------------------------------------------- web export

namespace {

struct WebHeader {
    char magic[8];            // "VGEOW\0\0\0"
    uint32_t version;         // 1
    uint32_t header_size;     // 128
    uint32_t cluster_count;
    uint32_t group_count;
    uint32_t vertex_count;    // cluster-local vertices over all clusters
    uint32_t tri_count;
    uint32_t material_count;
    uint32_t lod_levels;
    uint32_t source_triangles;
    uint32_t max_cluster_tris;
    float aabb_min[3];
    float aabb_max[3];
    float grid_origin[3];
    float grid_step;
    uint32_t off_clusters;    // 12 u32 per cluster
    uint32_t off_groups;      // 8 u32 per group (same as v2)
    // vertex stream: 4 u32 per vertex (x, y, z on the 21-bit grid, oct normal snorm16 x 2)
    // triangle stream: 1 u32 per triangle (i0 | i1 << 8 | i2 << 16 | material << 24)
    // both meshopt-encoded (meshopt_encodeVertexBuffer, vertex size 16 and 4)
    uint32_t off_vertices;
    uint32_t off_triangles;
    uint32_t off_materials;   // same layout as v2 (names, optional MATP)
    uint32_t file_size;
    uint32_t vertex_bytes;    // encoded sizes
    uint32_t triangle_bytes;
    uint32_t flags;           // bit 0: streams are meshopt-encoded
    uint32_t reserved;
};
static_assert(sizeof(WebHeader) == 128, "WebHeader layout");

inline uint32_t oct_encode(float x, float y, float z) {
    float s = std::fabs(x) + std::fabs(y) + std::fabs(z);
    if (s <= 0.f) return 0;
    x /= s;
    y /= s;
    if (z < 0.f) {
        float ox = (1.f - std::fabs(y)) * (x >= 0.f ? 1.f : -1.f);
        float oy = (1.f - std::fabs(x)) * (y >= 0.f ? 1.f : -1.f);
        x = ox;
        y = oy;
    }
    auto q = [](float v) {
        long i = std::lround(std::max(-1.f, std::min(1.f, v)) * 32767.f);
        return uint32_t(uint16_t(int16_t(i)));
    };
    return q(x) | (q(y) << 16);
}

}  // namespace

extern "C" VGEO_API int vgeo_export_web(void* handle, const char* path_utf8, uint64_t* out_bytes,
                                        char* err, int err_len) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !path_utf8) { set_err(err, err_len, "no asset or path"); return 1; }
    const vgeo2::Header& h = a->h;
    if (a->materials.size() > 256) { set_err(err, err_len, "web format supports at most 256 materials"); return 1; }

    // one global grid: a vertex shared by neighbouring clusters quantizes to the same value in both
    const float kMax = float((1u << 21) - 1);
    float extent = 0.f;
    for (int k = 0; k < 3; ++k) extent = std::max(extent, h.aabb_max[k] - h.aabb_min[k]);
    const float step = extent > 0.f ? extent / kMax : 1.f;

    std::vector<uint32_t> clusters(size_t(h.cluster_count) * 12, 0);
    std::vector<uint32_t> verts;
    std::vector<uint32_t> tris;
    verts.reserve(size_t(h.index_count) * 4 / 2);
    tris.reserve(h.index_count / 3);
    std::vector<int32_t> local(h.vertex_count, -1);
    std::vector<uint32_t> touched;
    uint32_t max_tris = 0;

    for (uint32_t i = 0; i < h.cluster_count; ++i) {
        const vgeo2::Cluster& c = a->clusters[i];
        if (c.tri_count > 256) { set_err(err, err_len, "cluster larger than 256 triangles"); return 1; }
        if (!cluster_ok(*a, i)) { set_err(err, err_len, "corrupt file (bad index)"); return 1; }
        const uint32_t vtx_off = uint32_t(verts.size() / 4);
        const uint32_t tri_off = uint32_t(tris.size());
        touched.clear();
        const uint32_t* idx = a->indices + c.index_offset;
        for (uint32_t t = 0; t < c.tri_count; ++t) {
            uint32_t lv[3];
            for (int j = 0; j < 3; ++j) {
                uint32_t v = idx[t * 3 + j];
                if (local[v] < 0) {
                    local[v] = int32_t(touched.size());
                    touched.push_back(v);
                    const float* p = a->positions + size_t(v) * 3;
                    uint32_t q[3];
                    for (int k = 0; k < 3; ++k) {
                        float f = (p[k] - h.aabb_min[k]) / step;
                        q[k] = uint32_t(std::lround(std::max(0.f, std::min(kMax, f))));
                    }
                    const float* n = a->normals + size_t(v) * 3;
                    verts.push_back(q[0]);
                    verts.push_back(q[1]);
                    verts.push_back(q[2]);
                    verts.push_back(oct_encode(n[0], n[1], n[2]));
                }
                lv[j] = uint32_t(local[v]);
            }
            if (touched.size() > 256) {
                for (uint32_t v : touched) local[v] = -1;
                set_err(err, err_len, "cluster with more than 256 vertices");
                return 1;
            }
            uint32_t mat = a->vmat[idx[t * 3]];
            tris.push_back(lv[0] | (lv[1] << 8) | (lv[2] << 16) | (std::min(mat, 255u) << 24));
        }
        for (uint32_t v : touched) local[v] = -1;
        uint32_t* o = &clusters[size_t(i) * 12];
        o[0] = vtx_off;
        o[1] = tri_off;
        o[2] = uint32_t(c.group);
        o[3] = uint32_t(c.refined);
        o[4] = uint32_t(touched.size()) | (c.tri_count << 16);
        o[5] = c.depth;
        std::memcpy(&o[6], c.center, 12);
        std::memcpy(&o[9], &c.radius, 4);
        max_tris = std::max(max_tris, c.tri_count);
    }

    // materials section copied from the v2 file (names + optional MATP), without tail padding
    const uint8_t* mat_begin = a->base + h.off_materials;
    const uint8_t* mat_end = a->base + std::min<uint64_t>(h.file_size, a->size);
    std::vector<uint8_t> mat_section(mat_begin, mat_end);
    while (!mat_section.empty() && mat_section.back() == 0) mat_section.pop_back();

    WebHeader w = {};
    std::memcpy(w.magic, "VGEOW\0\0\0", 8);
    w.version = 1;
    w.header_size = sizeof(WebHeader);
    w.cluster_count = h.cluster_count;
    w.group_count = h.group_count;
    w.vertex_count = uint32_t(verts.size() / 4);
    w.tri_count = uint32_t(tris.size());
    w.material_count = uint32_t(a->materials.size());
    w.lod_levels = h.lod_levels;
    w.source_triangles = h.source_triangles;
    w.max_cluster_tris = max_tris;
    std::memcpy(w.aabb_min, h.aabb_min, 12);
    std::memcpy(w.aabb_max, h.aabb_max, 12);
    std::memcpy(w.grid_origin, h.aabb_min, 12);
    w.grid_step = step;

    // meshopt vertex codec on both streams (byte-wise deltas + entropy coding; decoded in the browser)
    auto encode = [](const std::vector<uint32_t>& data, size_t stride_bytes) {
        const size_t count = data.size() * 4 / stride_bytes;
        std::vector<uint8_t> enc(meshopt_encodeVertexBufferBound(count, stride_bytes));
        enc.resize(meshopt_encodeVertexBufferLevel(enc.data(), enc.size(), data.data(), count, stride_bytes, 3, 1));
        return enc;
    };
    const std::vector<uint8_t> venc = encode(verts, 16);
    const std::vector<uint8_t> tenc = encode(tris, 4);
    w.vertex_bytes = uint32_t(venc.size());
    w.triangle_bytes = uint32_t(tenc.size());
    w.flags = 1;

    uint64_t off = sizeof(WebHeader);
    auto place = [&](uint64_t bytes) { uint64_t o = off; off = vgeo2::align16(off + bytes); return o; };
    const uint64_t oc = place(clusters.size() * 4);
    const uint64_t og = place(uint64_t(h.group_count) * 32);
    const uint64_t ov = place(venc.size());
    const uint64_t ot = place(tenc.size());
    const uint64_t om = place(mat_section.size());
    if (off > 0xFFFFFFF0ull) { set_err(err, err_len, "asset too large for the web format (4 GB)"); return 1; }
    w.off_clusters = uint32_t(oc);
    w.off_groups = uint32_t(og);
    w.off_vertices = uint32_t(ov);
    w.off_triangles = uint32_t(ot);
    w.off_materials = uint32_t(om);
    w.file_size = uint32_t(off);

    std::vector<uint8_t> out(size_t(off), 0);
    std::memcpy(out.data(), &w, sizeof(w));
    std::memcpy(out.data() + oc, clusters.data(), clusters.size() * 4);
    std::memcpy(out.data() + og, a->groups, size_t(h.group_count) * 32);
    std::memcpy(out.data() + ov, venc.data(), venc.size());
    std::memcpy(out.data() + ot, tenc.data(), tenc.size());
    if (!mat_section.empty()) std::memcpy(out.data() + om, mat_section.data(), mat_section.size());

    std::string tmp = std::string(path_utf8) + ".part";
    FILE* f = vgeo_io::open_utf8(tmp.c_str(), "wb");
    if (!f) { set_err(err, err_len, "cannot open for writing: " + tmp); return 1; }
    bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok || !vgeo_io::replace_utf8(tmp.c_str(), path_utf8)) {
        vgeo_io::remove_utf8(tmp.c_str());
        set_err(err, err_len, "write failed");
        return 1;
    }
    if (out_bytes) *out_bytes = out.size();
    return 0;
}

// ---------------------------------------------------------------- paged web export
//
// .vgeow version 2: the same clusters and streams, split into pages that a viewer
// can fetch with HTTP range requests. Every page holds whole groups (a group's
// members are either all loaded or not), pages are ordered coarse to fine, and
// groups are renumbered in page order so a page covers one contiguous group range.
//
//   header (160 bytes) | clusters | groups | page table | materials  <- the "head",
//   fetched first in one request | page 0 | page 1 | ...
//
// A viewer treats a group whose page is missing as fine enough, so it draws the
// coarser clusters made from it instead: the cut stays valid (and crack-free)
// while pages arrive, provided the coarsest groups are loaded first.

namespace {

struct WebHeaderV2 {
    char magic[8];            // "VGEOW\0\0\0"
    uint32_t version;         // 2
    uint32_t header_size;     // 160
    uint32_t cluster_count;
    uint32_t group_count;
    uint32_t vertex_count;
    uint32_t tri_count;
    uint32_t material_count;
    uint32_t lod_levels;
    uint32_t source_triangles;
    uint32_t max_cluster_tris;
    float aabb_min[3];
    float aabb_max[3];
    float grid_origin[3];
    float grid_step;
    uint32_t off_clusters;    // 12 u32 per cluster, in page order
    uint32_t off_groups;      // 8 u32 per group, renumbered in page order
    uint32_t off_pages;       // 12 u32 per page (below)
    uint32_t page_count;
    uint32_t off_materials;
    uint32_t head_bytes;      // everything before the first page
    uint64_t file_size;
    uint32_t flags;           // bit 0: page streams are meshopt-encoded; bit 1: cluster and group tables too
    uint32_t root_pages;      // leading pages that hold every terminal group (load with the head)
    uint32_t cluster_bytes;   // with flag bit 1: encoded size of the cluster table (else 0)
    uint32_t group_bytes;     // with flag bit 1: encoded size of the group table (else 0)
    uint32_t reserved[6];
};
static_assert(sizeof(WebHeaderV2) == 160, "WebHeaderV2 layout");

// page table entry: group first, group count, cluster first, cluster count,
// vertex first, vertex count, triangle first, triangle count,
// data offset (low, high), vertex bytes, triangle bytes
constexpr uint32_t kPageU32 = 12;

inline uint32_t part1by2(uint32_t x) {
    x &= 0x3FF;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    return x;
}

}  // namespace

extern "C" VGEO_API int vgeo_export_web_paged(void* handle, const char* path_utf8, uint32_t page_vertices,
                                              uint64_t* out_bytes, char* err, int err_len) {
    Asset* a = static_cast<Asset*>(handle);
    if (!a || !path_utf8) { set_err(err, err_len, "no asset or path"); return 1; }
    const vgeo2::Header& h = a->h;
    if (a->materials.size() > 256) { set_err(err, err_len, "web format supports at most 256 materials"); return 1; }
    if (page_vertices == 0) page_vertices = 4096;

    const float kMax = float((1u << 21) - 1);
    float extent = 0.f;
    for (int k = 0; k < 3; ++k) extent = std::max(extent, h.aabb_max[k] - h.aabb_min[k]);
    const float step = extent > 0.f ? extent / kMax : 1.f;

    // members of each group, and the group order: coarse (deep) first, then along a Morton curve
    std::vector<std::vector<uint32_t>> members(h.group_count);
    for (uint32_t i = 0; i < h.cluster_count; ++i) {
        if (!cluster_ok(*a, i)) { set_err(err, err_len, "corrupt file (bad index)"); return 1; }
        if (a->clusters[i].tri_count > 256) { set_err(err, err_len, "cluster larger than 256 triangles"); return 1; }
        members[uint32_t(a->clusters[i].group)].push_back(i);
    }
    std::vector<uint32_t> order;
    std::vector<uint64_t> key(h.group_count);
    for (uint32_t g = 0; g < h.group_count; ++g) {
        if (members[g].empty()) continue;
        const vgeo2::Group& gr = a->groups[g];
        uint32_t q[3];
        for (int k = 0; k < 3; ++k) {
            float t = extent > 0.f ? (gr.center[k] - h.aabb_min[k]) / extent : 0.f;
            q[k] = uint32_t(std::max(0.f, std::min(1023.f, t * 1023.f)));
        }
        uint32_t morton = part1by2(q[0]) | (part1by2(q[1]) << 1) | (part1by2(q[2]) << 2);
        // terminal groups (never simplified further) first: nothing coarser can stand in for them,
        // so they must load with the head. Then deepest first: a parents-before-children order.
        const uint64_t nonterminal = gr.error < FLT_MAX ? 1 : 0;
        const uint64_t depth = uint64_t(std::min<int32_t>(std::max<int32_t>(gr.depth, 0), 0x3FFFFFFF));
        key[g] = (nonterminal << 62) | ((0x3FFFFFFFull - depth) << 30) | morton;
        order.push_back(g);
    }
    uint32_t terminal_groups = 0;
    for (uint32_t g : order) terminal_groups += a->groups[g].error < FLT_MAX ? 0 : 1;
    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return key[x] < key[y]; });
    std::vector<int32_t> new_group(h.group_count, -1);
    for (uint32_t i = 0; i < order.size(); ++i) new_group[order[i]] = int32_t(i);

    // pages: whole groups, closed once they reach page_vertices (cluster-local vertices)
    struct Page { uint32_t g0, gn, c0, cn, v0, vn, t0, tn; std::vector<uint8_t> venc, tenc; };
    std::vector<Page> pages;
    std::vector<uint32_t> clusters;                 // 12 u32 each, new order
    std::vector<uint32_t> verts, tris;              // current page's raw streams
    std::vector<int32_t> local(h.vertex_count, -1);
    std::vector<uint32_t> touched;
    uint32_t vtotal = 0, ttotal = 0, max_tris = 0;
    auto encode = [](const std::vector<uint32_t>& data, size_t stride_bytes) {
        const size_t count = data.size() * 4 / stride_bytes;
        std::vector<uint8_t> enc(meshopt_encodeVertexBufferBound(count, stride_bytes));
        enc.resize(meshopt_encodeVertexBufferLevel(enc.data(), enc.size(), data.data(), count, stride_bytes, 3, 1));
        return enc;
    };
    Page cur = {};
    auto close_page = [&]() {
        cur.vn = vtotal - cur.v0;
        cur.tn = ttotal - cur.t0;
        cur.cn = uint32_t(clusters.size() / 12) - cur.c0;
        cur.venc = encode(verts, 16);
        cur.tenc = encode(tris, 4);
        const uint32_t g_next = cur.g0 + cur.gn;
        pages.push_back(std::move(cur));
        cur = {};
        cur.g0 = g_next;
        cur.c0 = uint32_t(clusters.size() / 12);
        cur.v0 = vtotal;
        cur.t0 = ttotal;
        verts.clear();
        tris.clear();
    };
    for (uint32_t gi = 0; gi < order.size(); ++gi) {
        for (uint32_t id : members[order[gi]]) {
            const vgeo2::Cluster& c = a->clusters[id];
            const uint32_t vtx_off = vtotal, tri_off = ttotal;
            touched.clear();
            const uint32_t* idx = a->indices + c.index_offset;
            for (uint32_t t = 0; t < c.tri_count; ++t) {
                uint32_t lv[3];
                for (int j = 0; j < 3; ++j) {
                    uint32_t v = idx[t * 3 + j];
                    if (local[v] < 0) {
                        local[v] = int32_t(touched.size());
                        touched.push_back(v);
                        const float* p = a->positions + size_t(v) * 3;
                        for (int k = 0; k < 3; ++k) {
                            float f = (p[k] - h.aabb_min[k]) / step;
                            verts.push_back(uint32_t(std::lround(std::max(0.f, std::min(kMax, f)))));
                        }
                        const float* n = a->normals + size_t(v) * 3;
                        verts.push_back(oct_encode(n[0], n[1], n[2]));
                        ++vtotal;
                    }
                    lv[j] = uint32_t(local[v]);
                }
                if (touched.size() > 256) {
                    for (uint32_t v : touched) local[v] = -1;
                    set_err(err, err_len, "cluster with more than 256 vertices");
                    return 1;
                }
                uint32_t mat = a->vmat[idx[t * 3]];
                tris.push_back(lv[0] | (lv[1] << 8) | (lv[2] << 16) | (std::min(mat, 255u) << 24));
                ++ttotal;
            }
            for (uint32_t v : touched) local[v] = -1;
            uint32_t o[12] = {};
            o[0] = vtx_off;
            o[1] = tri_off;
            o[2] = uint32_t(new_group[uint32_t(c.group)]);
            o[3] = c.refined < 0 ? 0xFFFFFFFFu : uint32_t(new_group[uint32_t(c.refined)]);
            o[4] = uint32_t(touched.size()) | (c.tri_count << 16);
            o[5] = c.depth;
            std::memcpy(&o[6], c.center, 12);
            std::memcpy(&o[9], &c.radius, 4);
            clusters.insert(clusters.end(), o, o + 12);
            max_tris = std::max(max_tris, c.tri_count);
        }
        cur.gn += 1;
        if (vtotal - cur.v0 >= page_vertices) close_page();
    }
    if (cur.gn) close_page();

    std::vector<uint32_t> groups(size_t(order.size()) * 8);
    for (uint32_t i = 0; i < order.size(); ++i) std::memcpy(&groups[size_t(i) * 8], &a->groups[order[i]], 32);

    const uint8_t* mat_begin = a->base + h.off_materials;
    const uint8_t* mat_end = a->base + std::min<uint64_t>(h.file_size, a->size);
    std::vector<uint8_t> mat_section(mat_begin, mat_end);
    while (!mat_section.empty() && mat_section.back() == 0) mat_section.pop_back();

    WebHeaderV2 w = {};
    std::memcpy(w.magic, "VGEOW\0\0\0", 8);
    w.version = 2;
    w.header_size = sizeof(WebHeaderV2);
    w.cluster_count = uint32_t(clusters.size() / 12);
    w.group_count = uint32_t(order.size());
    w.vertex_count = vtotal;
    w.tri_count = ttotal;
    w.material_count = uint32_t(a->materials.size());
    w.lod_levels = h.lod_levels;
    w.source_triangles = h.source_triangles;
    w.max_cluster_tris = max_tris;
    std::memcpy(w.aabb_min, h.aabb_min, 12);
    std::memcpy(w.aabb_max, h.aabb_max, 12);
    std::memcpy(w.grid_origin, h.aabb_min, 12);
    w.grid_step = step;
    w.page_count = uint32_t(pages.size());
    w.flags = 1;
    // the head is fetched before anything draws, so its tables are encoded too (lossless, the same
    // vertex codec as the pages: clusters are 48-byte records, groups 32-byte records)
    std::vector<uint8_t> cenc = encode(clusters, 48), genc = encode(groups, 32);
    const bool pack_head = !clusters.empty() && !groups.empty();
    if (pack_head) {
        w.flags |= 2;
        w.cluster_bytes = uint32_t(cenc.size());
        w.group_bytes = uint32_t(genc.size());
    }
    // root pages: the leading pages that hold every terminal group (load them with the head)
    uint32_t root_pages = 0;
    while (root_pages < pages.size() && pages[root_pages].g0 < terminal_groups) ++root_pages;
    w.root_pages = root_pages;

    uint64_t off = sizeof(WebHeaderV2);
    auto place = [&](uint64_t bytes) { uint64_t o = off; off = vgeo2::align16(off + bytes); return o; };
    w.off_clusters = uint32_t(place(pack_head ? cenc.size() : clusters.size() * 4));
    w.off_groups = uint32_t(place(pack_head ? genc.size() : groups.size() * 4));
    w.off_pages = uint32_t(place(uint64_t(pages.size()) * kPageU32 * 4));
    w.off_materials = uint32_t(place(mat_section.size()));
    if (off > 0xFFFFFFF0ull) { set_err(err, err_len, "head too large for the web format"); return 1; }
    w.head_bytes = uint32_t(off);
    std::vector<uint32_t> table(size_t(pages.size()) * kPageU32, 0);
    for (size_t p = 0; p < pages.size(); ++p) {
        const Page& pg = pages[p];
        uint64_t data = off;
        off = vgeo2::align16(off + pg.venc.size() + pg.tenc.size());
        uint32_t* e = &table[p * kPageU32];
        e[0] = pg.g0; e[1] = pg.gn; e[2] = pg.c0; e[3] = pg.cn;
        e[4] = pg.v0; e[5] = pg.vn; e[6] = pg.t0; e[7] = pg.tn;
        e[8] = uint32_t(data & 0xFFFFFFFFu); e[9] = uint32_t(data >> 32);
        e[10] = uint32_t(pg.venc.size()); e[11] = uint32_t(pg.tenc.size());
    }
    w.file_size = off;

    std::string tmp = std::string(path_utf8) + ".part";
    FILE* f = vgeo_io::open_utf8(tmp.c_str(), "wb");
    if (!f) { set_err(err, err_len, "cannot open for writing: " + tmp); return 1; }
    bool ok = true;
    uint64_t pos = 0;
    auto put = [&](uint64_t at, const void* src, size_t n) {
        static const uint8_t zeros[16] = {};
        while (ok && pos < at) {
            size_t pad = size_t(std::min<uint64_t>(16, at - pos));
            ok = std::fwrite(zeros, 1, pad, f) == pad;
            pos += pad;
        }
        if (ok && n) ok = std::fwrite(src, 1, n, f) == n;
        pos += n;
    };
    put(0, &w, sizeof(w));
    if (pack_head) {
        put(w.off_clusters, cenc.data(), cenc.size());
        put(w.off_groups, genc.data(), genc.size());
    } else {
        put(w.off_clusters, clusters.data(), clusters.size() * 4);
        put(w.off_groups, groups.data(), groups.size() * 4);
    }
    put(w.off_pages, table.data(), table.size() * 4);
    put(w.off_materials, mat_section.data(), mat_section.size());
    for (size_t p = 0; p < pages.size(); ++p) {
        uint64_t at = uint64_t(table[p * kPageU32 + 8]) | (uint64_t(table[p * kPageU32 + 9]) << 32);
        put(at, pages[p].venc.data(), pages[p].venc.size());
        put(at + pages[p].venc.size(), pages[p].tenc.data(), pages[p].tenc.size());
    }
    put(w.file_size, nullptr, 0);
    ok = (std::fclose(f) == 0) && ok;
    if (!ok || !vgeo_io::replace_utf8(tmp.c_str(), path_utf8)) {
        vgeo_io::remove_utf8(tmp.c_str());
        set_err(err, err_len, "write failed");
        return 1;
    }
    if (out_bytes) *out_bytes = w.file_size;
    return 0;
}
