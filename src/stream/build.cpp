// Triangle soup -> cluster LOD DAG -> .vgeo v2.
//
// The DAG itself comes from meshoptimizer's clusterlod (Nanite-style: group
// clusters, simplify each group with its border locked, re-split, repeat).
// This file welds the input, records the DAG, splits it into spatial chunks
// for incremental streaming, and writes the file.

#include "vgeo_stream.h"
#include "format_v2.h"
#include "io_util.h"

#include "meshoptimizer.h"
#include "clusterlod.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
    float mat;
};

struct OutCluster {
    uint32_t index_offset;
    uint32_t tri_count;
    int32_t group;
    int32_t refined;
    uint32_t depth;
    float center[3];
    float radius;
};

// Simplification can leave "fins": two triangles on the same three positions with opposite winding
// (a zero-volume sliver, often across a normal or UV seam). Renderers show them as dark specks.
// A free fin adds two uses to each of its edges, so dropping both triangles keeps the surface closed;
// a fin plugging a hole (its edges otherwise used once) keeps the side that agrees with its neighbours;
// a fin whose edges disagree (rare) stays, as removing either side would open an edge.
// pos_id: per vertex, an id shared by vertices at the same position. Marks the fins' triangles in
// drop (one byte per triangle) and returns how many.
size_t find_fins(const std::vector<uint32_t>& tris, const std::vector<uint32_t>& pos_id, std::vector<uint8_t>& drop) {
    const size_t n = tris.size() / 3;
    drop.assign(n, 0);
    if (n < 2) return 0;
    struct Key { uint32_t a, b, c; };
    std::unordered_map<uint64_t, std::vector<uint32_t>> by_key;   // hash of the sorted triple -> triangles
    by_key.reserve(n * 2);
    std::vector<Key> keys(n);
    std::vector<uint8_t> parity(n);
    for (size_t t = 0; t < n; ++t) {
        uint32_t p[3] = {pos_id[tris[t * 3]], pos_id[tris[t * 3 + 1]], pos_id[tris[t * 3 + 2]]};
        // orientation: parity of the permutation that sorts the triple
        uint8_t swaps = 0;
        if (p[0] > p[1]) { std::swap(p[0], p[1]); ++swaps; }
        if (p[1] > p[2]) { std::swap(p[1], p[2]); ++swaps; }
        if (p[0] > p[1]) { std::swap(p[0], p[1]); ++swaps; }
        keys[t] = {p[0], p[1], p[2]};
        parity[t] = swaps & 1;
        if (p[0] == p[1] || p[1] == p[2]) continue;   // degenerate: not a fin
        uint64_t h = (uint64_t(p[0]) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(p[1]) * 0xC2B2AE3D27D4EB4Full) ^
                     (uint64_t(p[2]) * 0x165667B19E3779F9ull);
        by_key[h].push_back(uint32_t(t));
    }
    // directed edge uses (position ids) of every triangle, to tell a free fin from one that plugs a hole
    std::unordered_map<uint64_t, int> directed;
    directed.reserve(n * 4);
    auto pid = [&](size_t t, int k) { return pos_id[tris[t * 3 + k]]; };
    auto dkey = [](uint32_t u, uint32_t v) { return (uint64_t(u) << 32) | v; };
    for (size_t t = 0; t < n; ++t)
        for (int k = 0; k < 3; ++k) ++directed[dkey(pid(t, k), pid(t, (k + 1) % 3))];
    size_t removed = 0;
    for (auto& kv : by_key) {
        std::vector<uint32_t>& list = kv.second;
        if (list.size() < 2) continue;
        for (size_t i = 0; i < list.size(); ++i) {
            const uint32_t a = list[i];
            if (drop[a]) continue;
            for (size_t j = i + 1; j < list.size(); ++j) {
                const uint32_t b = list[j];
                if (drop[b] || parity[a] == parity[b]) continue;
                if (keys[a].a != keys[b].a || keys[a].b != keys[b].b || keys[a].c != keys[b].c) continue;
                // other triangles on a's edges: a's directed edge u->v is matched by the fin's b (v->u),
                // so neighbours using v->u or u->v beyond the pair tell whether a hole is being plugged
                int odd = 0;
                bool a_clashes = false;
                for (int k = 0; k < 3; ++k) {
                    uint32_t u = pid(a, k), v = pid(a, (k + 1) % 3);
                    int same = directed[dkey(u, v)] - 1;      // without a itself
                    int rev = directed[dkey(v, u)] - 1;       // without b itself
                    if ((same + rev) & 1) ++odd;
                    if (same > 0) a_clashes = true;           // a neighbour already runs u->v: a is reversed
                }
                if (odd == 0) {
                    drop[a] = drop[b] = 1;                    // a free fin: both go
                    removed += 2;
                } else if (odd == 3) {
                    drop[a_clashes ? a : b] = 1;              // plugs a hole: keep the consistent side
                    removed += 1;
                }   // mixed: dropping either would open an edge; leave the pair as it is
                break;
            }
        }
    }
    return removed;
}

void set_err(char* err, int err_len, const std::string& msg) {
    if (err && err_len > 0) {
        std::snprintf(err, size_t(err_len), "%s", msg.c_str());
    }
}

// Splits leaf-cluster centers into `leaves` spatial cells (median kd-tree),
// then any cluster can be assigned by descending with its own center.
struct KdTree {
    struct Node {
        int axis = -1;        // -1 = leaf
        float split = 0.f;
        int left = -1, right = -1;
        int chunk = -1;
    };
    std::vector<Node> nodes;
    int chunk_count = 0;

    int build(std::vector<uint32_t>& items, size_t begin, size_t end, int leaves,
              const std::vector<OutCluster>& clusters) {
        int id = int(nodes.size());
        nodes.push_back(Node());
        if (leaves <= 1 || end - begin < 2) {
            nodes[id].chunk = chunk_count++;
            return id;
        }
        float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (size_t i = begin; i < end; ++i) {
            const float* c = clusters[items[i]].center;
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], c[k]);
                hi[k] = std::max(hi[k], c[k]);
            }
        }
        int axis = 0;
        for (int k = 1; k < 3; ++k)
            if (hi[k] - lo[k] > hi[axis] - lo[axis]) axis = k;
        int left_leaves = leaves / 2;
        size_t mid = begin + (end - begin) * size_t(left_leaves) / size_t(leaves);
        mid = std::max(begin + 1, std::min(mid, end - 1));
        std::nth_element(items.begin() + begin, items.begin() + mid, items.begin() + end,
                         [&](uint32_t a, uint32_t b) { return clusters[a].center[axis] < clusters[b].center[axis]; });
        float split = clusters[items[mid]].center[axis];
        int l = build(items, begin, mid, left_leaves, clusters);
        int r = build(items, mid, end, leaves - left_leaves, clusters);
        nodes[id].axis = axis;
        nodes[id].split = split;
        nodes[id].left = l;
        nodes[id].right = r;
        return id;
    }

    int find(const float* p) const {
        int n = 0;
        while (nodes[n].axis >= 0)
            n = p[nodes[n].axis] < nodes[n].split ? nodes[n].left : nodes[n].right;
        return nodes[n].chunk;
    }
};

}  // namespace

extern "C" VGEO_API int vgeo_version(void) { return VGEO_STREAM_VERSION; }

extern "C" VGEO_API int vgeo_build(const vgeo_build_input* in, const char* path_utf8,
                                   vgeo_progress_fn progress, void* user,
                                   vgeo_build_stats* stats, char* err, int err_len) {
    auto t0 = std::chrono::steady_clock::now();
    if (!in || !in->positions || in->tri_count == 0 || !path_utf8) {
        set_err(err, err_len, "empty input");
        return 1;
    }
    if (uint64_t(in->tri_count) * 3 > 0xFFFFFFF0ull / 2) {
        set_err(err, err_len, "mesh too large (index count overflow)");
        return 1;
    }
    auto report = [&](int stage, float p) -> bool { return progress && progress(user, stage, p) != 0; };

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    const size_t corner_count = size_t(in->tri_count) * 3;
    if (report(0, 0.f)) { set_err(err, err_len, "cancelled"); return 2; }

    if (in->indices) {
        // ---- indexed input: vertices as given, split only where materials meet
        const size_t vc = in->vertex_count;
        if (vc == 0) { set_err(err, err_len, "indexed input without vertices"); return 1; }
        for (size_t i = 0; i < corner_count; ++i)
            if (in->indices[i] >= vc) { set_err(err, err_len, "index out of range"); return 1; }
        vertices.resize(vc);
        for (size_t i = 0; i < vc; ++i) {
            Vertex& v = vertices[i];
            v.px = in->positions[i * 3 + 0];
            v.py = in->positions[i * 3 + 1];
            v.pz = in->positions[i * 3 + 2];
            v.nx = in->normals ? in->normals[i * 3 + 0] : 0.f;
            v.ny = in->normals ? in->normals[i * 3 + 1] : 0.f;
            v.nz = in->normals ? in->normals[i * 3 + 2] : 1.f;
            v.u = in->uvs ? in->uvs[i * 2 + 0] : 0.f;
            v.v = in->uvs ? in->uvs[i * 2 + 1] : 0.f;
            v.mat = -1.f;  // unassigned
        }
        indices.assign(in->indices, in->indices + corner_count);
        std::vector<std::pair<uint64_t, unsigned>> splits;  // (vertex<<16 | mat) -> new vertex
        for (size_t i = 0; i < corner_count; ++i) {
            unsigned v = indices[i];
            float m = in->materials ? float(in->materials[i / 3]) : 0.f;
            if (vertices[v].mat < 0.f) {
                vertices[v].mat = m;
            } else if (vertices[v].mat != m) {
                splits.push_back({(uint64_t(v) << 16) | uint64_t(m), unsigned(i)});
            }
        }
        if (!splits.empty()) {
            std::sort(splits.begin(), splits.end());
            uint64_t last = ~0ull;
            unsigned nv = 0;
            for (auto& s : splits) {
                if (s.first != last) {
                    last = s.first;
                    Vertex copy = vertices[size_t(s.first >> 16)];
                    copy.mat = float(s.first & 0xFFFF);
                    nv = unsigned(vertices.size());
                    vertices.push_back(copy);
                }
                indices[s.second] = nv;
            }
        }
        for (Vertex& v : vertices)
            if (v.mat < 0.f) v.mat = 0.f;  // unreferenced vertex
        if (!in->normals) {
            // area-weighted smooth normals
            for (Vertex& v : vertices) v.nx = v.ny = v.nz = 0.f;
            for (size_t t = 0; t < corner_count; t += 3) {
                Vertex* c[3] = {&vertices[indices[t]], &vertices[indices[t + 1]], &vertices[indices[t + 2]]};
                float e1[3] = {c[1]->px - c[0]->px, c[1]->py - c[0]->py, c[1]->pz - c[0]->pz};
                float e2[3] = {c[2]->px - c[0]->px, c[2]->py - c[0]->py, c[2]->pz - c[0]->pz};
                float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                for (int k = 0; k < 3; ++k) { c[k]->nx += n[0]; c[k]->ny += n[1]; c[k]->nz += n[2]; }
            }
            for (Vertex& v : vertices) {
                float l = std::sqrt(v.nx * v.nx + v.ny * v.ny + v.nz * v.nz);
                if (l > 0.f) { v.nx /= l; v.ny /= l; v.nz /= l; } else { v.nz = 1.f; }
            }
        }
    } else {
        // ---- corner soup: weld identical corners into shared vertices
        std::vector<Vertex> corners(corner_count);
        for (size_t i = 0; i < corner_count; ++i) {
            Vertex& v = corners[i];
            v.px = in->positions[i * 3 + 0];
            v.py = in->positions[i * 3 + 1];
            v.pz = in->positions[i * 3 + 2];
            if (in->normals) {
                v.nx = in->normals[i * 3 + 0];
                v.ny = in->normals[i * 3 + 1];
                v.nz = in->normals[i * 3 + 2];
            } else {
                v.nx = v.ny = v.nz = 0.f;
            }
            v.u = in->uvs ? in->uvs[i * 2 + 0] : 0.f;
            v.v = in->uvs ? in->uvs[i * 2 + 1] : 0.f;
            v.mat = in->materials ? float(in->materials[i / 3]) : 0.f;
        }
        if (!in->normals) {
            // flat normals so the file is always complete
            for (size_t t = 0; t < in->tri_count; ++t) {
                Vertex* c = &corners[t * 3];
                float e1[3] = {c[1].px - c[0].px, c[1].py - c[0].py, c[1].pz - c[0].pz};
                float e2[3] = {c[2].px - c[0].px, c[2].py - c[0].py, c[2].pz - c[0].pz};
                float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (l > 0.f) { n[0] /= l; n[1] /= l; n[2] /= l; }
                for (int k = 0; k < 3; ++k) { c[k].nx = n[0]; c[k].ny = n[1]; c[k].nz = n[2]; }
            }
        }
        std::vector<unsigned int> remap(corner_count);
        size_t vertex_count = meshopt_generateVertexRemap(remap.data(), nullptr, corner_count,
                                                          corners.data(), corner_count, sizeof(Vertex));
        vertices.resize(vertex_count);
        meshopt_remapVertexBuffer(vertices.data(), corners.data(), corner_count, sizeof(Vertex), remap.data());
        indices.swap(remap);  // identity index buffer remapped == remap itself
    }

    // drop degenerate triangles (same welded vertex twice) - they break nothing but waste clusters
    {
        size_t w = 0;
        for (size_t i = 0; i < indices.size(); i += 3) {
            unsigned a = indices[i], b = indices[i + 1], c = indices[i + 2];
            if (a == b || b == c || a == c) continue;
            indices[w++] = a; indices[w++] = b; indices[w++] = c;
        }
        indices.resize(w);
    }
    if (indices.empty()) { set_err(err, err_len, "all triangles are degenerate"); return 1; }
    const uint32_t source_tris = uint32_t(indices.size() / 3);
    if (report(0, 1.f)) { set_err(err, err_len, "cancelled"); return 2; }

    // ---- cluster LOD DAG
    const size_t max_tris = in->max_triangles ? std::min<size_t>(256, std::max<size_t>(16, in->max_triangles)) : 128;
    clodConfig config = clodDefaultConfig(max_tris);

    const float attribute_weights[3] = {0.5f, 0.5f, 0.5f};  // normals
    clodMesh mesh = {};
    mesh.indices = indices.data();
    mesh.index_count = indices.size();
    mesh.vertex_count = vertices.size();
    mesh.vertex_positions = &vertices[0].px;
    mesh.vertex_positions_stride = sizeof(Vertex);
    mesh.vertex_attributes = &vertices[0].nx;
    mesh.vertex_attributes_stride = sizeof(Vertex);
    mesh.attribute_weights = attribute_weights;
    mesh.attribute_count = 3;
    // keep hard edges, UV seams and material borders intact (attributes nx ny nz u v mat)
    mesh.attribute_protect_mask = 0x3F;

    // vertices at the same position share an id (vertices are split by normals, UVs, materials)
    std::vector<uint32_t> pos_id(vertices.size());
    {
        std::unordered_map<uint64_t, uint32_t> seen;
        seen.reserve(vertices.size() * 2);
        for (size_t v = 0; v < vertices.size(); ++v) {
            uint32_t bx, by, bz;
            std::memcpy(&bx, &vertices[v].px, 4); std::memcpy(&by, &vertices[v].py, 4); std::memcpy(&bz, &vertices[v].pz, 4);
            uint64_t k = (uint64_t(bx) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(by) * 0xC2B2AE3D27D4EB4Full) ^ uint64_t(bz);
            auto it = seen.find(k);
            if (it != seen.end() && vertices[it->second].px == vertices[v].px && vertices[it->second].py == vertices[v].py &&
                vertices[it->second].pz == vertices[v].pz)
                pos_id[v] = pos_id[it->second];
            else {
                pos_id[v] = uint32_t(v);
                if (it == seen.end()) seen.emplace(k, uint32_t(v));
            }
        }
    }
    uint64_t fins_removed = 0;
    std::vector<uint32_t> group_tris;
    std::vector<size_t> group_tri_owner;
    std::vector<uint8_t> group_drop;

    std::vector<clodGroup> groups;
    std::vector<OutCluster> clusters;
    std::vector<uint32_t> out_indices;
    out_indices.reserve(indices.size() * 2);
    uint64_t emitted_tris = 0;
    bool cancelled = false;

    clodBuild(config, mesh, [&](clodGroup group, const clodCluster* cl, size_t count) -> int {
        int gid = int(groups.size());
        groups.push_back(group);
        // fins: within the clusters of each simplification (same refined group); original geometry
        // (refined < 0) is the source as given and is never touched
        std::unordered_map<int, std::vector<size_t>> by_refined;
        for (size_t i = 0; i < count; ++i)
            if (cl[i].refined >= 0) by_refined[cl[i].refined].push_back(i);
        std::vector<std::vector<uint32_t>> kept(count);
        for (size_t i = 0; i < count; ++i) kept[i].assign(cl[i].indices, cl[i].indices + cl[i].index_count);
        for (auto& kv : by_refined) {
            group_tris.clear();
            group_tri_owner.clear();
            for (size_t i : kv.second)
                for (size_t t = 0; t < kept[i].size() / 3; ++t) {
                    group_tris.insert(group_tris.end(), kept[i].begin() + t * 3, kept[i].begin() + t * 3 + 3);
                    group_tri_owner.push_back(i);
                }
            size_t removed = find_fins(group_tris, pos_id, group_drop);
            if (!removed) continue;
            fins_removed += removed;
            for (size_t i : kv.second) kept[i].clear();
            for (size_t t = 0; t < group_drop.size(); ++t)
                if (!group_drop[t]) {
                    std::vector<uint32_t>& dst = kept[group_tri_owner[t]];
                    dst.insert(dst.end(), group_tris.begin() + t * 3, group_tris.begin() + t * 3 + 3);
                }
        }
        for (size_t i = 0; i < count; ++i) {
            OutCluster oc;
            oc.index_offset = uint32_t(out_indices.size());
            oc.tri_count = uint32_t(kept[i].size() / 3);
            oc.group = gid;
            oc.refined = cl[i].refined;
            oc.depth = uint32_t(group.depth);
            std::memcpy(oc.center, cl[i].bounds.center, sizeof(oc.center));
            oc.radius = cl[i].bounds.radius;
            out_indices.insert(out_indices.end(), kept[i].begin(), kept[i].end());
            clusters.push_back(oc);
            emitted_tris += oc.tri_count;
        }
        // the whole DAG is roughly twice the source; progress is an estimate
        if (!cancelled && report(1, std::min(0.99f, float(double(emitted_tris) / (2.0 * source_tris)))))
            cancelled = true;
        return gid;
    });
    if (cancelled) { set_err(err, err_len, "cancelled"); return 2; }
    if (clusters.empty()) { set_err(err, err_len, "clusterization produced nothing"); return 1; }
    if (out_indices.size() > 0xFFFFFFF0ull) { set_err(err, err_len, "DAG too large"); return 1; }

    int max_depth = 0;
    for (const clodGroup& g : groups) max_depth = std::max(max_depth, g.depth);
    uint32_t coarsest = 0;
    for (const OutCluster& c : clusters)
        if (groups[c.group].simplified.error == FLT_MAX) coarsest += c.tri_count;

    // ---- spatial chunks for incremental streaming
    std::vector<uint32_t> leaf;
    for (uint32_t i = 0; i < clusters.size(); ++i)
        if (clusters[i].refined < 0) leaf.push_back(i);
    int target = int(in->target_chunks);
    if (target <= 0) target = int(std::min<uint32_t>(512, std::max<uint32_t>(1, source_tris / 40000)));
    target = std::max(1, std::min(target, int(leaf.size())));
    KdTree kd;
    kd.build(leaf, 0, leaf.size(), target, clusters);
    const uint32_t chunk_count = uint32_t(kd.chunk_count);

    std::vector<uint32_t> cluster_chunk(clusters.size());
    std::vector<uint32_t> chunk_sizes(chunk_count, 0);
    for (size_t i = 0; i < clusters.size(); ++i) {
        cluster_chunk[i] = uint32_t(kd.find(clusters[i].center));
        chunk_sizes[cluster_chunk[i]]++;
    }
    std::vector<vgeo2::Chunk> chunks(chunk_count);
    {
        uint32_t off = 0;
        for (uint32_t c = 0; c < chunk_count; ++c) {
            chunks[c].cluster_offset = off;
            chunks[c].cluster_count = 0;
            off += chunk_sizes[c];
        }
    }
    std::vector<uint32_t> chunk_clusters(clusters.size());
    std::vector<float> clo(chunk_count * 3, FLT_MAX), chi(chunk_count * 3, -FLT_MAX);
    for (uint32_t i = 0; i < clusters.size(); ++i) {
        vgeo2::Chunk& ch = chunks[cluster_chunk[i]];
        chunk_clusters[ch.cluster_offset + ch.cluster_count++] = i;
        for (int k = 0; k < 3; ++k) {
            clo[cluster_chunk[i] * 3 + k] = std::min(clo[cluster_chunk[i] * 3 + k], clusters[i].center[k] - clusters[i].radius);
            chi[cluster_chunk[i] * 3 + k] = std::max(chi[cluster_chunk[i] * 3 + k], clusters[i].center[k] + clusters[i].radius);
        }
    }
    for (uint32_t c = 0; c < chunk_count; ++c) {
        float r2 = 0.f;
        for (int k = 0; k < 3; ++k) {
            chunks[c].center[k] = 0.5f * (clo[c * 3 + k] + chi[c * 3 + k]);
            float h = 0.5f * (chi[c * 3 + k] - clo[c * 3 + k]);
            r2 += h * h;
        }
        chunks[c].radius = std::sqrt(r2);
    }
    if (report(2, 0.f)) { set_err(err, err_len, "cancelled"); return 2; }

    // ---- write
    vgeo2::Header h = {};
    std::memcpy(h.magic, vgeo2::kMagic, 8);
    h.version = vgeo2::kVersion;
    h.header_size = sizeof(vgeo2::Header);
    h.vertex_count = uint32_t(vertices.size());
    h.index_count = uint32_t(out_indices.size());
    h.cluster_count = uint32_t(clusters.size());
    h.group_count = uint32_t(groups.size());
    h.chunk_count = chunk_count;
    h.chunk_cluster_count = uint32_t(chunk_clusters.size());
    h.material_count = in->material_count;
    h.flags = vgeo2::kHasNormals | (in->uvs ? vgeo2::kHasUVs : 0u);  // normals are computed when absent
    h.lod_levels = uint32_t(max_depth + 1);
    h.source_triangles = source_tris;
    for (int k = 0; k < 3; ++k) { h.aabb_min[k] = FLT_MAX; h.aabb_max[k] = -FLT_MAX; }
    for (const Vertex& v : vertices) {
        const float p[3] = {v.px, v.py, v.pz};
        for (int k = 0; k < 3; ++k) { h.aabb_min[k] = std::min(h.aabb_min[k], p[k]); h.aabb_max[k] = std::max(h.aabb_max[k], p[k]); }
    }

    std::vector<float> positions(vertices.size() * 3), normals(vertices.size() * 3), uvs;
    std::vector<uint16_t> vmat(vertices.size());
    if (in->uvs) uvs.resize(vertices.size() * 2);
    for (size_t i = 0; i < vertices.size(); ++i) {
        const Vertex& v = vertices[i];
        positions[i * 3 + 0] = v.px; positions[i * 3 + 1] = v.py; positions[i * 3 + 2] = v.pz;
        normals[i * 3 + 0] = v.nx; normals[i * 3 + 1] = v.ny; normals[i * 3 + 2] = v.nz;
        if (in->uvs) { uvs[i * 2 + 0] = v.u; uvs[i * 2 + 1] = v.v; }
        vmat[i] = uint16_t(v.mat);
    }
    vertices.clear();
    vertices.shrink_to_fit();

    std::vector<vgeo2::Cluster> fclusters(clusters.size());
    for (size_t i = 0; i < clusters.size(); ++i) {
        vgeo2::Cluster& f = fclusters[i];
        const OutCluster& c = clusters[i];
        f.index_offset = c.index_offset;
        f.tri_count = c.tri_count;
        f.group = c.group;
        f.refined = c.refined;
        f.chunk = cluster_chunk[i];
        f.depth = c.depth;
        std::memcpy(f.center, c.center, sizeof(f.center));
        f.radius = c.radius;
    }
    std::vector<vgeo2::Group> fgroups(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        vgeo2::Group& f = fgroups[i];
        std::memcpy(f.center, groups[i].simplified.center, sizeof(f.center));
        f.radius = groups[i].simplified.radius;
        f.error = groups[i].simplified.error;
        f.depth = groups[i].depth;
    }
    std::string names;
    {
        uint32_t n = in->material_count;
        names.append(reinterpret_cast<const char*>(&n), 4);
        for (uint32_t i = 0; i < n; ++i) {
            const char* s = (in->material_names && in->material_names[i]) ? in->material_names[i] : "";
            uint32_t len = uint32_t(std::strlen(s));
            names.append(reinterpret_cast<const char*>(&len), 4);
            names.append(s, len);
        }
        // optional block after the names (older readers stop before it):
        // "MATP" + material_count * float4 (base color rgb, roughness)
        if (in->material_params && n) {
            names.append("MATP", 4);
            names.append(reinterpret_cast<const char*>(in->material_params), size_t(n) * 16);
        }
    }

    uint64_t off = vgeo2::align16(sizeof(h));
    auto place = [&](uint64_t bytes) { uint64_t o = off; off = vgeo2::align16(off + bytes); return o; };
    h.off_positions = place(positions.size() * 4);
    h.off_normals = place(normals.size() * 4);
    h.off_uvs = in->uvs ? place(uvs.size() * 4) : 0;
    h.off_vmat = place(vmat.size() * 2);
    h.off_indices = place(out_indices.size() * 4);
    h.off_clusters = place(fclusters.size() * sizeof(vgeo2::Cluster));
    h.off_groups = place(fgroups.size() * sizeof(vgeo2::Group));
    h.off_chunks = place(chunks.size() * sizeof(vgeo2::Chunk));
    h.off_chunk_clusters = place(chunk_clusters.size() * 4);
    h.off_materials = place(names.size());
    h.file_size = off;

    // write to a temp file and rename, so a failed build never leaves a truncated asset
    std::string tmp = std::string(path_utf8) + ".part";
    FILE* f = vgeo_io::open_utf8(tmp.c_str(), "wb");
    if (!f) { set_err(err, err_len, "cannot open for writing: " + tmp); return 1; }
    bool ok = true;
    auto put = [&](uint64_t at, const void* data, uint64_t bytes) {
        if (!ok || bytes == 0) return;
        ok = vgeo_io::seek(f, at) && std::fwrite(data, 1, size_t(bytes), f) == bytes;
    };
    put(0, &h, sizeof(h));
    put(h.off_positions, positions.data(), positions.size() * 4);
    put(h.off_normals, normals.data(), normals.size() * 4);
    if (in->uvs) put(h.off_uvs, uvs.data(), uvs.size() * 4);
    put(h.off_vmat, vmat.data(), vmat.size() * 2);
    put(h.off_indices, out_indices.data(), out_indices.size() * 4);
    put(h.off_clusters, fclusters.data(), fclusters.size() * sizeof(vgeo2::Cluster));
    put(h.off_groups, fgroups.data(), fgroups.size() * sizeof(vgeo2::Group));
    put(h.off_chunks, chunks.data(), chunks.size() * sizeof(vgeo2::Chunk));
    put(h.off_chunk_clusters, chunk_clusters.data(), chunk_clusters.size() * 4);
    put(h.off_materials, names.data(), names.size());
    // pad to the declared size
    if (ok && h.file_size > h.off_materials + names.size()) {
        const char zero[16] = {};
        put(h.file_size - 1, zero, 1);
    }
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) {
        vgeo_io::remove_utf8(tmp.c_str());
        set_err(err, err_len, "write failed (disk full?)");
        return 1;
    }
    if (!vgeo_io::replace_utf8(tmp.c_str(), path_utf8)) {
        vgeo_io::remove_utf8(tmp.c_str());
        set_err(err, err_len, std::string("cannot replace ") + path_utf8);
        return 1;
    }
    report(2, 1.f);

    if (stats) {
        stats->source_triangles = source_tris;
        stats->vertices = h.vertex_count;
        stats->clusters = h.cluster_count;
        stats->groups = h.group_count;
        stats->chunks = chunk_count;
        stats->lod_levels = h.lod_levels;
        stats->coarsest_triangles = coarsest;
        stats->file_bytes = h.file_size;
        stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    return 0;
}
