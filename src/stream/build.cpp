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
    // more per-vertex data (more UV maps, colours), welded and seam-protected like the UVs
    const size_t E = (in->extras && in->extra_count) ? size_t(in->extra_count) : 0;
    std::vector<float> extras;
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
        if (E) extras.assign(in->extras, in->extras + vc * E);
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
                    const size_t src = size_t(s.first >> 16);
                    Vertex copy = vertices[src];
                    copy.mat = float(s.first & 0xFFFF);
                    nv = unsigned(vertices.size());
                    vertices.push_back(copy);
                    for (size_t k = 0; k < E; ++k) extras.push_back(extras[src * E + k]);
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
        size_t vertex_count;
        if (E) {
            const meshopt_Stream streams[2] = {{corners.data(), sizeof(Vertex), sizeof(Vertex)},
                                               {in->extras, E * sizeof(float), E * sizeof(float)}};
            vertex_count = meshopt_generateVertexRemapMulti(remap.data(), nullptr, corner_count, corner_count,
                                                            streams, 2);
            extras.resize(vertex_count * E);
            meshopt_remapVertexBuffer(extras.data(), in->extras, corner_count, E * sizeof(float), remap.data());
        } else {
            vertex_count = meshopt_generateVertexRemap(remap.data(), nullptr, corner_count,
                                                       corners.data(), corner_count, sizeof(Vertex));
        }
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
    // seams that only the extra data has (a second UV map, a colour boundary): protect those vertices
    std::vector<unsigned char> locks;
    if (E) {
        std::vector<unsigned int> same(vertices.size());
        meshopt_generatePositionRemap(same.data(), &vertices[0].px, vertices.size(), sizeof(Vertex));
        locks.assign(vertices.size(), 0);
        for (size_t i = 0; i < vertices.size(); ++i) {
            const size_t r = same[i];
            if (r != i && std::memcmp(&extras[i * E], &extras[r * E], E * sizeof(float)) != 0) {
                locks[i] |= meshopt_SimplifyVertex_Protect;
                locks[r] |= meshopt_SimplifyVertex_Protect;
            }
        }
        mesh.vertex_lock = locks.data();
    }

    std::vector<clodGroup> groups;
    std::vector<OutCluster> clusters;
    std::vector<uint32_t> out_indices;
    out_indices.reserve(indices.size() * 2);
    uint64_t emitted_tris = 0;
    bool cancelled = false;

    clodBuild(config, mesh, [&](clodGroup group, const clodCluster* cl, size_t count) -> int {
        int gid = int(groups.size());
        groups.push_back(group);
        for (size_t i = 0; i < count; ++i) {
            OutCluster oc;
            oc.index_offset = uint32_t(out_indices.size());
            oc.tri_count = uint32_t(cl[i].index_count / 3);
            oc.group = gid;
            oc.refined = cl[i].refined;
            oc.depth = uint32_t(group.depth);
            std::memcpy(oc.center, cl[i].bounds.center, sizeof(oc.center));
            oc.radius = cl[i].bounds.radius;
            out_indices.insert(out_indices.end(), cl[i].indices, cl[i].indices + cl[i].index_count);
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
    std::string desc;
    if (E) {
        h.reserved[vgeo2::kExtraOffset] = place(extras.size() * 4);
        h.reserved[vgeo2::kExtraCount] = E;
    }
    if (in->extra_desc && *in->extra_desc) {   // stored with or without extra data (UV name, texture space)
        uint32_t len = uint32_t(std::strlen(in->extra_desc));
        desc.append(reinterpret_cast<const char*>(&len), 4);
        desc.append(in->extra_desc, len);
        h.reserved[vgeo2::kExtraDesc] = place(desc.size());
    }
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
    if (E) put(h.reserved[vgeo2::kExtraOffset], extras.data(), extras.size() * 4);
    if (!desc.empty()) put(h.reserved[vgeo2::kExtraDesc], desc.data(), desc.size());
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
