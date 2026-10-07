// On-disk layout of .vgeo format v2 (cluster LOD DAG for streaming).
//
// Little-endian. Every section starts on a 16-byte boundary. Vertex data is
// stored as separate arrays so it can be uploaded or memory-mapped as-is.
//
// Selection rule (Nanite / meshoptimizer clusterlod): cluster c in group g is
// part of the cut when error(g) > threshold and (c.refined < 0 or
// error(groups[c.refined]) <= threshold), where error() projects a group's
// bounds + simplification error to the screen. Terminal groups have
// error = FLT_MAX and are never simplified further.
#pragma once

#include <stdint.h>

namespace vgeo2 {

static const char kMagic[8] = {'V', 'G', 'E', 'O', '2', 0, 0, 0};
static const uint32_t kVersion = 2;

enum : uint32_t {
    kHasNormals = 1u << 0,
    kHasUVs = 1u << 1,
};

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t header_size;

    uint32_t vertex_count;
    uint32_t index_count;
    uint32_t cluster_count;
    uint32_t group_count;
    uint32_t chunk_count;
    uint32_t chunk_cluster_count;
    uint32_t material_count;
    uint32_t flags;
    uint32_t lod_levels;
    uint32_t source_triangles;

    float aabb_min[3];
    float aabb_max[3];

    uint64_t off_positions;   // float3[vertex_count]
    uint64_t off_normals;     // float3[vertex_count] (if kHasNormals)
    uint64_t off_uvs;         // float2[vertex_count] (if kHasUVs)
    uint64_t off_vmat;        // uint16[vertex_count]
    uint64_t off_indices;     // uint32[index_count], global vertex indices
    uint64_t off_clusters;    // Cluster[cluster_count]
    uint64_t off_groups;      // Group[group_count]
    uint64_t off_chunks;      // Chunk[chunk_count]
    uint64_t off_chunk_clusters; // uint32[chunk_cluster_count]
    uint64_t off_materials;   // uint32 count, then per name: uint32 length + UTF-8 bytes,
                              // then optionally "MATP" + count * float4 (base rgb, roughness)
    uint64_t file_size;
    // reserved[0] = offset of extras: float[vertex_count * extra_count] (0 = none)
    // reserved[1] = extra_count
    // reserved[2] = offset of the extra description: uint32 length + UTF-8 bytes (0 = none)
    uint64_t reserved[4];
};
enum : uint32_t { kExtraOffset = 0, kExtraCount = 1, kExtraDesc = 2 };

struct Cluster {
    uint32_t index_offset;    // into indices
    uint32_t tri_count;
    int32_t group;            // group this cluster belongs to
    int32_t refined;          // group it was simplified from, -1 = original geometry
    uint32_t chunk;
    uint32_t depth;           // DAG depth of its group
    float center[3];          // culling bounds
    float radius;
};

struct Group {
    float center[3];          // simplified bounds (monotonic across the DAG)
    float radius;
    float error;              // FLT_MAX for terminal groups
    int32_t depth;
    uint32_t reserved[2];
};

struct Chunk {
    uint32_t cluster_offset;  // into chunk_clusters
    uint32_t cluster_count;
    float center[3];
    float radius;
};

static_assert(sizeof(Cluster) == 40, "Cluster layout");
static_assert(sizeof(Group) == 32, "Group layout");
static_assert(sizeof(Chunk) == 24, "Chunk layout");

inline uint64_t align16(uint64_t v) { return (v + 15) & ~uint64_t(15); }

}  // namespace vgeo2
