// VGEO stream - C API for building and streaming cluster-LOD geometry.
//
// One shared library, called from Blender through ctypes (so a single binary
// serves every Blender/Python version) and later compiled to WebAssembly for
// the web runtime. Everything here is plain C types.
//
// Build:   triangle soup (per-corner attributes) -> .vgeo (format v2)
// Runtime: open .vgeo -> select a DAG cut for one or more views ->
//          extract the cut chunk by chunk as indexed meshes.
#pragma once

#include <stdint.h>

#ifdef _WIN32
#define VGEO_API __declspec(dllexport)
#else
#define VGEO_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VGEO_STREAM_VERSION 2

// ---------------------------------------------------------------- build

// Two input layouts:
//  - corner soup (indices == NULL): positions/normals/uvs hold one entry per
//    triangle corner (tri_count*3); identical corners are welded.
//  - indexed (indices != NULL): positions/normals/uvs hold vertex_count
//    entries and indices holds tri_count*3 vertex indices. Much lighter for
//    smooth meshes without UV seams (scans, terrain). Vertices on material
//    borders are split automatically.
typedef struct vgeo_build_input {
    uint32_t tri_count;
    const float* positions;       // 3 floats per corner (soup) or per vertex (indexed)
    const float* normals;         // 3 floats per corner/vertex, or NULL
    const float* uvs;             // 2 floats per corner/vertex, or NULL
    const uint16_t* materials;    // tri_count, or NULL
    uint32_t vertex_count;        // indexed layout only
    const uint32_t* indices;      // tri_count*3, or NULL for corner soup
    uint32_t material_count;
    const char* const* material_names;  // material_count UTF-8 strings, or NULL
    uint32_t max_triangles;       // triangles per cluster, 0 = 128
    uint32_t target_chunks;       // streaming chunks, 0 = auto
    const float* material_params; // material_count * 4 (r, g, b, roughness), or NULL
} vgeo_build_input;

// stage: 0 = welding, 1 = building DAG (progress 0..1 is approximate), 2 = writing
// return nonzero to cancel
typedef int (*vgeo_progress_fn)(void* user, int stage, float progress);

typedef struct vgeo_build_stats {
    uint32_t source_triangles;
    uint32_t vertices;
    uint32_t clusters;
    uint32_t groups;
    uint32_t chunks;
    uint32_t lod_levels;
    uint32_t coarsest_triangles;  // triangles in the least detailed full cut
    uint64_t file_bytes;
    double seconds;
} vgeo_build_stats;

// returns 0 on success; on failure writes a message to err
VGEO_API int vgeo_build(const vgeo_build_input* input, const char* path_utf8,
                        vgeo_progress_fn progress, void* user,
                        vgeo_build_stats* stats, char* err, int err_len);

// ---------------------------------------------------------------- runtime

typedef struct vgeo_info {
    uint32_t vertex_count;
    uint32_t cluster_count;
    uint32_t group_count;
    uint32_t chunk_count;
    uint32_t material_count;
    uint32_t lod_levels;
    uint32_t flags;               // bit0 normals, bit1 uvs
    uint32_t source_triangles;
    float aabb_min[3];
    float aabb_max[3];
} vgeo_info;

// A view, in the asset's local space.
// perspective: proj = cot(fov_y / 2); ortho: ortho_height = visible height.
// threshold is the allowed error as a fraction of the view height
// (pixels / viewport height in pixels).
//
// frustum_mode (needs planes):
//   0 = ignore the frustum
//   1 = coarsen: groups entirely outside use threshold * offscreen_scale.
//       Off-screen geometry stays (shadows, reflections) but at low detail.
//       Still a valid cut: a group outside the frustum has all its finer
//       groups outside too, so pass/fail stays monotonic.
//   2 = cull: clusters outside are dropped
typedef struct vgeo_view {
    float camera[3];
    float proj;
    float znear;
    float threshold;
    int32_t ortho;
    float ortho_height;
    int32_t frustum_mode;
    float offscreen_scale;
    float planes[6][4];           // inward-facing: dot(n, p) + d >= 0 inside
} vgeo_view;

typedef struct vgeo_cut_stats {
    uint64_t triangles;
    uint32_t clusters;
    uint32_t changed_chunks;
} vgeo_cut_stats;

// Chunk contents, valid until the next vgeo_extract/vgeo_close on this handle.
typedef struct vgeo_chunk_data {
    uint32_t vertex_count;
    uint32_t tri_count;
    const float* positions;       // vertex_count * 3
    const float* normals;         // vertex_count * 3 (NULL if absent)
    const float* uvs;             // vertex_count * 2 (NULL if absent)
    const int32_t* corner_verts;  // tri_count * 3
    const int32_t* face_materials;// tri_count
    const int32_t* face_lod;      // tri_count, DAG depth (for debug views)
    // Blender-style edges: unique vertex pairs, and per corner the edge from
    // that corner to the next corner of its triangle
    uint32_t edge_count;
    const int32_t* edge_verts;    // edge_count * 2
    const int32_t* corner_edges;  // tri_count * 3
} vgeo_chunk_data;

VGEO_API void* vgeo_open(const char* path_utf8, char* err, int err_len);
VGEO_API void vgeo_close(void* handle);
VGEO_API int vgeo_get_info(void* handle, vgeo_info* info);
VGEO_API int vgeo_material_name(void* handle, uint32_t index, char* buf, int buf_len);

// Select the cut that satisfies every view (a cluster is refined while its
// error is above threshold in any view). chunk_sig receives one signature per
// chunk; a chunk needs rebuilding when its signature changed.
VGEO_API int vgeo_select(void* handle, const vgeo_view* views, int view_count,
                         uint64_t* chunk_sig, vgeo_cut_stats* stats);

// Select the whole asset at a fixed DAG depth (-1 = coarsest, 0 = full detail).
VGEO_API int vgeo_select_level(void* handle, int depth, uint64_t* chunk_sig, vgeo_cut_stats* stats);

// Geometric error of drawing the whole asset at each uniform DAG level
// (what vgeo_select_level(depth) shows): out[0] = 0, out[L] = largest
// simplification error of the groups level L was simplified from. Writes
// min(max_levels, lod_levels) values and returns that count.
VGEO_API int vgeo_level_errors(void* handle, float* out, int max_levels);

// Extract the currently selected geometry of one chunk.
VGEO_API int vgeo_extract(void* handle, uint32_t chunk, vgeo_chunk_data* out);

// Files are memory-mapped: opening reads only the header and cluster tables, and
// geometry pages load when a cut first touches them (the OS evicts them under
// memory pressure, so assets larger than RAM work). Ask the OS to start reading
// what these chunks' current selection needs, before extracting them. A hint:
// returns immediately.
VGEO_API int vgeo_prefetch(void* handle, const uint32_t* chunks, int count);

// Bytes mapped from the file, and bytes this handle allocated itself.
VGEO_API int vgeo_memory(void* handle, uint64_t* mapped_bytes, uint64_t* heap_bytes);

// 1 if a cluster with out-of-range indices was found (and skipped) so far.
VGEO_API int vgeo_corrupt(void* handle);

// Write the compact web variant (.vgeow) of an opened asset: self-contained
// clusters (8-bit local indices, material per triangle), positions on a global
// 21-bit grid (identical inputs quantize identically, so cuts stay crack-free),
// octahedral normals. About a third of the v2 size, in the layout the WebGPU
// viewer reads directly. Returns 0 on success.
VGEO_API int vgeo_export_web(void* handle, const char* path_utf8, uint64_t* out_bytes, char* err, int err_len);

// The same, as .vgeow version 2: split into pages of whole groups (about
// page_vertices vertices each, 0 = 4096), ordered coarse to fine, so a viewer
// can load the head, show the coarse levels and fetch finer pages with HTTP
// range requests as the view needs them. Returns 0 on success.
VGEO_API int vgeo_export_web_paged(void* handle, const char* path_utf8, uint32_t page_vertices,
                                   uint64_t* out_bytes, char* err, int err_len);

// ---------------------------------------------------------------- cut atoms and many copies

// An atom is the set of clusters sharing a (group, refined) pair. The cut rule tests only those
// two groups, so an atom is always in or out of the cut as a whole: a fixed piece of the asset at
// a fixed detail. Copies of an asset can be drawn as instances of atom meshes that never change.
typedef struct vgeo_atom {
    int32_t group;                // group its clusters belong to
    int32_t refined;              // group they were simplified from, -1 = original geometry
    uint32_t cluster_offset;      // into vgeo_atom_clusters
    uint32_t cluster_count;
    uint32_t tri_count;
    uint32_t depth;               // DAG depth of `group`
    float center[3];              // bounds enclosing its clusters' bounds
    float radius;
} vgeo_atom;

// Number of atoms; writes min(max_atoms, count) atoms to out (out may be NULL).
VGEO_API int vgeo_atoms(void* handle, vgeo_atom* out, int max_atoms);

// Extract one atom as an indexed mesh (same layout as vgeo_extract; valid until the next extract).
VGEO_API int vgeo_extract_atom(void* handle, uint32_t atom, vgeo_chunk_data* out);

// Lights that let copies outside the view stay, because their shadow can reach it (Shadows mode).
typedef struct vgeo_shadow_light {
    int32_t kind;                 // 0 = sun (dir), 1 = point or spot (pos, range, axis, cos_half)
    float dir[3];                 // sun: unit direction its light travels
    float reach;                  // sun: how far a shadow reaches along dir; <= 0 = 2 r / max(-dir.z, 0.1)
    float pos[3];
    float range;
    float axis[3];                // spot axis (unit); ignored when half_angle >= pi/2
    float half_angle;             // pi for point lights
} vgeo_shadow_light;

enum { VGEO_OFFSCREEN_FULL = 0, VGEO_OFFSCREEN_COARSEN = 1, VGEO_OFFSCREEN_CULL = 2,
       VGEO_OFFSCREEN_SHADOWS = 3 };

typedef struct vgeo_instances_stats {
    uint64_t triangles;           // over every pair written
    uint32_t pairs;               // pairs needed (may exceed max_pairs: then nothing past it is written)
    uint32_t copies_drawn;        // copies with at least one atom
} vgeo_instances_stats;

// Select the cut of every copy of the asset in one call and return (copy, atom) pairs, copy-major.
// views are in WORLD space (camera, planes) with the asset's units (threshold as in vgeo_view;
// frustum_mode and offscreen_scale of the views are ignored: `mode` and `offscreen_scale` apply).
// transforms: n * 12 floats, row-major 3x4 local-to-world matrices; scales: n uniform scales
// (how the matrix scales lengths; errors and radii are multiplied by it). copy_mask (n bytes, or
// NULL) skips copies whose byte is 0. Modes: FULL ignores the frustum; COARSEN uses
// threshold * offscreen_scale for groups outside every view; CULL drops atoms outside every view;
// SHADOWS is CULL, except atoms whose shadow from `lights` can reach a view are kept (coarsened).
VGEO_API int vgeo_select_instances(void* handle, const vgeo_view* views, int view_count,
                                   const float* transforms, const float* scales, const uint8_t* copy_mask,
                                   int copy_count, int mode, float offscreen_scale,
                                   const vgeo_shadow_light* lights, int light_count,
                                   uint32_t* out_pairs, uint32_t max_pairs, vgeo_instances_stats* stats);

VGEO_API int vgeo_version(void);

#ifdef __cplusplus
}
#endif
