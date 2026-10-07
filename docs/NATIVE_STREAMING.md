# Native streaming: virtualized geometry rendered by EEVEE and Cycles

The original VGEO path renders meshlets with its own headless Vulkan
renderer and copies the pixels into the viewport. That can never look like
EEVEE or reach Cycles, because it replaces Blender's renderers instead of
feeding them.

The native streaming path keeps Blender's renderers and virtualizes the
*geometry* instead: a Nanite-style cluster DAG lives on disk and in the
native library, and only the view-dependent cut is handed to Blender as
ordinary meshes. Everything Blender can do with a mesh (materials, lights,
shadows, ray tracing, Cycles, EEVEE) just works, and Blender only ever holds
the triangles the current view needs.

## Pipeline

```
Blender mesh (modifiers applied)
    │  addons/vgeo/build.py: triangles, normals, UVs, materials (indexed when smooth)
    ▼
vgeo_stream.dll: vgeo_build
    │  weld → meshoptimizer clusterlod (group clusters, simplify each group with
    │  its border locked, re-split, repeat) → spatial chunks → .vgeo v2
    ▼
.vgeo file (next to the .blend, in //vgeo/)
    │
    ▼
vgeo_stream.dll: vgeo_select / vgeo_extract          (ctypes, no Python ABI)
    │  per view: pick the crack-free cut, hash each chunk's cluster set,
    │  extract changed chunks as indexed meshes with edges
    ▼
addons/vgeo/stream.py
    │  proxy object + chunk objects (parented to it, in a collection linked next to it)
    │  live loop in the viewport, render_pre handler for final renders
    ▼
EEVEE / Cycles / Workbench
```

## Why the cut never cracks

Each cluster in group `g`, simplified from group `r`, is drawn when
`error(g) > t` and `error(r) <= t` (or it is original geometry). Group bounds
enclose their children's bounds and errors only grow up the DAG, so the
decision is monotonic and neighbouring clusters always agree on the border
they share. The tests weld every cut and count open edges: always zero.

**Off-screen coarsening** (the default) raises the threshold for groups
entirely outside the view. That stays crack-free: a group outside the view
has all its finer groups outside too, so monotonicity holds. Off-screen
geometry is kept at low detail, so it still casts shadows and shows in
reflections.

## Landing updates without hitches

Every chunk is a pair of objects. How an update lands depends on the views:

| Views | Strategy | Why |
|---|---|---|
| Solid, Workbench, EEVEE | **staged**: fill the hidden back (scale 0, no shadows) a few ms per tick, then flip scales | Blender prepares the back's GPU buffers as it fills; the flip is a transform change |
| Cycles Rendered | **settle**: update once the view is still for 0.3 s | Cycles rebuilds its BVH and restarts sampling anyway |

The chunks are ordinary objects parented to the proxy, linked through a
collection next to it. Until September 2026 they were instanced by a
Geometry Nodes modifier on the proxy (Collection Info). That made every chunk
change re-evaluate the proxy, and EEVEE then treated the whole instanced
surface as changed: refilling one hidden back cost ~120 ms per frame, so
EEVEE views landed updates in one batch (unlinked spare meshes swapped in),
which cost 100-300 ms in the landing frame (tests/eevee_land_bench.py):

| EEVEE, 70 chunks / 3.1M triangles, 5.1.2 | Geometry Nodes instancing | Separate objects |
|---|---|---|
| Idle frame | 9.7 ms | 9.6 ms |
| Frames while one hidden back is refilled per tick | 119 ms | 25 ms |
| Landing by swapping mesh data | 119 ms | 192 ms |
| Landing prefilled backs by a scale flip | 43 ms | 28 ms |

The proxy keeps its materials and eight loose points at the asset's bounds
(Frame Selected works; nothing is drawn or rendered). Selection, hiding and
render visibility follow the proxy: clicking the surface selects the proxy,
and hiding the proxy hides the chunk collection. Files made with the old
modifier are switched over when opened; a deleted proxy's chunk collection is
unlinked. The BATCH strategy remains in `stream_step` for scripts.

Lessons measured along the way (tests/viewport_bench.py, tests/swap_bench.py):

- Creating or deleting datablocks, and assigning materials, force a
  depsgraph relations rebuild; mid-stream material assignment made EEVEE
  re-sync the whole scene every frame. Streaming now creates nothing and
  assigns materials up front.
- `foreach_set` takes a per-item path for topology arrays. The add-on copies
  arrays straight into mesh attributes (verified once per session on a fixed
  mesh, falling back to `foreach_set`), and the native library computes
  edges so `mesh.update()` does not. Writes went from 0.9M to 8-10M
  triangles/s.
- Around 512 chunks is the sweet spot: 2048 objects made every frame and
  every relations rebuild slower than the smaller updates saved.

## Measured (RTX A4500 over a remote desktop, Blender 5.1.2)

Terrain demo: 33.5M triangles from Geometry Nodes (`examples/terrain_demo.py`).

| | |
|---|---|
| Build (clusterlod, single thread) | 129-141 s, 19 LOD levels, 512 chunks, 1.2 GB file |
| Cut from the hero camera, 1080p, 1 px, off-screen coarsened | 2.5M triangles (7.4 %) |
| Full rebuild of a 7.7M-triangle cut | 1.3 s |
| Viewport idle, Solid / EEVEE | ~75 / ~55 fps |
| Viewport while continuously streaming, Solid | ~29 fps, worst frame ~45-48 ms |
| Viewport while continuously streaming, EEVEE | ~26 fps, worst frame ~56 ms (was 0.3-0.4 s with Geometry Nodes instancing) |
| Cycles, 1080p, 64 samples, same material: raw 33.5M mesh | 93-118 s |
| Cycles, same frame from the VGEO cut (9.8M triangles, 0.5 px) | 49-51 s |
| Difference between the two images | mean 0.02/255, 99th percentile 1/255 |

## Memory: the file is mapped, not read

`vgeo_open` memory-maps the .vgeo instead of reading it. Opening reads the
header and the cluster, group and chunk tables; geometry pages load when a
cut first touches them, and the OS evicts them under memory pressure, so an
asset larger than RAM can be opened and streamed, and every handle on the
same file (the proxy, instancer level builds, web export, per-placement
cuts) shares one copy of the pages. Indices are validated per cluster on
first use instead of by a scan at open (a bad cluster is skipped and
`vgeo_corrupt` reports it). The per-handle scratch that mapped global vertex
indices to chunk-local ones (8 bytes per vertex of the asset) is now a hash
sized to the chunk. Before extracting, the add-on calls `vgeo_prefetch` for
the changed chunks, which asks the OS to start reading their index and
vertex ranges. The hint runs on a background thread per handle (the OS call
can block for seconds when pages are cold: done inline it produced 2-3 s
frames), and vertex ranges of clusters whose vertices span most of the file
are skipped.

Terrain asset (1.2 GB, 33.5M triangles), two handles, plain Python
(`tests/mapped_bench.py`):

| | Read whole file (before) | Mapped |
|---|---|---|
| Open, per handle | 760-900 ms | 10-12 ms |
| Process private memory after opening two handles | +2.7 GB | +15 MB |
| Extract a close-up cut (6.4M tris), warm cache | 1.15 s | 0.98 s |
| Extract full detail (33.5M tris), warm cache | 4.9 s | 3.3 s |
| Same, first mapped run (pages not yet in memory) | | 1.5 s / 6.1 s |

On Windows the file stays open (shared for reading and deleting) while a
handle is open; Reload or Restore closes it.

## Instancing (full scenes)

`Scatter Instances` (active VGEO object + another selected mesh) creates an
instancer: a point mesh whose vertices are placements with rotation and scale
attributes. The asset's uniform LOD levels are built once into shared level
meshes (each a complete, crack-free cut); every update picks, per placement,
the coarsest level whose error stays under the pixel threshold, and writes it
to an integer attribute that a Geometry Nodes Instance on Points tree reads.
Navigating never rebuilds geometry, and EEVEE and Cycles get real instances.

Scatter demo (`examples/scatter_demo.py`): 2,000 copies of a 1.3M-triangle
boulder = 2.6 billion source triangles.

| | |
|---|---|
| Build | virtualize 3.6 s, 15 level meshes 0.3 s |
| Viewport cut at 1 px | 6.6M triangles (0.25 %) |
| Final render at 0.5 px, 1600x900 | 19.4M triangles: EEVEE 1.7 s, Cycles 8.3 s (64 samples, OptiX) |
| Viewport while flying | Solid ~117 fps, EEVEE ~68-79 fps, worst frame 42-45 ms |

**Streamed copies.** A whole-asset level wastes triangles on a copy seen up
close: level 0 is the entire asset at full detail, though most of it is far
away or facing away. Copies that want level 0 (nearest first, up to
**Streamed Copies**, default 4, only for assets whose level 0 is at least
20k triangles) are shown by a streamed copy instead: a VGEO proxy parented
to the instancer, placed like the copy, hidden from selection, streaming a
view-dependent cut through the same live loop. It starts empty; the copy
keeps its instance until the streamed cut has landed, then the instance is
dropped (a `vgeo_streamed` point attribute feeds Instance on Points'
Selection) in the same tick, so there is never a gap or a doubled surface.
Walking away hands the copy back and empties the streamed copy. Final
renders make the streamed cuts synchronously.

Scatter demo, viewport 1.8 radii from a boulder (`tests/slot_gui_check.py`):
the boulder's streamed cut is 353k triangles instead of 1.31M for level 0,
and it takes over 0.85-0.95 s after the view arrives (5.0.1 and 5.1.2,
Solid and EEVEE).

## On the web

`web/vgeo-viewer.js` renders the same assets with WebGPU: a compute pass
applies the same cut rule per cluster and one indirect draw renders the
result; its cut matches the native runtime to the triangle. **Export for Web**
writes a compact `.vgeow` (meshopt-compressed, positions on a global 21-bit
grid so cuts stay watertight) with the viewer and a page. The file is paged
(`vgeo_export_web_paged`): the viewer loads the head and the root pages with
range requests, shows the coarse levels at once and fetches finer pages as the
view needs them; an instanced scene's nearest copies stream their own cut.
See `web/README.md`.

## Using it

1. Build `vgeo_stream` (see below); the DLL lands in `addons/vgeo/bin/`.
2. Install `addons/vgeo` as an add-on (or zip it with `tools/package_addon.py`).
3. Select a mesh, **Object > Virtualize Mesh** (or the VGEO sidebar tab).
   The original is hidden and kept (or removed, if you ask).
4. The proxy streams automatically. Sidebar settings: viewport and render
   pixel error, off-screen mode, freeze, LOD colors, restore.

Save the .blend first so the asset lands in `//vgeo/` next to it.

## Building

```
cmake -S . -B build-stream -G Ninja -DCMAKE_BUILD_TYPE=Release -DVGEO_STREAM_ONLY=ON
cmake --build build-stream
```

Needs a C++20 compiler; no Vulkan SDK, no Python headers. meshoptimizer is
fetched at a pinned commit (or pass `-DVGEO_MESHOPTIMIZER_DIR=...`).
The DLL links the C runtime statically so it loads inside Blender with no
redistributable installed.

## Tests

```
blender -b --factory-startup --python tests/test_stream.py
blender scene.blend --python tests/viewport_bench.py -- --motion fly|orbit
```

## Toward the web

The .vgeo v2 layout (separate vertex arrays, clusters, groups, chunks) was
chosen so a WebGPU runtime can read the same file: the selection rule is the
same few lines, and `vgeo_stream` can be compiled to WebAssembly for the
cut selection if needed.

## Materials see what they saw before (library version 3)

A streamed chunk is its own mesh, so anything a material reads has to travel with the geometry. Besides
positions, normals, the active UV map and material slots, a .vgeo now carries:

- **every other UV map** and **colour and float attributes** on points or corners (up to 64 floats per
  vertex), welded with the rest (a seam in a second UV map splits vertices like a seam in the first) and
  protected from simplification (vertices on such seams are flagged `meshopt_SimplifyVertex_Protect`);
- **the active UV map's own name** (materials whose UV Map node names it kept working only for "UVMap")
  and which maps and colours are active and render-active;
- **the texture space**: each chunk gets the source's, so Generated coordinates line up across chunks
  instead of being fitted to every chunk.

They live in the format's reserved header fields (extra data and a JSON description), so older files open
as before and older libraries ignore the new build fields. `tests/test_material_inputs.py` renders each
input before and after Virtualize Mesh (mean difference 0.16-0.44/255) and checks Scatter's level meshes.
Before: Generated-coordinate checkers were scrambled per chunk, a second UV map rendered untextured, colour
attributes rendered black.

Real scans (Poly Haven, 4K albedo + normal + roughness, EEVEE and Cycles, RTX 5090, 5.1.2):

| | Original | VGEO | Difference |
|---|---|---|---|
| Coastal cliff, 1.54M triangles, wide shot | 2.6 s / 6.0 s | 1.9 s / 3.8 s, 571k-triangle cut | mean 0.27 / 0.33 of 255 |
| Same, close-up | 2.6 s / 4.6 s | 1.6 s / 3.9 s | mean 0.00 / 0.03 |
| Moon rock displaced into 15.3M real triangles | 11.5 s / 15.7 s | 2.9 s / 4.4 s | mean 0.49 / 0.54 |
| 1,000 scattered copies of it: 15.3 billion source triangles | (not renderable) | 4.0 s / 4.7 s | |

## A 15-billion-triangle field in the viewport (RTX 5090, 5.1.2, Xvfb window)

1,000 scattered copies of the 15.3M-triangle displaced rock, flown through in the viewport
(`tests/viewport_bench.py`-style flight, frame times from draw handler timestamps):

| | Before | After |
|---|---|---|
| Live-loop tick | 120-220 ms | 16-19 ms |
| Solid, flying | 3.8 fps (260 ms) | ~13-15 fps (66-78 ms) |
| EEVEE Material Preview, flying | 2-3 fps | ~25 fps (39 ms median) |
| Same view frozen (no streaming): Solid / EEVEE | | 27 / 34 ms |
| Triangles on screen | 120M | 99-107M |

What it took:
- **Chunk lookups.** Idle streamed copies were "released" again every tick, which dropped their chunk
  cache, and rebuilding it looked up each chunk by name in `bpy.data.objects` (a scan of every object):
  95k lookups, 8.8 of 10.8 profiled seconds. Idle copies are left alone, emptying keeps the cache, and
  lookups go through the chunk collection.
- **Error-based instance levels.** Levels were uniform DAG depths, whose error is the worst cluster's: the
  rock's depth 5 was 476k triangles at 7.7 mm and depth 6 jumped to 4.6 cm, so most copies sat at depth 5.
  Levels are now the cheapest crack-free cut under an error that doubles per level (ortho select). Copies
  wanting a level of 20k+ triangles (not just level 0) get the streamed copies.
- **Settling.** In the live loop, a copy refines at once but coarsens only when two levels too fine, and
  level writes happen at most every 0.2 s. EEVEE views now update cuts once the view is still (like Cycles):
  streaming near copies while flying cost EEVEE ~200 ms a frame.

Still far from Nanite here: ~100M triangles are drawn because a copy is a whole-rock level (no per-cluster
back-face or occlusion culling), and every changed mesh costs Blender a buffer rebuild. Getting to
Nanite-like interactivity in the viewport needs drawing the clusters ourselves on the GPU (culling per
cluster, one indirect draw), as the web viewer already does, with EEVEE/Cycles for final renders.
`tests/slot_gui_check.py` in Solid now switches copies (it timed out before) but not yet the one in front
of the camera within its window.

## Not yet

- A streamed copy's first assignment creates its chunk objects (one
  relations rebuild, ~0.1 s in EEVEE); later reassignments reuse them.
- Building without a full Blender mesh in memory (import straight from
  disk, or tiled builds), for sources beyond what Blender can hold.
- Rare back-to-back "fins" left by simplification (about 3 per 100k
  triangles at coarse levels) are harmless but could be filtered.
