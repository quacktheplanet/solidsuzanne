# Agent brief: make Solid Suzanne the best Nanite-style geometry add-on for Blender

This brief is for an agent picking up the work. Read all of it before you change
anything. The detailed plan is in `docs/ROADMAP.md`; this file explains the goal,
the context behind it, how to work, and the traps we already hit.

## The mission

Solid Suzanne is a Blender add-on (`addons/vgeo`, "VGEO", virtualized geometry)
with a C++ core (`src/stream`, built as the `vgeo_stream` shared library). It
turns very dense meshes into a cluster hierarchy, like Unreal's Nanite. Each
view then gets only the detail it needs, so scenes with billions of triangles
work in the viewport, EEVEE and Cycles.

The owner started it before coding agents were this capable. Everyone is doing
virtual geometry now, but nobody does it for Blender's own renderers. Octane
has meshlets, but only inside the Octane renderer; the Blender add-on stores
sell classic distance-based LOD switchers. The goal is the best add-on of its
kind for Blender, at least as efficient as the browser version
(`web/vgeo-viewer.js` and the three.js virtual-geometry library it was compared
with).

"As efficient as the browser" is defined by five measurable targets (T1-T5 in
`docs/ROADMAP.md`). The headline one is T1: draw no more than 10% above a
meshlet cut on every copy with copies outside the view skipped. The benchmark
reports that number as `cut+cull`.

## Read these first, in this order

1. `README.md`, then `docs/NATIVE_STREAMING.md` (how it works today, measured).
2. `docs/ROADMAP.md` (targets, baseline numbers, phases 1-5, checklists).
3. `spec/vgeo_format.md` and `src/stream/format_v2.h` (file layout and the cut rule).
4. `addons/vgeo/instances.py` (instancers), `addons/vgeo/stream.py` (proxies,
   live loop, render handlers), `addons/vgeo/native.py` (ctypes bindings),
   `src/stream/runtime.cpp` (selection), `src/stream/build.cpp` (DAG build).
5. `tests/test_stream.py` (the headless suite) and `tools/instancing_bench.py`.
6. PR #2 (`phase1-instancing`): the first phase-1 work and its measurements.

## The project in one page

- **Build.** `vgeo_build` uses meshoptimizer's clusterlod to make a DAG of
  ~128-triangle clusters. Clusters are grouped and simplified, level after
  level, into a memory-mapped `.vgeo` file. A cluster stores `group` (the group
  it belongs to) and `refined` (the group it was simplified from, -1 for
  original geometry). The cut rule is in `format_v2.h`: cluster c in group g is
  drawn when error(g) > threshold and (c.refined < 0 or error(c.refined) <=
  threshold), with errors projected to the screen.
- **Proxies.** A virtualized object is a proxy with chunk objects. Each tick,
  `vgeo_select` picks the cut for the current views, and the chunks whose
  contents changed are rewritten as ordinary meshes (`stream.fill_mesh`). EEVEE
  and Cycles render them with full materials, lights and shadows. Final renders
  cut synchronously in `render_pre` and `frame_change_post`.
- **Instancers.** Many copies of one asset sit on a point mesh with rotation,
  scale and level attributes. The asset's uniform DAG levels are built once as
  level meshes, and Geometry Nodes' Instance on Points picks one level per copy.
  The heaviest copies get a streamed copy: a proxy placed like the copy, with a
  view-dependent cut. Copies outside the view follow the instancer's
  Off-screen mode: Coarsen, Sun Shadows or Cull. Culled copies get level -1.
- **Web.** `vgeo_export_web(_paged)` writes `.vgeow`. The WebGPU viewer picks
  the same cut per cluster in a compute pass and matches the native runtime to
  the triangle.

## The one constraint that shapes every decision

A Python add-on cannot feed EEVEE or Cycles from its own GPU buffers. All
geometry reaches them as meshes and instances through the depsgraph, and every
changed mesh is uploaded again. The browser version picks and draws on the GPU.
We have to choose on the CPU (in C++) and hand Blender as little as possible, in
forms it handles cheaply. Prefer instances of meshes that never change over
meshes that change every frame. Phase 2 rests on this.

## Where things stand (end of November 2026)

The statue avenue (`examples/statue_avenue`) has 1,704 carved statues of 1.43M
triangles and 1,188 floor patches of 1.62M, 4.36 billion triangles in all.
Numbers below are at 1280x720 and 1.5 px, averaged over the 8 cameras in
`cameras.json`:

| | Triangles drawn |
|---|---|
| master (before PR #2) | 2.77M |
| PR #2: Coarsen / Sun Shadows / Cull | 1.78M / 1.57M / 1.51M |
| Ideal: meshlet cut on every copy, coarsened / culled (`cut` / `cut+cull`) | 1.67M / 1.24M |

The synthetic field (`--synthetic`, 1,500 copies of an 82k-triangle rock) shows
the real gap. It draws 15.96M (Coarsen) or 14.31M (Cull) against an ideal of
7.13M and 5.91M. Whole-copy levels cost about twice a meshlet cut: a level must
be fine enough for the copy's nearest point, and levels halve in size. Streamed
copies only help when a few copies are very heavy. Phase 2 fixes this for every
copy.

Not yet measured: viewport frame rates. Our environment had no GPU, so those
need the owner's machine. Also open: memory grew 0.3-0.6 GB per frame during
the demo film render (headless bpy 5.1.2, EEVEE in software GL, one still render
per frame in one session). The film had to be rendered in 8-frame processes.

## Your work, in order

Each step is a branch and a PR. Every PR reports the benchmark before and after.

### 0. Set up and reproduce the baseline (do this first)

- Build the core library:
  `cmake -S . -B build-stream -DCMAKE_BUILD_TYPE=Release -DVGEO_STREAM_ONLY=ON && cmake --build build-stream`.
  The library lands in `addons/vgeo/bin/`. Building fetches meshoptimizer at a
  pinned commit.
- Get Blender 5.1 headless: either a Blender binary or the bpy module
  (`pip install bpy==5.1.2`, Python 3.13).
- Run the tests: `blender -b --factory-startup --python tests/test_stream.py`
  (or `python tests/test_stream.py` with bpy). Expect `103/103 passed` on the
  PR #2 branch.
- Build the statue avenue: `python examples/statue_avenue/build_scene.py -- OUT`.
  That takes about 1.5 min on 2 CPUs and writes a 160 MB `.blend` plus the
  `.vgeo` assets.
- Run `tools/instancing_bench.py OUT/statue_avenue.blend -- --cameras examples/statue_avenue/cameras.json --error 1.5 --offscreen COARSEN,SHADOWS,CULL`
  and check it reproduces the table above. Run `--synthetic` too.

If anything does not reproduce, stop and find out why before building on it.

### 1. Finish phase 1

- **Memory growth in long renders** (target T3). Write `tools/render_memory_bench.py`.
  It renders N frames of the statue avenue in one session along a camera path
  through `cameras.json` and logs, per frame: VmRSS from `/proc/self/status`
  (or psutil); `len(bpy.data.meshes)`, `objects` and `collections`; orphan
  meshes; and the native runtime's own memory (`vgeo_memory`).
  - Run it with the add-on's updates frozen as well, to separate the add-on
    from EEVEE.
  - Suspects: chunk meshes of released streamed copies, back buffers, mesh
    datablocks left with zero users, numpy buffers kept in Python, native
    asset handles.
  - Done when RSS stays within 5% over 100 frames in EEVEE and Cycles.
- **Default Off-screen for new instancers:** set Sun Shadows in
  `create_instancer` (and in scatter), not as the property default. Files saved
  before the change must keep Coarsen.
- **Sun Shadows:**
  - Cache the sun list per tick instead of scanning `scene.objects` on every
    update.
  - Also keep copies within range of shadow-casting point and spot lights
    (bounding sphere against the light's range or cutoff).
  - Add tests next to the existing sun-shadow tests.
- **Viewport check on a GPU:** `tests/viewport_bench.py`, Solid and EEVEE,
  flying through the statue avenue, before and after PR #2. If you have no
  GPU, prepare the exact commands for the owner.

### 2. Phase 2 spike: can Blender take instanced cut atoms? (a few days, decide before building)

The idea: instance the pieces of the cut instead of the copies.

An **atom** is the set of clusters with the same (`group`, `refined`) pair. The
cut rule's level-of-detail test depends only on those two groups, so an atom's
clusters are always drawn together: it is a fixed piece of geometry at a fixed
detail. Per-cluster frustum culling (Cull mode) is finer than that; per atom,
cull with the union of its clusters' bounds, which is slightly conservative.
Verify this on real assets before relying on it: for random views, the cut
built from atoms must equal the per-cluster cut.

Measured on the statue avenue assets:

| Asset | Clusters | Groups | Atoms | Triangles per atom (10th / median / 90th percentile) |
|---|---|---|---|---|
| Statue | 23,334 | 1,440 | 4,542 | 128 / 384 / 1,920 |
| Floor patch | 26,420 | 1,619 | 5,155 | 128 / 384 / 1,920 |

The whole DAG holds about twice the source triangles: 2.86M for the 1.43M
statue. That is what the atom meshes cost in memory, once per asset.

The spike tests whether Blender copes, using plain bpy without the add-on:

- Make 5,000 distinct small meshes in that size range (one per atom).
- Build a point cloud of 50k, 200k and 500k points with transform and atom-index
  attributes. Instance them with Geometry Nodes (Instance on Points with Pick
  Instance from a collection).
- Each tick, rewrite the attributes for 1%, 10% and 100% of the points.
- Measure:
  - Geometry Nodes evaluation time
  - depsgraph update time
  - Solid and EEVEE frame time (on a GPU)
  - Cycles BVH build time and memory
  - RSS
- Compare against today's path on the statue avenue.
- If the numbers are bad, try these mitigations: bigger blocks (merge atoms
  that are often drawn together), one point cloud per region so a change
  touches one small object, and updating only copies whose cut changed.

Write the results into `docs/ROADMAP.md` (phase 2) with a go or no-go. Get the
owner's agreement before the full implementation.

### 3. Phase 2: a meshlet cut on every copy

- **C++:** a new `vgeo_select_instances(handle, views, n_views, transforms, n,
  thresholds, mode, out_pairs, ...)`. It selects the cut for every copy and
  returns (copy, atom) pairs, culling atoms outside the view per mode.
  Selection for 3,000 copies must stay well under a frame. The benchmark
  already does about 0.1 ms per copy per view through Python; do it in one
  call.
- **Build:** atoms become small meshes in the asset's hidden collection (where
  level meshes live now). Give them stable names and indices, and keep the
  build incremental.
- **Python:** the instancer writes the pairs into its point cloud, and Instance
  on Points picks the atom mesh. Keep Off-screen modes and the render handlers
  working the same way.
- **Unify:** a proxy is an instancer with one copy. Once the atom path is
  faster everywhere, move proxies to it, then retire chunk meshes and streamed
  copies. Keep old files loading by upgrading them in place.
- **Tests:**
  - Atom cut equals per-cluster cut.
  - No open edges in an instanced cut (crack test, like the existing ones).
  - Culled atoms leave the instancing.
  - Save and load.
  - Final render cuts.
- Done when T1 holds on both benchmark scenes (within 10% of `cut+cull`),
  viewport fps is at least as good as today (T2) and memory is flat (T3).

### 4. Phase 3: occlusion culling

- Rasterize coarse occluders into a small CPU depth buffer (about 256x144) in
  `vgeo_stream`. Use the coarsest levels of the nearest copies and large
  proxies. Build a depth pyramid, and test copies and atoms against it. Masked
  software occlusion culling is the reference technique; check licenses before
  borrowing code.
- Shadows:
  - Sun Shadows mode: drop an occluded copy only if its swept shadow bounds
    are occluded too.
  - Coarsen mode: coarsen occluded copies, don't drop them.
- Done when ground-level views of the statue avenue draw at least 30% fewer
  triangles than phase 2 alone, with no pixel differences in the shadow check.

### 5. Phase 4: product

The details are in `docs/ROADMAP.md`. Do these after the engine work, unless
the owner asks otherwise:

- Virtualize any Geometry Nodes scatter, particle system or collection instance setup.
- Built-in debug views: Meshlets, LOD ramp, Triangles, and a stats overlay.
  The demo film packed `(cluster_id << 6) | depth` into the runtime's per-face
  LOD and wrote a corner BYTE_COLOR attribute `vgeo_viz` (RGB = cluster hash,
  A = depth / 16). The statue avenue materials already read it.
- Asset Browser support.
- Import from disk without loading the mesh into Blender.
- Blender Extensions packaging with CI builds for Windows, macOS and Linux,
  and CI running the headless tests.
- Unreal export via USD.
- Docs and demo scenes.

## How to work

- One branch and PR per step. Do not push to `master`. The owner merges.
- Every change to what gets drawn reports `tools/instancing_bench.py` before
  and after, on the statue avenue and the synthetic field.
- Every culling change also needs a pixel check. Render one frame looking away
  from a low sun, with and without the change, and diff the images
  (`docs/images/offscreen_modes.jpg` shows the expected result). Shadows must
  not disappear unless the mode says they may.
- Keep old `.blend` files working. Upgrade node groups and attributes in
  place, as `_ensure_streamed_selection` and `_ensure_culled_selection` do.
  Never change what an existing file renders just by changing a property
  default.
- Do not widen test tolerances to make a test pass. Find the cause.
- Update `docs/ROADMAP.md` checklists and `docs/NATIVE_STREAMING.md` in the
  same PR as the change.
- Measure before claiming: numbers in docs and PRs come from runs you did, with
  the command that produced them.

## Traps we already hit

- **Stale camera projection.** After changing a camera's data (lens or
  angle), call `depsgraph.update()` before `calc_matrix_camera`, or you get the
  previous projection. This silently skewed an analysis until the numbers were
  compared with the recorded ones.
- **Render-path cuts.** `render_pre` reads a depsgraph that the render
  pipeline has not moved to the new frame. Animation renders therefore re-cut
  in `frame_change_post` while rendering (fixed in PR #2). Keep both paths in
  mind when touching render handlers.
- **Streamed copies' error.** Streamed copies cut at their instancer's
  `render_pixel_error`, not at the error passed to `instances.update`. Set both
  when benchmarking.
- **Writing colour attributes.** Write BYTE_COLOR attributes with
  `attribute.data.foreach_set("color", float_rgba)`. A raw fast-copy of float
  data into a byte-colour attribute corrupted the heap in a demo build.
- **Large integers in ID properties.** Int ID properties overflow above 2^31
  (billions of triangles). Store such numbers as floats.
- **Headless EEVEE.** Without a GPU, EEVEE runs in software GL at about 14 s a
  frame at 720p on the statue avenue. Sun shadows dominate; turn off shadow
  casting on floor patches to save time. Memory also grows per frame (phase 1).
- **The GitHub CLI** in Claude Code cloud sessions cannot use GraphQL. Open PRs
  with `gh api repos/quacktheplanet/solidsuzanne/pulls -f title=... -f head=...
  -f base=master -F body=@file`.

## Reporting back

At the end of each step, give the owner:

- What changed, in a sentence or two.
- The benchmark table before and after.
- Anything you could not check, and the exact command for it, for example
  viewport fps on a GPU.
- The PR link.

Keep it short: the owner reads on a phone.
