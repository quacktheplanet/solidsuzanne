# Roadmap: Nanite-style geometry for Blender

Goal: the best virtual-geometry add-on for Blender, at least as efficient as the
browser version, with billion-triangle scenes that work in the viewport, in
EEVEE and in Cycles.

Status: November 2026. Phase 1 has started (see the checklist below).

Picking this up as an agent? Start with `docs/AGENT_BRIEF.md`.

---

## What "as good as the browser" means

The browser library and `web/vgeo-viewer.js` pick a meshlet cut on the GPU for
every copy of every asset, skip copies outside the view (and, in the library,
copies hidden behind others) and draw straight from GPU buffers. These targets
make that comparison measurable:

| # | Target | How it is measured |
|---|---|---|
| T1 | Triangles drawn within 10% of a meshlet cut on every copy with off-screen copies skipped (the `cut+cull` column) | `tools/instancing_bench.py` on the statue avenue and the synthetic field |
| T2 | Viewport while flying through the statue avenue: EEVEE 30+ fps, worst frame under 100 ms; Solid 60+ fps (RTX 3060-class GPU) | `tests/viewport_bench.py` |
| T3 | Memory flat in the number of copies; a long animation render does not grow (within 5% over 100 frames) | RSS over a 100-frame render of the statue avenue |
| T4 | Final renders of the 4.36-billion-triangle statue avenue on a 16 GB machine, EEVEE and Cycles | render times and peak RSS |
| T5 | No holes, cracks or popping a viewer would notice; a new cut lands within 1 s | headless tests (open-edge checks), viewport bench |

## Where it stands

Working today (see `docs/NATIVE_STREAMING.md`):

- Cluster DAG built with meshoptimizer's clusterlod into a memory-mapped `.vgeo`.
- Single objects (proxies) stream a view-dependent, crack-free cut into ordinary
  mesh chunks, so EEVEE and Cycles render them with full materials, lights and
  shadows.
- Instancers: whole-asset level meshes picked per copy, with the heaviest copies
  given their own streamed cut; Off-screen modes (Coarsen, Sun Shadows, Cull).
- Web export (paged `.vgeow`) with a WebGPU viewer whose cut matches the native
  runtime to the triangle.
- 103 headless tests.

Measured on the statue avenue (`examples/statue_avenue`: 1,704 statues of 1.43M
triangles and 1,188 floor patches of 1.62M, 4.36 billion in all; 1280x720,
1.5 px, mean of the 8 film cameras in `examples/statue_avenue/cameras.json`):

| | Triangles drawn |
|---|---|
| Before phase 1 | 2.77M |
| After phase 1 so far: Coarsen / Sun Shadows / Cull | 1.78M / 1.57M / 1.51M |
| Ideal, meshlet cut on every copy, off-screen coarsened | 1.67M |
| Ideal, meshlet cut on every copy, off-screen skipped (T1 reference) | 1.24M |

On the synthetic field (`tools/instancing_bench.py --synthetic`: 1,500 copies of
an 82k-triangle rock, 1 px) the gap is much larger, because no single copy is
heavy enough for a streamed copy to matter: 15.96M drawn (Coarsen), 14.31M
(Cull), against 7.13M and 5.91M for the ideal. Whole-copy levels cost about
twice a meshlet cut: a level has to be fine enough for the copy's nearest point,
and levels halve in size, so the chosen one is up to 2x finer than needed.
Phase 2 is what closes this.

Known problems:

- During the demo film render (headless bpy 5.1.2, EEVEE in software GL, one
  still render per frame in one session) memory grew 0.3-0.6 GB per frame; the
  film had to be rendered in 8-frame processes. Not yet known whether the
  add-on (chunk meshes, back buffers) or EEVEE's software path holds it.
- A streamed copy's first assignment creates its chunk objects (one relations
  rebuild, ~0.1 s in EEVEE).
- Building needs the full mesh in Blender's memory.
- Rare back-to-back "fins" from simplification at coarse levels.

## Nobody else does this for Blender's own renderers

As of November 2026: Octane has meshlets in its own render engine (also inside
Blender), but they work only in Octane. Unreal (Nanite), Unity packages, Bevy and
several WebGPU projects do virtual geometry inside their own renderers. The
Blender add-ons in the stores are classic discrete LOD generators and
distance-based LOD switchers. Blender itself made Vulkan the default backend on
Windows and Linux in 5.3 and is considering removing OpenGL in 6.0, but has no
virtual geometry of its own on the roadmap.

## The constraint that shapes everything

A Python add-on cannot feed EEVEE or Cycles from its own GPU buffers. Geometry
reaches them as meshes and instances through the depsgraph, and every changed
mesh is uploaded again. So the add-on wins by choosing on the CPU (in C++, in
`vgeo_stream`) and handing Blender as little as possible, in forms Blender
handles cheaply: instances of meshes that never change, rather than meshes that
change every frame.

---

## Phase 1: quick wins

- [x] Animation renders cut each frame from the render's own depsgraph (the
      cut stayed on the first frame's camera).
- [x] `tools/instancing_bench.py`: what each instancer draws against the ideal,
      per camera, any `.blend` or a synthetic field.
- [x] Streamed copies go to the copies whose level meshes cost the most, at any
      level, with hysteresis; Streamed Copies default 8, max 64.
- [x] Instancer Off-screen modes: Coarsen, Sun Shadows (keeps copies whose sun
      shadow can reach the view), Cull. Pixel-identical to Coarsen in the
      shadow check (`docs/images/offscreen_modes.jpg`).
- [x] `examples/statue_avenue`: the film's scene as a reproducible benchmark.
- [x] Memory growth over long renders: `tools/render_memory_bench.py`, 100
      frames of the statue avenue in one session, EEVEE and Cycles on a GPU.
      Memory warms up over the first ~50 frames, then stays flat (second half
      -7.6 % to +1.7 %); frozen runs separate the add-on from the engine. The
      film's per-frame growth was EEVEE's software-GL path. T3 holds on a GPU.
- [x] Viewport check on a GPU (`tests/viewport_bench.py --cameras
      examples/statue_avenue/cameras.json --seconds 24`, 2100x1341, 1 px, RTX
      A4500), flying the film path: before PR #2 Solid 35.5 / EEVEE 15.9 fps
      (worst 120 ms); with PR #2 (Shadows, 8 streamed copies) 17.5 / 13.0 fps
      (worst 264 ms). More streamed copies cost the viewport more than their
      saved triangles gain; phase 2 (53.6 / 23.6 fps) replaces them.
- [x] New instancers (Scatter, `create_instancer`) use Shadows; the property
      default stays Coarsen, so existing files render what they did.
- [x] Shadows mode (was Sun Shadows): lamps scanned once per tick; copies
      within reach of a shadow-casting point or spot lamp whose light reaches
      the view are kept (custom distance, else EEVEE's influence radius; spot
      cone tested).

## Phase 2: a meshlet cut on every copy

The big one. Today a copy is either a whole level mesh (cheap to update, about
2x too many triangles) or a streamed copy (exact, but each one is a set of
unique chunk meshes, so only a handful are affordable).

Design: instance the pieces of the cut instead of the copies.

1. **Cut atoms.** In the cluster DAG, the clusters made by simplifying one group
   are always selected together (they share their own error and their parent's
   error). Each such set is an atom: a fixed piece of the asset at a fixed
   detail. A 1.4M-triangle asset has a few thousand atoms. Build each atom once
   as a small Blender mesh, in a hidden collection per asset (the same place
   the level meshes live today).
2. **Per-copy cut in C++.** Each update, `vgeo_stream` selects the cut for every
   copy (the benchmark already does 2,900 selections per view in well under a
   second) and returns (copy, atom) pairs. New entry point:
   `vgeo_select_instances(asset, views, transforms[n], out_pairs)`, with frustum
   culling per atom, so a copy that is half in view sends only its visible
   half.
3. **One point cloud, one Instance on Points.** The pairs become points with
   the copy's transform and an atom index; Geometry Nodes picks the atom mesh by
   index. Updating means rewriting point attributes, never rebuilding meshes,
   and EEVEE and Cycles see instances of meshes that never change.
4. **Unify.** A proxy is an instancer with one copy, so single objects can use
   the same path; streamed copies and chunk meshes then go away. Keep the chunk
   path until the new one is faster everywhere.

Spike first (a few days), because everything depends on Blender handling it:

- 50k-500k instances of 4k distinct small meshes, the point attributes
  rewritten every tick.
- Measure: Geometry Nodes evaluation time, EEVEE frame time and the cost of a
  changed instance list, Cycles BVH build time, memory.
- If it is too slow: batch atoms into larger blocks (fewer, bigger instances),
  update only the copies whose cut changed, or split the point cloud per region
  so a change touches one small object.

Done when: T1 on both benchmark scenes, viewport fps at least as good as today
(T2), memory flat (T3), and the existing crack tests pass for instanced cuts.

### Spike result (November 2026): go, with two mitigations

Measured on the statue avenue and the synthetic field (RTX A4500, Blender 5.1.2):

- Atoms are real: on every copy, the atom cut equals the per-cluster cut to the
  triangle (`tools/atoms_check.py`, 8 cameras; 200 random views per asset in the
  spike). Statue 4,542 atoms, floor patch 5,155, 128 / 384 / 1,920 triangles
  (10th / median / 90th percentile).
- A view needs few instances: 1.8-3k (copy, atom) pairs per avenue instancer,
  20-28k on the synthetic field.
- Mitigation 1, no atom objects: with ~10k atom objects in the scene, every
  depsgraph update cost ~95 ms, even for an unrelated mesh. Instead, each slice
  of 512 atoms is one mesh (atom index as a face attribute) on a library object
  whose Split to Instances runs once; each slice has a point cloud that reads it
  with Object Info + Pick Instance. A full avenue cut change: ~10 ms depsgraph.
- Mitigation 2, an instance budget: EEVEE pays ~2 us per instance per frame.
  In the viewport an instancer draws at most 8,000 instances: copies start as
  one whole level mesh and switch to their atoms in order of triangles saved
  per extra instance, when each extra instance saves at least 1,000 triangles
  (synthetic field, EEVEE flying at 2100x1341: 17.7 fps atoms only, 43.5 fps
  with the budget; at 1048x683, 71.9 fps with the minimum gain).
- Parent-merged blocks would cut instances by another 33-40 % at no triangle
  cost (not built yet).

Implemented as **Meshlet Instancing** on instancers (experimental, off by
default; branch `phase2-atoms`):

- [x] `vgeo_atoms`, `vgeo_extract_atom`, `vgeo_select_instances` (one call, all
      copies, Off-screen modes per atom, multithreaded): 2-7 ms per instancer.
- [x] Sliced object-free atom libraries, one point cloud per slice, spare points.
- [x] Live loop: paced updates (at most every 0.2 s while the view moves, at
      once when it stops); per-copy level fallback under an instance budget.
- [x] Tests: atom cut = per-cluster cut, watertight instanced cut, culled atoms
      leave the instancing, final render cut, save/load, switching back.
- [x] Parent-merged blocks: the atoms of one parent group drawn as one piece
      when all are in a copy's cut (statue avenue: 17-33 % fewer instances, same
      triangles). Off by default: one more copy of the DAG's geometry in Blender
      (+0.64 GB, .blend 303 -> 397 MB) for an fps gain within noise.
- [x] T4: the statue avenue renders with meshlet instancing at 3.3-3.4 GB peak
      RSS (today 1.7-2.2 GB); EEVEE 0.65 s / Cycles 1.30 s a frame at 1280x720,
      16 samples (today 0.57 / 1.47).
- [ ] Proxies as single-copy instancers; retire chunk meshes and streamed copies
      (needs the owner's go).

T1, triangles drawn (`tools/instancing_bench.py`, 1280x720, the bench's error):

| | Today (Shadows / Cull) | Atoms (Shadows / Cull) | cut+cull |
|---|---|---|---|
| Statue avenue, mean of 8 cameras, 1.5 px | 1,572,264 / 1,505,053 | 1,316,119 / 1,208,493 | 1,243,178 |
| Synthetic field, 1 px | 14,424,403 / 14,309,348 | 6,061,381 / 5,919,547 | 5,911,265 |

Atoms draw within 6 % of cut+cull in Shadows mode and at or under it in Cull
(whole copies outside the view are skipped by the asset's tighter bounds).

T2, GUI viewport flying the film path (`tests/viewport_bench.py --cameras ...
--offscreen SHADOWS --seconds 24`, 2100x1341 region, 1 px, RTX A4500):

| | Solid flying | EEVEE flying (worst frame) | EEVEE still |
|---|---|---|---|
| Statue avenue, today | 17.5 fps | 13.0 fps (264 ms) | 24.6 fps |
| Statue avenue, atoms | 53.6 fps | 23.6 fps (96 ms) | 42.7 fps |
| Synthetic field, today | 33.9 fps | 21.6 fps (131 ms) | 52.2 fps |
| Synthetic field, atoms, 8k instance budget | 92.2 fps | 43.5 fps (44 ms) | 53.7 fps |

At a 1280x800 window (1048x683 region, about the bench's 720p), with blocks
(`--window-geometry 0 0 1280 800`):

| | Solid flying | EEVEE flying (worst frame) | EEVEE still |
|---|---|---|---|
| Statue avenue, today | 16.3 fps | 13.1 fps (210 ms) | 30.5 fps |
| Statue avenue, atoms | 64.2 fps | 36.3 fps (75 ms) | 53.7 fps |
| Synthetic field, today | 54.4 fps | 36.8 fps (100 ms) | 83.6 fps |
| Synthetic field, atoms | 93.2 fps | 45.1 fps (44 ms) | 52.8 fps |
| Statue avenue, atoms, min gain 1,000 | 63.5 fps | 38.5 fps (79 ms) | 57.6 fps |
| Statue avenue, atoms, min gain 1,000, no blocks | 86.2 fps | 50.5 fps (66 ms) | 78.8 fps |
| Synthetic field, atoms, min gain 1,000 | 114.5 fps | 71.9 fps (39 ms) | 81.6 fps |

With the instance budget alone, the synthetic field's still EEVEE frame was
slower than today's (8k instances cost more than the triangles they saved in a
small viewport). A copy now also switches to atoms only when each extra
instance saves at least 1,000 triangles: T2 holds on both scenes, and no mode
is slower than today.

Run-to-run spread of EEVEE numbers on this remote machine is about 20 %. The
synthetic field first ran with a fixed 300-triangle cost per instance (29.5
fps flying, 26.7 still: 20k+ instances); the budget keeps EEVEE still frames at
today's speed.

Final renders (EEVEE, 1280x720, 1.5 px): Shadows mode is pixel-identical to
Coarsen with atoms, as without; atoms against today's path differ by 0.13-0.2
/255 on average (fine carved detail on mid-distance statues, where whole-copy
levels draw finer than needed).

## Phase 3: occlusion culling

Skip copies (and, with phase 2, atoms) hidden behind others: rows of statues and
dense forests hide most of what is in the frustum.

- In `vgeo_stream`: rasterize coarse occluders (the coarsest levels of the
  nearest copies and large proxies) into a small depth buffer on the CPU (about
  256x144), build a depth pyramid, and test each copy's or atom's bounds
  against it. Masked software occlusion culling (Intel's published technique)
  is the reference; check the license before borrowing code.
- Shadows: a copy hidden from the camera can still cast a visible shadow. In
  Sun Shadows mode, drop an occluded copy only if its swept shadow bounds are
  occluded too; in Coarsen mode, occluded copies are coarsened, not dropped.
- Done when: at least 30% fewer triangles than phase 2 alone on ground-level
  views of the statue avenue, with no pixel differences in the shadow check
  (render with and without, diff).

Result (November 2026): built (`vgeo_occlusion_*`, the instancer's Occlusion
Culling option, off by default), but it saves little on the benchmark scenes,
so the 30 % target is not met:

| Cull mode, triangles drawn | Atoms | + occlusion (256x144) | + ideal buffer (1280x720, exact, no erosion) |
|---|---|---|---|
| Statue avenue, mean of 8 cameras | 1,208,493 | 1,175,400 (-2.7 %) | 1,137,881 (-5.8 %) |
| Synthetic field, mean of 3 cameras | 5,919,547 | 5,793,349 (-2.1 %) | 5,597,985 (-5.4 %) |

Even with a near-ideal buffer the gain is ~6 %: level of detail already makes
what hides behind the nearest copies cheap (distant, coarse), and atoms are
tested by bounding spheres. Building the buffer costs 20-30 ms per update.
Worth revisiting for interiors or dense forests seen from inside; not for open
scenes like these.

## Phase 4: the product

- **Any scatter.** Turn any Geometry Nodes scatter, particle system or
  collection instance setup into a VGEO instancer ("Virtualize Instances"), or
  ship a node group asset that users drop into their own trees.
- **Debug views** built in: Meshlets (a colour per cluster), LOD (an ordered
  ramp from full detail to coarsest), Triangles (flat facets), plus a stats
  overlay (scene triangles, drawn, meshlets, streamed copies). The demo film
  did this by packing `(cluster_id << 6) | depth` into the runtime's per-face
  LOD and writing a corner colour attribute `vgeo_viz` (RGB = cluster hash,
  A = depth / 16) that an Attribute node reads; the statue avenue materials
  already have the switch.
- **Asset Browser**: virtualized assets as drag-and-drop assets.
- **Import without Blender's memory**: build from OBJ, PLY, glTF or USD on disk,
  tiled for sources larger than RAM (photogrammetry scans).
- **Packaging**: Blender Extensions platform; CI builds of `vgeo_stream` for
  Windows, macOS (arm64, x64) and Linux; headless tests in CI with the bpy
  module.
- **Export**: Unreal (Nanite-ready source meshes and instance transforms via
  USD), web (`.vgeow`, exists).
- **Docs and demo scenes**: statue avenue, asteroid belt, terrain; short videos.

## Phase 5: GPU-driven viewport (research)

- A viewport mode that draws clusters straight from GPU buffers, like
  `web/vgeo-viewer.js`, for laying out huge scenes; final renders keep using
  EEVEE and Cycles.
- First check what Blender's Python `gpu` module exposes on Vulkan (compute
  shaders, storage buffers, indirect draws). If it is not enough, the old
  Vulkan proof of concept (readback into a texture) is the fallback, or an
  upstream proposal for a draw-engine hook.

---

## Benchmarks and checks

| What | Command |
|---|---|
| Triangles against the ideal | `blender -b scene.blend --python tools/instancing_bench.py -- --cameras cams.json --offscreen COARSEN,SHADOWS,CULL` |
| Synthetic field | `blender -b --factory-startup --python tools/instancing_bench.py -- --synthetic` |
| Statue avenue | `blender -b --factory-startup --python examples/statue_avenue/build_scene.py -- OUT`, then the bench with `examples/statue_avenue/cameras.json` |
| Viewport | `blender scene.blend --python tests/viewport_bench.py -- --motion fly` |
| Tests | `blender -b --factory-startup --python tests/test_stream.py` |

Every change that affects what is drawn reports the bench before and after in
its PR. Culling changes also render one frame looking away from a low sun with
and without the change and diff the images.
