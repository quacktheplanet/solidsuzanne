# Roadmap: Nanite-style geometry for Blender

Goal: the best virtual-geometry add-on for Blender, at least as efficient as the
browser version, with billion-triangle scenes that work in the viewport, in
EEVEE and in Cycles.

Status: November 2026. Phase 1 has started (see the checklist below).

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
- [ ] Memory growth over long renders: render 100 frames of the statue avenue
      in one session with EEVEE (GPU) and Cycles, log RSS per frame, and find
      what holds memory (chunk meshes of released streamed copies, back
      buffers, mesh datablocks left with zero users, EEVEE caches). Done when
      T3 holds.
- [ ] Viewport check of the new modes on a GPU (`tests/viewport_bench.py`),
      Solid and EEVEE, flying through the statue avenue. Record fps and worst
      frame before and after.
- [ ] Decide the default Off-screen mode for new instancers (Sun Shadows is
      the candidate; existing files keep Coarsen).
- [ ] Sun Shadows: cache the sun list per tick, and keep copies within range
      of shadow-casting point and spot lights (sphere against light radius).

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
