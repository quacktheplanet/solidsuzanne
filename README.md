<p align="center">
  <img src="docs/images/SolidSuzanne.png" width="180" alt="VGEO logo"/>
</p>

<h1 align="center">VGEO — Virtualized Geometry for Blender</h1>

<p align="center">
  Nanite-style cluster LOD for Blender's own renderers: billions of triangles in the viewport, EEVEE and Cycles.
</p>

---

VGEO turns dense meshes into a Nanite-style cluster hierarchy (built with
meshoptimizer's clusterlod into a memory-mapped `.vgeo` file) and hands Blender
only the crack-free, view-dependent cut each view needs, as ordinary meshes and
instances. EEVEE and Cycles render it with full materials, lights and shadows.
Nothing else does this for Blender's own renderers: other virtual geometry lives
inside its own engine.

- **Virtualize Mesh**: one object streams its cut into mesh chunks (a 33.5M
  triangle terrain streams at ~28 fps while moving).
- **Scatter Instances / Virtualize Instances**: thousands of copies of an asset,
  scattered by VGEO or taken over from your own Geometry Nodes, particle or
  collection-instance scatter. Each copy gets its own level of detail.
- **Meshlet Instancing** (experimental): every copy drawn as its own meshlet cut,
  built from instanced pieces of the asset that never change. On the 4.36-billion
  triangle statue avenue it draws within 6 % of a per-copy meshlet cut (the rule
  the browser viewers use) and flies at 50 fps in EEVEE, 86 fps in Solid (1280x800
  window, RTX A4500), against 13 and 16 fps for whole-copy levels.
- **Off-screen modes**: copies outside the view coarsened, culled, or kept only
  while their shadow can reach it.
- **Debug views**: Meshlets, LOD and Triangles, plus a stats overlay.
- **Web export**: the same assets in a WebGPU viewer (`web/`), paged over HTTP
  range requests.

How it works and every measurement: [docs/NATIVE_STREAMING.md](docs/NATIVE_STREAMING.md).
Where it is going: [docs/ROADMAP.md](docs/ROADMAP.md).

## Quick start

```
cmake -S . -B build-stream -G Ninja -DCMAKE_BUILD_TYPE=Release -DVGEO_STREAM_ONLY=ON
cmake --build build-stream
```

Needs a C++20 compiler (no Vulkan SDK, no Python headers); the library lands in
`addons/vgeo/bin/`. Install `addons/vgeo` as an add-on (or zip it with
`tools/package_addon.py`), save your .blend, select a mesh and run
**Object > Virtualize Mesh**. The VGEO sidebar tab has everything else.

Tests and benchmarks (Blender 5.1):

```
blender -b --factory-startup --python tests/test_stream.py
blender -b --factory-startup --python examples/statue_avenue/build_scene.py -- OUT
blender -b OUT/statue_avenue.blend --python tools/instancing_bench.py -- --cameras examples/statue_avenue/cameras.json --error 1.5 --offscreen SHADOWS,CULL
```

---

## The original Vulkan proof of concept

The sections below describe where the project started: a headless Vulkan
meshlet renderer whose pixels are copied into the viewport (`addons/vgeo_blender`).

---

## What it does

VGEO converts meshes into a compact meshlet format and renders them live inside Blender's viewport using a headless Vulkan renderer. The current proof-of-concept renders full meshlet color debug visualization — each cluster gets a unique color, matching the kind of view Unreal's Nanite debug mode shows.

<p align="center">
  <img src="docs/images/meshletsTEST.png" width="720" alt="Meshlet colorization in Blender viewport with N-panel"/>
</p>

*Meshlet color debug view with the VGEO N-panel. Blender render engine set to VGEO.*

---

## Current state

- Convert any Blender mesh to `.vgeo` with one button
- Live viewport rendering via headless Vulkan → pixel readback → Blender GPU texture
- Meshlet color debug mode (toggle in the N-panel)
- Camera orbit, pan, and zoom tracked from Blender's viewport
- Proxy object transform tracked — move/rotate the proxy and the render follows
- Scene stats panel showing meshlet and cluster counts

---

## How to use

### Build

```
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Requires Vulkan SDK, CMake 3.20+, and a C++20 compiler. Builds `vgeo_build.exe` and `vgeo_native.pyd` (Python extension for Blender).

### Install the addon

1. In Blender: **Preferences → File Paths → Scripts** — point it to the repo root (`NewRepo/`)
2. **Preferences → Add-ons** — search for "VGEO" and enable it
3. Copy `build/src/python/Release/vgeo_native.pyd` next to the addon's `__init__.py`

### Convert and view

1. Select a mesh in Blender
2. Open the **N-panel → VGEO → Convert Active Mesh → VGEO**
3. Save the `.vgeo` file anywhere
4. Switch the render engine to **VGEO** (Properties → Render → Render Engine)
5. Toggle **Meshlet Colors** in the N-panel to see cluster visualization

---

## Architecture

```
Blender mesh
    │
    ▼
vgeo_build.exe        ← meshlet generation via meshoptimizer
    │
    ▼
.vgeo file            ← compact binary: meshlets + cluster BVH + indices
    │
    ▼
vgeo_native.pyd       ← pybind11 C++ extension loaded by Blender
    │
    ▼
VGEORenderEngine      ← Blender RenderEngine subclass
    │
    ├── vgeo_native.Scene    — scene graph, object transforms
    ├── vgeo_native.Camera   — synced from Blender viewport each frame
    └── vgeo_native.Renderer — headless Vulkan renderer
            │
            └── pixel readback → gpu.texture → draw_texture_2d
```

---

## Project structure

```
addons/vgeo_blender/     Blender addon (engine, operators, panels)
src/
  core/                  Camera, scene graph, math utilities
  preprocess/            Mesh → .vgeo converter (meshlet builder)
  runtime/               Headless Vulkan offscreen renderer
  python/                pybind11 bindings
tools/vgeo_build/        CLI converter tool
src/runtime/shaders/     GLSL shaders (compiled to SPIR-V at build time)
spec/                    Format specifications
```

---

## Requirements

- GPU with Vulkan 1.3+
- Windows 10+ (Linux/macOS via MoltenVK untested)
- Blender 4.0+
- CMake 3.20+, C++20 compiler, Vulkan SDK

---

## Acknowledgments

Built on insights from Epic Games' Nanite, [meshoptimizer](https://github.com/zeux/meshoptimizer) by Arseny Kapoulkine, and NVIDIA mesh shader research.
