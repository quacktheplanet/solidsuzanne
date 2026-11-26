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

## Acknowledgments

Built on insights from Epic Games' Nanite, [meshoptimizer](https://github.com/zeux/meshoptimizer) by Arseny Kapoulkine, and NVIDIA mesh shader research.

---

## License

VGEO is free and open source software under the [GNU General Public License
v3.0 or later](LICENSE), the GPL like Blender itself: use it for anything,
commercial work included; if you distribute a modified version, share its source
under the same terms. Third-party components are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Support development

VGEO is built in spare time. If it is useful to you, you can
[sponsor it on GitHub](https://github.com/sponsors/quacktheplanet); donations decide
what comes next on the [roadmap](docs/ROADMAP.md), such as macOS and Linux builds.
