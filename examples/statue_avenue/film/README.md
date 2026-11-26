# Statue avenue film pipeline

This folder holds the scripts that made the "Solid Suzanne in Blender" tech demo: 66 s at
1280x720 and 24 fps, rendered offline in EEVEE. The film moves from cinematic shots to breakdowns:
meshlet colours, an ordered LOD ramp and live triangle counts. Use it as the starting point for new
demo films: change the timeline and captions, and keep the machinery.

| File | What it does |
|---|---|
| `timeline.py` | Every timing: shots and camera paths, view mode per moment (beauty, meshlets, facets), wipes, the error-budget dial, the frozen cut and its observer camera. `state(t)` returns everything for time t. |
| `render_frames.py` | Renders frames in one Blender session. It places the camera, sets the error budget and the material view, and freezes the cut for the observer shots. Per frame it writes `NNNN_a.jpg` (and `_b.jpg` for wipes) plus `NNNN.json` with the triangle stats and overlay positions. |
| `render_chunks.sh` | Renders the next unfinished 8-frame chunks until a time budget runs out. It is resumable: call it until it prints `ALL DONE`. |
| `compose.py` | Composites the frames with PIL: wipes, bloom, star flare, section titles, captions, stats panel, LOD legend, frustum overlay and end card. Then it encodes one MP4 per segment. `--still N` composites one frame to `WORK/still_NNNN.png`. |
| `audio.py` | A synthesized score, with its hits on the timeline's cuts, wipes and dial. |
| `assemble.sh` | Joins the segments and adds the score. |
| `demo_viz.patch` | The debug-view hack for the breakdown colours, applied to a scratch copy of the add-on. It never goes on a real branch. |

## Run it

```sh
# 1. a scratch copy of the add-on with the breakdown attributes, and its native library
git worktree add --detach ../ss-demo HEAD
(cd ../ss-demo && git apply examples/statue_avenue/film/demo_viz.patch &&
 cmake -S . -B build-demo -DVGEO_STREAM_ONLY=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-demo)
export VGEO_ADDONS=$(realpath ../ss-demo/addons) PY=python3   # a Python with bpy 5.1
export WORK=$(realpath -m film_work) SCENE=$(realpath -m sa_demo/statue_avenue.blend)

# 2. the scene, built with the patched add-on (the level meshes need the attributes too)
$PY examples/statue_avenue/build_scene.py -- sa_demo

# 3. look before rendering everything: a few key frames, composited
$PY examples/statue_avenue/film/render_frames.py a 620 621
$PY examples/statue_avenue/film/compose.py --still 620

# 4. the full render; repeat until it prints ALL DONE
sh examples/statue_avenue/film/render_chunks.sh 420

# 5. score, segments, final file
$PY examples/statue_avenue/film/audio.py
$PY examples/statue_avenue/film/compose.py
sh examples/statue_avenue/film/assemble.sh statue-avenue-demo.mp4 s1 s2 s3 s4
```

## Things to know

- **Cost.** EEVEE in software GL (2 CPUs, no GPU) takes about 14 s per frame at 720p, so the
  66 s film was about 6.5 hours including the wipe pass. With a GPU it is minutes. Sun shadows
  are the biggest cost; the floor patches do not cast shadows in the film.
- **Memory.** Memory grows with every frame rendered in one session (0.3-0.6 GB a frame in software
  GL), so `render_chunks.sh` starts a fresh process every 8 frames. A run that gets killed just
  leaves frames missing, and the next call renders them.
- **Freezing the cut.** For the observer shots, `render_frames.py` computes the instancer levels for
  the frozen "eye" view and then sets `stream.FREEZE = True`, so the render handlers keep that cut
  while the camera flies out to look at it.
- **Numbers.** Every number on screen comes from `NNNN.json`, which records what the add-on drew
  for that frame. Change the add-on and the numbers change with it. Never type numbers into the
  captions.
- **Breakdown colours.** The patch packs `(cluster_id << 6) | depth` into the runtime's per-face LOD
  and writes a corner BYTE_COLOR `vgeo_viz` (RGB = cluster hash, A = depth / 16). The statue
  materials read it in modes 1 and 2. Proper built-in debug views are on the roadmap (phase 4).
