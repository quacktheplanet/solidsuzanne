# Statue avenue film pipeline

This folder holds the scripts behind the Solid Suzanne tech demos, rendered offline in EEVEE at
1280x720 and 24 fps. The current timeline is the phase 1 film (74 s): a cinematic open, a sculpted
close-up, a breakdown to meshlets and an ordered LOD ramp with live counts, the streamed copies
(tinted, tagged with their distance and how near they are), the same frames against the old code,
the Off-screen modes seen from above over a frozen cut, a low-sun shadow payoff (Cull against Sun
Shadows) and an end card. Use it as the starting point for new demo films: change the timeline and
captions, and keep the machinery.

| File | What it does |
|---|---|
| `timeline.py` | Every timing: shots and camera paths, views (material mode, Off-screen mode, tint on streamed copies) and the wipes between them, sections, the frozen cut and its observer camera. `state(t)` returns everything for time t. |
| `render_frames.py` | Renders frames in one Blender session. Every frame cuts explicitly before rendering (the camera's view, or the frozen eye's) at the view's Off-screen mode, with the streamed copies released first so the numbers do not depend on earlier cuts. It marks streamed copies' chunks (`vgeo_streamed`) for the tint it adds to the materials, and writes `NNNN_a.jpg` + `NNNN.json` (and `NNNN_b.jpg` + `NNNN_b.json` for the incoming view of wipes): triangle stats, every mode's count in the Off-screen and payoff shots, streamed-copy tags, overlay positions. |
| `before_stats.py` | The "before" numbers: run with the add-on of the old commit (a worktree of master at 239e3a9) on the same scene, it cuts every frame the way the old code would (Coarsen, its 4 streamed copies) and writes `WORK/before.json`. Stats only, nothing rendered. |
| `render_chunks.sh` | Renders the next unfinished chunks (`STEP` frames each, a fresh process per chunk) until a time budget runs out. It is resumable: call it until it prints `ALL DONE`. |
| `compose.py` | Composites the frames with PIL: wipes, bloom, star flare, section titles, captions, stats panel, LOD legend, frustum overlay and end card. Then it encodes one MP4 per segment. `--still N` composites one frame to `WORK/still_NNNN.png`. |
| `audio.py` | A synthesized score; it reads the cuts, wipes, mode switches and freeze from `timeline.py`. |
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

# 3. the old code's numbers for every frame (a worktree of master at 239e3a9, stats only)
git worktree add --detach ../ss-before 239e3a9    # plus its native library, built as in step 1
VGEO_ADDONS_BEFORE=$(realpath ../ss-before/addons) $PY examples/statue_avenue/film/before_stats.py

# 4. look before rendering everything: key frames (pass b for wipes), composited
$PY examples/statue_avenue/film/render_frames.py a 1500 1501
$PY examples/statue_avenue/film/render_frames.py b 1500 1501
$PY examples/statue_avenue/film/compose.py --still 1500

# 5. the full render; repeat until it prints ALL DONE
sh examples/statue_avenue/film/render_chunks.sh 420

# 6. score, segments, final file
$PY examples/statue_avenue/film/audio.py
$PY examples/statue_avenue/film/compose.py
sh examples/statue_avenue/film/assemble.sh statue-avenue-phase1.mp4 open closeup flight frozen payoff finale
```

On Windows with a Blender binary instead of the bpy module, run the Blender steps as
`blender -b --factory-startup --python SCRIPT -- ARGS`, and for `render_chunks.sh` set
`PY=python BLENDER=/path/to/blender.exe`. The compositor wants Inter (`FONT_DIR`, OFL, from
github.com/rsms/inter); without libraqm in Pillow it draws proportional instead of tabular figures.

## Things to know

- **Cost.** EEVEE in software GL (2 CPUs, no GPU) takes about 14 s per frame at 720p, so a
  66 s film was about 6.5 hours including the wipe pass. On an RTX A4500 the phase 1 film took
  about 2 s a frame (cuts included). Sun shadows are the biggest cost; the floor patches do not
  cast shadows in the film.
- **Memory.** In software GL memory grows with every frame rendered in one session (0.3-0.6 GB a
  frame), so `render_chunks.sh` starts a fresh process every `STEP` frames (8 there). On a GPU
  memory stays flat (`tools/render_memory_bench.py`), so `STEP=48` saves start-up time. A run that
  gets killed just leaves frames missing, and the next call renders them.
- **Finding a shot.** The payoff camera was chosen by rendering candidates in both Off-screen modes
  and keeping the one where Cull visibly loses the most shadow: a statue right beside the camera
  stays in view by its bounding sphere and casts its shadow in every mode.
- **Freezing the cut.** For the observer shots, `render_frames.py` computes the instancer levels for
  the frozen "eye" view and then sets `stream.FREEZE = True`, so the render handlers keep that cut
  while the camera flies out to look at it.
- **Numbers.** Every number on screen comes from `NNNN.json` / `NNNN_b.json`, which record what the
  add-on drew for that frame and view, or from `before.json` (the old code on the same frame). Change the add-on and the numbers change with it. Never type numbers into the
  captions.
- **Breakdown colours.** The patch packs `(cluster_id << 6) | depth` into the runtime's per-face LOD
  and writes a corner BYTE_COLOR `vgeo_viz` (RGB = cluster hash, A = depth / 16). The statue
  materials read it in modes 1 and 2. Proper built-in debug views are on the roadmap (phase 4).
