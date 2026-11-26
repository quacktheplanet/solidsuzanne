#!/bin/sh
# Renders the next unfinished chunks (pass a, then pass b) until a time budget is spent.
# Safe to call again and again: finished frames are skipped. Fresh process per chunk: in software GL
# memory grows with every rendered frame (keep STEP=8 there); on a GPU it stays flat, so bigger chunks
# save the per-process start-up. Usage: sh render_chunks.sh [budget_seconds]
# Env: PY (a plain python, reads the timeline), BLENDER (renders; default: PY with the bpy module),
#      STEP (frames per process, default 8), WORK, SCENE, VGEO_ADDONS (see render_frames.py).
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
WORK=${WORK:-work}; export WORK
STEP=${STEP:-8}; export STEP
BUDGET=${1:-420}
START=$(date +%s)
NEXT=$(cd "$D" && $PY - <<'PYEOF'
import os
import timeline as T
N, STEP = int(T.DURATION * T.FPS), int(os.environ.get("STEP", "8"))
OUT = os.path.join(os.path.abspath(os.environ.get("WORK", "work")), "frames")
def wipe(f):
    return bool(T.state((f + 0.5) / T.FPS)["viewB"])
todo = []
for f0 in range(0, N, STEP):
    fr = range(f0, min(f0 + STEP, N))
    if any(not (os.path.exists(f"{OUT}/{f:04d}_a.jpg") and os.path.exists(f"{OUT}/{f:04d}.json")) for f in fr):
        todo.append(f"a:{f0}:{min(f0 + STEP, N)}")
for f0 in range(0, N, STEP):
    fr = range(f0, min(f0 + STEP, N))
    if any(wipe(f) and not (os.path.exists(f"{OUT}/{f:04d}_b.jpg") and os.path.exists(f"{OUT}/{f:04d}_b.json"))
           for f in fr):
        todo.append(f"b:{f0}:{min(f0 + STEP, N)}")
print(" ".join(todo) if todo else "NONE")
PYEOF
)
[ "$NEXT" = "NONE" ] && { echo "ALL DONE"; exit 0; }
for job in $NEXT; do
  now=$(date +%s); [ $((now - START)) -gt "$BUDGET" ] && break
  pass=${job%%:*}; rest=${job#*:}; f=${rest%%:*}; e=${rest#*:}
  if [ -n "$BLENDER" ]; then
    "$BLENDER" -b --factory-startup --python "$D/render_frames.py" -- "$pass" "$f" "$e" 2>&1 | grep -E "CHUNK|Error|Traceback|Killed"
  else
    $PY "$D/render_frames.py" "$pass" "$f" "$e" 2>&1 | grep -E "CHUNK|Error|Traceback|Killed"
  fi
done
echo "chunks left before this call: $(echo $NEXT | wc -w)"
