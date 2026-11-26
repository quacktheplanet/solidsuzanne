"""Check cut atoms against the per-cluster cut, copy by copy, and time vgeo_select_instances.

For each camera and instancer: the per-cluster cut of every copy (vgeo_select with the copy's local
view, as tools/instancing_bench.py's ideal does) against one vgeo_select_instances call over all
copies. With the frustum ignored or coarsening only (FULL, COARSEN) the two must match to the
triangle; with culling (CULL) atoms are culled by bounds enclosing their clusters, so they may keep a
little more than the per-cluster cut, never less.

Usage:
  blender -b scene.blend --python tools/atoms_check.py -- [--cameras cams.json] [--error 1.5] [--copies 300]
  blender -b --factory-startup --python tools/atoms_check.py -- --synthetic
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addons"))
sys.path.insert(0, os.path.join(REPO, "tools"))

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


import vgeo  # noqa: E402
from vgeo import instances, native, stream  # noqa: E402


def world_view(view, err, mode="FULL"):
    v, window, height, persp, clip_start = view
    planes = stream._frustum_planes(window @ v)
    return native.make_view(v.inverted().translation, window[1][1], max(clip_start, 1e-6), err / max(1, height),
                            planes=planes, mode=mode, offscreen_scale=8.0)


def local_view(m, view, err, mode):
    v, window, height, persp, clip_start = view
    scale = max(abs(s) for s in m.to_scale()) or 1.0
    cam_local = m.inverted() @ v.inverted().translation
    planes = stream._frustum_planes(window @ v @ m)
    return native.make_view(cam_local, window[1][1], max(clip_start / scale, 1e-6), err / max(1, height),
                            planes=planes, mode=mode, offscreen_scale=8.0)


def copy_transforms(inst, idx):
    mats = [inst.matrix_world @ instances.placement_matrix(inst, int(i)) for i in idx]
    tr = np.array([[list(m[r]) for r in range(3)] for m in mats], dtype=np.float32)
    sc = np.array([max(abs(s) for s in m.to_scale()) for m in mats], dtype=np.float32)
    return mats, tr, sc


def main():
    import instancing_bench
    if "--synthetic" in argv:
        cams = instancing_bench.synthetic_scene()
    else:
        vgeo.register()
        cams = json.load(open(arg("--cameras"))) if arg("--cameras") else None
    from instancing_bench import bench_camera
    scene = bpy.context.scene
    err = float(arg("--error", 1.5))
    max_copies = int(arg("--copies", 300))
    insts = [i for i in instances.instancers(scene) if not i.hide_render]
    ok_all = True
    for ci, spec in enumerate(cams or [None]):
        view = bench_camera(spec, scene) if spec else stream.camera_view(scene)
        for inst in insts:
            asset = native.Asset(stream.asset_path(inst.vgeo_inst.source))
            n = len(inst.data.vertices)
            idx = np.linspace(0, n - 1, min(n, max_copies)).astype(int)
            mats, tr, sc = copy_transforms(inst, idx)
            atoms = asset.atoms()
            line = {"camera": ci, "instancer": inst.name, "copies": len(idx), "atoms": len(atoms)}
            for mode in ("FULL", "COARSEN", "CULL"):
                per = []
                for m in mats:
                    asset.select([local_view(m, view, err, mode)])
                    per.append(int(asset.last.triangles))
                t0 = time.perf_counter()
                pairs, st = asset.select_instances([world_view(view, err)], tr, sc, mode=mode)
                ms = (time.perf_counter() - t0) * 1000
                got = np.bincount(pairs[:, 0], weights=atoms["tri_count"][pairs[:, 1]], minlength=len(idx))
                per = np.array(per)
                if mode == "CULL":
                    # whole copies outside the view are skipped by the asset's bounding sphere (tighter
                    # than the per-cluster spheres, which can poke into the view while no vertex does);
                    # every other copy keeps at least its per-cluster cut
                    info = asset.info
                    lo, hi = np.array(info["aabb_min"]), np.array(info["aabb_max"])
                    c_loc, r_loc = (lo + hi) * 0.5, np.linalg.norm(hi - lo) * 0.5 * 1.0001
                    planes = np.array(stream._frustum_planes(view[1] @ view[0]))
                    cw = tr[:, :, :3] @ c_loc + tr[:, :, 3]
                    inside = np.all(cw @ planes[:, :3].T + planes[:, 3] >= -(r_loc * sc)[:, None], axis=1)
                    ok = bool(np.all(got[inside] >= per[inside]) and np.all(got[~inside] == 0))
                    line["copies_outside"] = int((~inside).sum())
                    line["cluster_cut_inside"] = int(per[inside].sum())
                else:
                    ok = bool(np.array_equal(got.astype(np.int64), per))
                ok_all &= ok
                line[mode] = {"cluster_cut": int(per.sum()), "atom_cut": int(got.sum()),
                              "over_pct": round((got.sum() / max(1, per.sum()) - 1) * 100, 2),
                              "copies_differ": int((got.astype(np.int64) != per).sum()), "ok": ok,
                              "select_ms": round(ms, 2), "pairs": int(st.pairs)}
            # timing on every copy of the instancer
            _m, tr_all, sc_all = copy_transforms(inst, range(n))
            t0 = time.perf_counter()
            for _ in range(5):
                asset.select_instances([world_view(view, err)], tr_all, sc_all, mode="CULL")
            line["all_copies_cull_ms"] = round((time.perf_counter() - t0) * 200, 2)
            print("ATOMS " + json.dumps(line), flush=True)
            asset.close()
    print("ATOMS CHECK", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
