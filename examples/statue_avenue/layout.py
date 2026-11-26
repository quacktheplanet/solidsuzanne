"""Scene layout shared by the builder, the timeline and the renderer (Blender units = metres, Z up).
The avenue runs along +Y between two rows of monoliths facing each other."""
import math

import numpy as np

SUN_DIR = tuple(np.array([-0.62, 0.74, 0.20]) / np.linalg.norm([-0.62, 0.74, 0.20]))   # toward the sun
PLANET_R = 1500.0
_pd = np.array([0.22, 1.0, 0.11]); _pd /= np.linalg.norm(_pd)
PLANET_POS = tuple(_pd * 9500.0 + np.array([0, 0, -150.0]))

PATCH = 24.0
PATCH_N = 900
ROW_X = 12.0          # avenue rows at x = +-ROW_X
STEP_Y = 18.0
Y0, Y1 = -30.0, 720.0


def statue_placements():
    rng = np.random.default_rng(5)
    pos, rot, sc = [], [], []
    # the avenue: two rows facing the centre line
    for y in np.arange(Y0, Y1, STEP_Y):
        for side in (-1, 1):
            pos.append((side * ROW_X, y, 0.0))
            yaw = (-math.pi / 2 if side < 0 else math.pi / 2) + rng.normal(0, 0.06)
            # Suzanne faces -Y; yaw turns her toward the avenue
            rot.append((0.0, 0.0, yaw + math.pi))
            sc.append(1.0)
    # the plain: a loose grid of monoliths out to the horizon, facing the avenue
    for gx in range(-16, 17):
        if abs(gx) < 2:
            continue
        x = gx * 17.0 + rng.normal(0, 1.5)
        for y in np.arange(Y0 - 60, Y1 + 200, 19.0):
            if abs(x) < 26:
                continue
            yy = y + rng.normal(0, 2.0)
            pos.append((x, yy, 0.0))
            rot.append((0.0, 0.0, (math.pi / 2 if x < 0 else -math.pi / 2) + math.pi + rng.normal(0, 0.25)))
            sc.append(float(rng.uniform(0.85, 1.15)))
    return np.array(pos), np.array(rot), np.array(sc)


def floor_placements():
    pos, rot, sc = [], [], []
    half = 13
    for ix in range(-half, half + 1):
        for iy in range(-4, 40):
            pos.append((ix * PATCH, iy * PATCH, 0.0))
            rot.append((0.0, 0.0, 0.0))
            sc.append(1.0)
    return np.array(pos), np.array(rot), np.array(sc)
