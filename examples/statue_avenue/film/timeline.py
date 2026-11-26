"""Shot list for the Solid Suzanne Blender demo (Z up, avenue along +Y)."""
import math

FPS = 24
DURATION = 66.0
FRAMES = int(FPS * DURATION)
ERR = 1.0
ERR_MAX = 8.0

SHOTS = [("avenue", 0.0, 12.0), ("closeup", 12.0, 22.0), ("breakdown", 22.0, 57.0), ("finale", 57.0, 66.0)]
CUTS = (12.0, 22.0, 57.0)
WIPES = [(23.4, 25.4, "beauty", "meshlod"), (34.0, 36.0, "meshlod", "facetlod"), (45.6, 47.4, "facetlod", "meshlod")]
DIAL = (37.0, 39.2, 42.6, 44.8)
FREEZE = (45.5, 47.0, 54.5, 57.0)    # freeze from, pull-back start, pull-back end, end


def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp01(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def cruise(u, ramp=0.22):
    u = clamp01(u)
    v = 1 / (1 - ramp)
    if u < ramp:
        return 0.5 * v * u * u / ramp
    if u > 1 - ramp:
        w = 1 - u
        return 1 - 0.5 * v * w * w / ramp
    return 0.5 * v * ramp + v * (u - ramp)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp3(a, b, t):
    return tuple(lerp(x, y, t) for x, y in zip(a, b))


def catmull(pts, k):
    i = min(len(pts) - 2, int(math.floor(k)))
    s = k - i
    p0, p1, p2, p3 = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(len(pts) - 1, i + 2)]
    return tuple(0.5 * (2 * b + (-a + c) * s + (2 * a - 5 * b + 4 * c - d) * s * s + (-a + 3 * b - 3 * c + d) * s ** 3)
                 for a, b, c, d in zip(p0, p1, p2, p3))


# S4 fly-by along the right row at head height; S5 eye = its end
FLY = [(-1.0, 8.0, 6.0), (2.0, 60.0, 7.0), (4.0, 120.0, 7.4), (4.5, 168.0, 7.4)]
EYE = {"pos": (4.5, 168.0, 7.4), "target": (2.5, 260.0, 5.5), "fov": 62.0}


def state(t):
    st = {"roll": 0.0}
    if t < 12.0:
        u = cruise(t / 12.0, 0.25)
        st.update(pos=lerp3((-1.6, -62.0, 2.0), (-0.9, -24.0, 2.6), u),
                  target=lerp3((0.0, 200.0, 12.0), (0.0, 200.0, 11.0), u), fov=52.0, roll=0.012)
    elif t < 22.0:
        u = cruise((t - 12.0) / 10.0, 0.25)
        # right-row statue at y = 24 faces -X: slide past her sunlit face
        st.update(pos=lerp3((-1.0, 17.0, 4.9), (0.4, 29.5, 5.4), u),
                  target=lerp3((9.8, 22.8, 6.0), (9.8, 25.0, 6.2), u), fov=40.0)
    elif t < 34.0:
        u = cruise((t - 22.0) / 12.0, 0.3)
        st.update(pos=lerp3((-3.5, -18.0, 4.0), (-1.0, 8.0, 6.0), u), target=lerp3((1.0, 180.0, 3.0), (1.5, 200.0, 4.0), u), fov=58.0)
    elif t < FREEZE[0]:
        u = cruise((t - 34.0) / (FREEZE[0] - 34.0), 0.18)
        pos = catmull(FLY, u * (len(FLY) - 1))
        st.update(pos=pos, target=(lerp(1.5, 2.5, u), pos[1] + 92.0, lerp(4.0, 5.5, u)), fov=lerp(58.0, 62.0, u))
    elif t < FREEZE[3]:
        u = cruise((t - FREEZE[1]) / (FREEZE[2] - FREEZE[1]), 0.3)
        vantage, look = (-62.0, 110.0, 70.0), (6.0, 230.0, 0.0)
        st.update(pos=lerp3(EYE["pos"], vantage, u), target=lerp3(EYE["target"], look, u), fov=lerp(62.0, 66.0, u))
        st["eye"] = EYE
    else:
        u = cruise((t - 57.0) / 9.0, 0.2)
        st.update(pos=lerp3((-19.0, 44.0, 10.0), (-25.0, 34.0, 19.0), u),
                  target=lerp3((24.0, 400.0, 5.0), (26.0, 420.0, 9.0), u), fov=58.0, roll=-0.02)

    view, viewB, wipe = "beauty", None, None
    for (a, b, fr, to) in WIPES:
        if t >= b:
            view = to
        elif t >= a:
            view, viewB, wipe = fr, to, smoother((t - a) / (b - a))
            break
        else:
            break
    if t >= 57.0 or t < 22.0:
        view, viewB, wipe = "beauty", None, None
    err = ERR
    if DIAL[0] <= t < DIAL[3]:
        k = smooth((t - DIAL[0]) / (DIAL[1] - DIAL[0])) * (1 - smooth((t - DIAL[2]) / (DIAL[3] - DIAL[2])))
        err = math.exp(math.log(ERR) + (math.log(ERR_MAX) - math.log(ERR)) * k)
    st.update(view=view, viewB=viewB, wipe=wipe, errorPx=err)
    return st
