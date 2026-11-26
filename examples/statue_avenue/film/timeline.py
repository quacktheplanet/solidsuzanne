"""Shot list for the Solid Suzanne phase 1 demo (Z up, avenue along +Y, low sun toward -X/+Y).

state(t) returns everything a frame needs: camera, view (material mode + Off-screen mode + tint),
the incoming view and wipe progress during wipes, the error budget, and the frozen "eye" camera
whose cut is kept while the camera flies out to look at it.
"""
import math

FPS = 24
DURATION = 74.0
FRAMES = int(FPS * DURATION)
ERR = 1.0

# (name, start, end): segments the compositor encodes separately (cuts between them)
SHOTS = [("open", 0.0, 9.0), ("closeup", 9.0, 16.0), ("flight", 16.0, 47.0), ("frozen", 47.0, 59.0),
         ("payoff", 59.0, 66.5), ("finale", 66.5, 74.0)]
CUTS = (9.0, 16.0, 47.0, 59.0, 66.5)

# views: material mode (0 beauty, 1 meshlets + LOD ramp), the instancers' Off-screen mode, tint on
# the streamed copies (0/1)
VIEWS = {
    "beauty": (0, "COARSEN", 0),
    "meshlod": (1, "COARSEN", 0),
    "tint": (0, "COARSEN", 1),
    "lod_coarsen": (1, "COARSEN", 0),
    "lod_shadows": (1, "SHADOWS", 0),
    "lod_cull": (1, "CULL", 0),
    "sun_cull": (0, "CULL", 0),
    "sun_shadows": (0, "SHADOWS", 0),
}
# (start, end, from view, to view); between wipes the last "to" view holds
WIPES = [(16.8, 18.6, "beauty", "meshlod"),
         (30.0, 31.6, "meshlod", "tint"),
         (39.6, 41.2, "tint", "meshlod"),
         (52.4, 53.8, "lod_coarsen", "lod_shadows"),
         (55.6, 57.0, "lod_shadows", "lod_cull"),
         (61.6, 63.4, "sun_cull", "sun_shadows")]
FIRST_VIEW = {"open": "beauty", "closeup": "beauty", "flight": "beauty", "frozen": "lod_coarsen",
              "payoff": "sun_cull", "finale": "beauty"}

# sections: (start, end, key) - the compositor holds their titles and captions
SECTIONS = [(17.0, 30.0, "meshlets"), (30.0, 40.0, "streamed"), (40.0, 47.0, "before"),
            (47.2, 59.0, "offscreen"), (59.2, 66.5, "payoff")]

FREEZE = (47.0, 47.6, 51.2, 59.0)   # freeze at, pull-back start, pull-back end, end


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


# the flight: down the avenue (breakdown), then close along the right row past the statues' heads
# (streamed copies), then on down the avenue (before/after); its end is the frozen eye
FLIGHT = [(-3.0, -20.0, 4.0), (-1.0, 10.0, 5.5), (3.5, 50.0, 6.4), (6.2, 92.0, 6.2), (6.4, 132.0, 6.2),
          (4.5, 176.0, 7.0), (3.0, 214.0, 7.4)]
EYE = {"pos": (3.0, 214.0, 7.4), "target": (1.5, 306.0, 5.5), "fov": 62.0}
VANTAGE, VANTAGE_LOOK = (-88.0, 150.0, 92.0), (10.0, 262.0, 0.0)
# the payoff: ground level in the avenue, looking away from the low sun (toward +X/-Y), the statues
# behind the camera throw long shadows into the view
# (found by rendering candidates in both modes and keeping the one where Cull loses the most shadow)
PAYOFF_A = {"pos": (0.0, 344.0, 2.2), "target": (16.9, 307.7, -0.2), "fov": 60.0}
PAYOFF_B = {"pos": (0.5, 337.0, 2.4), "target": (17.4, 300.7, 0.0), "fov": 60.0}


def flight_state(t):
    u = cruise((t - 16.0) / (47.0 - 16.0), 0.12)
    k = u * (len(FLIGHT) - 1)
    pos = catmull(FLIGHT, k)
    ahead = catmull(FLIGHT, min(len(FLIGHT) - 1.0, k + 0.35))
    look = (lerp(1.0, 1.5, u), ahead[1] + 92.0, lerp(3.5, 5.5, u))
    # past the statues (30-40 s) the camera turns toward the right row, where the streamed copies are
    turn = smooth((t - 29.0) / 2.5) * (1 - smooth((t - 38.5) / 2.5))
    look = lerp3(look, (12.0, pos[1] + 26.0, 4.6), 0.62 * turn)
    return {"pos": pos, "target": look, "fov": lerp(58.0, 62.0, u)}


def state(t):
    st = {"roll": 0.0}
    if t < 9.0:
        u = cruise(t / 9.0, 0.25)
        st.update(pos=lerp3((-1.6, -64.0, 1.8), (-1.0, -30.0, 2.7), u),
                  target=lerp3((0.0, 200.0, 12.0), (0.0, 200.0, 11.0), u), fov=52.0, roll=0.012)
    elif t < 16.0:
        u = cruise((t - 9.0) / 7.0, 0.25)
        # right-row statue at y = 24 faces -X: slide past her sunlit face
        st.update(pos=lerp3((-1.0, 17.0, 4.9), (0.4, 29.5, 5.4), u),
                  target=lerp3((9.8, 22.8, 6.0), (9.8, 25.0, 6.2), u), fov=40.0)
    elif t < FREEZE[0]:
        st.update(flight_state(t))
    elif t < FREEZE[3]:
        u = cruise((t - FREEZE[1]) / (FREEZE[2] - FREEZE[1]), 0.3)
        st.update(pos=lerp3(EYE["pos"], VANTAGE, u), target=lerp3(EYE["target"], VANTAGE_LOOK, u),
                  fov=lerp(62.0, 66.0, u))
        st["eye"] = EYE
    elif t < 66.5:
        u = cruise((t - 59.0) / 7.5, 0.3)
        st.update(pos=lerp3(PAYOFF_A["pos"], PAYOFF_B["pos"], u), target=lerp3(PAYOFF_A["target"], PAYOFF_B["target"], u),
                  fov=60.0)
    else:
        u = cruise((t - 66.5) / 7.5, 0.2)
        st.update(pos=lerp3((-19.0, 44.0, 10.0), (-25.0, 34.0, 19.0), u),
                  target=lerp3((24.0, 400.0, 5.0), (26.0, 420.0, 9.0), u), fov=58.0, roll=-0.02)

    shot = next(name for name, a, b in SHOTS if a <= t < b) if t < DURATION else SHOTS[-1][0]
    view, viewB, wipe = FIRST_VIEW[shot], None, None
    s0, s1 = next((a, b) for name, a, b in SHOTS if name == shot)
    for (a, b, fr, to) in WIPES:
        if not (s0 <= a < s1):
            continue
        if t >= b:
            view = to
        elif t >= a:
            view, viewB, wipe = fr, to, smoother((t - a) / (b - a))
            break
        else:
            break
    st.update(shot=shot, view=view, viewB=viewB, wipe=wipe, errorPx=ERR)
    return st
