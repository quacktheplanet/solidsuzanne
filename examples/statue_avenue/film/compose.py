"""Composites the phase 1 statue avenue film: wipes, bloom, star flare, letterbox, titles, section captions,
live stats, LOD legend, streamed-copy tags, before/after, Off-screen mode tabs, frozen-frustum overlay,
end card. Every number on screen is read from the frames' JSON (render_frames.py) or WORK/before.json
(before_stats.py). Each segment is piped to ffmpeg and renamed only on success.

Usage: python compose.py [segment ...]       (default: all)
       python compose.py --still <frame> ...  (composited PNGs for review, WORK/still_NNNN.png)
Env:   WORK, FONT_DIR (Inter: Inter-*.otf and InterDisplay-*.otf)
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as T  # noqa: E402

WORK = os.path.abspath(os.environ.get("WORK", "work"))
FR = os.path.join(WORK, "frames")
OUTD = os.path.join(WORK, "segments")
os.makedirs(OUTD, exist_ok=True)
W, H, FPS = 1280, 720, T.FPS
FRAMES = T.FRAMES
SEGMENTS = [(name, int(round(a * FPS)), int(round(b * FPS))) for name, a, b in T.SHOTS]
CUTS = T.CUTS
RAQM = features.check("raqm")   # OpenType features (tabular figures) need libraqm

LOD_STOPS = [
    [0.97, 0.99, 1.00], [0.45, 0.93, 1.00], [0.20, 0.75, 1.00], [0.12, 0.50, 1.00], [0.25, 0.32, 1.00],
    [0.48, 0.26, 1.00], [0.70, 0.22, 0.95], [0.90, 0.20, 0.80], [1.00, 0.22, 0.55], [1.00, 0.26, 0.30],
    [1.00, 0.40, 0.15], [1.00, 0.56, 0.10], [1.00, 0.72, 0.12], [1.00, 0.86, 0.30],
]

F = os.environ.get("FONT_DIR", "/usr/share/fonts/opentype/inter/")


def font(name, size):
    try:
        return ImageFont.truetype(os.path.join(F, name), size)
    except OSError:   # Inter not installed: DejaVu, then Pillow's built-in font
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default(size)


FT_TITLE = font("InterDisplay-Light.otf", 74)
FT_SUB = font("Inter-Regular.otf", 25)
FT_CAP1 = font("Inter-Medium.otf", 34)
FT_CAP2 = font("Inter-Regular.otf", 24)
FT_SECNUM = font("Inter-SemiBold.otf", 19)
FT_SECT = font("InterDisplay-Medium.otf", 40)
FT_DESC = font("Inter-Regular.otf", 24)
FT_LBL = font("Inter-SemiBold.otf", 15)
FT_VAL = font("Inter-Medium.otf", 27)
FT_BIG = font("InterDisplay-Light.otf", 96)
FT_CMP = font("InterDisplay-Light.otf", 58)
FT_END2 = font("Inter-Regular.otf", 27)
FT_END3 = font("Inter-Regular.otf", 19)
FT_TAG = font("Inter-SemiBold.otf", 15)
FT_TAGV = font("Inter-Regular.otf", 14)

ACCENT = (120, 214, 255)
AMBER = (255, 170, 80)
WHITE = (245, 242, 236)
DIM = (200, 198, 196)
GREEN = (140, 230, 160)


def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def window(t, a, b, fin=0.8, fout=0.8):
    return smooth((t - a) / fin) * (1 - smooth((t - (b - fout)) / fout))


def fmt(n):
    return f"{int(round(n)):,}"


def mtris(n):
    return f"{n / 1e6:.2f}M"


# ---------------------------------------------------------------- stats
def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


STATS = [_load(f"{FR}/{f:04d}.json") for f in range(FRAMES)]
STATS_B = [_load(f"{FR}/{f:04d}_b.json") for f in range(FRAMES)]
_bp = os.path.join(WORK, "before.json")
BEFORE = {int(k): v for k, v in json.load(open(_bp)).items()} if os.path.exists(_bp) else {}


def shown(f):
    """The stats of the view on screen: during a wipe, the incoming view's once it covers half."""
    s = STATS[f]
    if s and s.get("viewB") and (s.get("wipe") or 0) > 0.5 and STATS_B[f]:
        return STATS_B[f]
    return s


def stat(f, key, default=0):
    s = shown(f)
    return s.get(key, default) if s else default


def first(key):
    for s in STATS:
        if s and s.get(key) is not None:
            return s[key]
    return None


def scene_tris():
    return first("sceneTris") or 0


def copies():
    s = next((s for s in STATS if s and s.get("statues")), None)
    return (s["statues"] + s["patches"]) if s else 0


def flight_frames():
    return [f for f in range(FRAMES) if STATS[f] and STATS[f].get("shot") == "flight" and f in BEFORE]


def avg_now_before():
    fs = flight_frames()
    if not fs:
        return 0, 0
    return (sum(STATS[f]["drawnTriangles"] for f in fs) / len(fs), sum(BEFORE[f] for f in fs) / len(fs))


def payoff_modes():
    """Off-screen mode counts on the payoff frame shown longest after the wipe (the last payoff frame)."""
    fs = [f for f in range(FRAMES) if STATS[f] and STATS[f].get("shot") == "payoff" and STATS[f].get("modes")]
    return STATS[fs[-1]]["modes"] if fs else None


# ---------------------------------------------------------------- image ops
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_U = _xx / W + 0.22 * (_yy / H - 0.5)
_cx, _cy = W / 2, H / 2
_VIG = (1 - 0.32 * np.clip(np.sqrt(((_xx - _cx) / _cx) ** 2 * 0.8 + ((_yy - _cy) / _cy) ** 2 * 0.6) - 0.3, 0, 1) ** 1.6)[..., None].astype(np.float32)


def wipe(a, b, p):
    front = -0.2 + 1.4 * p
    m = np.clip((front - _U) * W * 0.9 + 0.5, 0, 1)
    img = a * (1 - m[..., None]) + b * m[..., None]
    d = np.abs(_U - front) * W
    glow = np.exp(-(d / 2.2) ** 2) + np.exp(-(d / 22.0) ** 2) * 0.28
    return img, glow


def bloom(img, strength):
    """Soft glow from the brightest parts (the star, sunlit rims)."""
    lum = img.mean(axis=2)
    mask = np.clip((lum - 200.0) / 55.0, 0, 1)[..., None]
    bright = Image.fromarray(np.clip(img * mask, 0, 255).astype(np.uint8))
    small = bright.resize((W // 4, H // 4), Image.BILINEAR)
    g1 = np.asarray(small.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR), np.float32)
    g2 = np.asarray(small.filter(ImageFilter.GaussianBlur(14)).resize((W, H), Image.BILINEAR), np.float32)
    return img + strength * (g1 * 0.55 + g2 * 0.9) * np.array([1.0, 0.95, 0.88], np.float32)


def star_flare(img, s, raw):
    """Anamorphic streak + glow + ghosts at the star's screen position, scaled by how much of it is visible."""
    if not s or "star" not in s:
        return img
    sx, sy, front = s["star"]
    if not front or sx < -300 or sx > W + 300 or sy < -300 or sy > H + 300:
        return img
    # visibility: fraction of near-white pixels in a small box at the star (outside the frame: assume visible)
    ix, iy = int(round(sx)), int(round(sy))
    if 4 <= ix < W - 4 and 4 <= iy < H - 4:
        box = raw[iy - 4:iy + 5, ix - 4:ix + 5].mean(axis=2)
        vis = float(np.clip((box > 235).mean() * 1.6, 0, 1))
    else:
        vis = 0.6
    if vis <= 0.01:
        return img
    dx, dy = _xx - sx, _yy - sy
    streak = np.exp(-(dy / 2.2) ** 2) * np.exp(-np.abs(dx) / 330.0) * 0.9 + np.exp(-(dy / 9.0) ** 2) * np.exp(-np.abs(dx) / 140.0) * 0.35
    glow = np.exp(-(dx * dx + dy * dy) / (2 * 60.0 ** 2)) * 0.55 + np.exp(-(dx * dx + dy * dy) / (2 * 220.0 ** 2)) * 0.18
    out = img + (vis * 255.0) * (streak[..., None] * np.array([0.62, 0.80, 1.0], np.float32) +
                                 glow[..., None] * np.array([1.0, 0.86, 0.70], np.float32))
    # ghosts along the line from the star through the centre
    for k, r, a, col in ((0.55, 26, 0.10, (0.5, 0.8, 1.0)), (1.25, 48, 0.07, (1.0, 0.6, 0.4)), (1.7, 16, 0.12, (0.6, 1.0, 0.8))):
        gx, gy = sx + (_cx - sx) * k * 2 / 2, sy + (_cy - sy) * k * 2 / 2
        dd = np.sqrt((_xx - gx) ** 2 + (_yy - gy) ** 2)
        ring = np.clip(1 - np.abs(dd - r) / (r * 0.35), 0, 1) ** 2 * 0.6 + np.clip(1 - dd / r, 0, 1) * 0.4
        out = out + (vis * 255.0 * a) * ring[..., None] * np.array(col, np.float32)
    return out


def grade(img, f):
    img = img * _VIG
    x = np.clip(img / 255.0, 0, 1)
    x = x + 0.07 * (x - 0.5) * (1 - np.abs(2 * x - 1))
    img = x * 255.0
    rng = np.random.default_rng(f * 7919 + 13)
    return img + rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)


def letterbox_h(t):
    """Bars in the cinematic shots, opening up for the breakdowns."""
    full = 88
    if t < 16.0:
        return full
    if t < 17.4:
        return full * (1 - smooth((t - 16.0) / 1.4))
    if t < 66.5:
        return 0
    return full


def fade_black(t):
    k = smooth(t / 2.0)
    for cut in CUTS:
        if cut - 0.45 <= t < cut:
            k *= 1 - smooth((t - (cut - 0.45)) / 0.45)
        elif cut <= t < cut + 0.6:
            k *= smooth((t - cut) / 0.6)
    k *= 1 - smooth((t - (T.DURATION - 1.6)) / 1.6)
    return k


# ---------------------------------------------------------------- text
def draw_text(d, xy, s, ft, fill, alpha, anchor="la", spacing=0, features=None):
    if alpha <= 0.003:
        return
    col = (*fill, int(255 * clamp01(alpha)))
    feats = features if RAQM else None
    if spacing:
        x, y = xy
        total = sum(d.textlength(ch, font=ft) for ch in s) + spacing * (len(s) - 1)
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        for ch in s:
            d.text((x, y), ch, font=ft, fill=col, anchor="l" + anchor[1])
            x += d.textlength(ch, font=ft) + spacing
    else:
        d.text(xy, s, font=ft, fill=col, anchor=anchor, features=feats)


def shadowed(layer, radius=6, strength=150):
    a = layer.split()[-1].filter(ImageFilter.GaussianBlur(radius))
    a = a.point(lambda v: min(255, int(v * strength / 255 * 1.8)))
    sh = Image.new("RGBA", (W, H), (4, 5, 8, 0))
    sh.putalpha(a)
    return Image.alpha_composite(sh, layer)


def legend(d, a):
    """Ordered detail-level legend, bottom-left."""
    x0, y0, sw, hh = 64, H - 92, 30, 14
    draw_text(d, (x0, y0 - 26), "DETAIL LEVEL", FT_LBL, DIM, a * 0.9)
    for i, c in enumerate(LOD_STOPS):
        col = tuple(int(v * 255) for v in c) + (int(255 * a),)
        d.rectangle([x0 + i * sw, y0, x0 + (i + 1) * sw - 2, y0 + hh], fill=col)
    draw_text(d, (x0, y0 + hh + 8), "FULL DETAIL", FT_LBL, WHITE, a * 0.95)
    draw_text(d, (x0 + len(LOD_STOPS) * sw - 2, y0 + hh + 8), "COARSEST", FT_LBL, WHITE, a * 0.95, "ra")


# ---------------------------------------------------------------- captions
def section_text(key, f):
    """(number, title, line 1, line 2) of a section; line 2 of the Off-screen section follows the mode."""
    if key == "meshlets":
        return ("01", "MESHLETS & DETAIL", "Each patch is a meshlet: up to 128 triangles.",
                "Colour is its detail level: full near the camera, coarse far away.")
    if key == "streamed":
        return ("02", "THE HEAVIEST COPIES GET THEIR OWN CUT",
                "New: the copies whose whole-copy level costs most stream a meshlet cut (amber).",
                "Fine where the camera looks, coarse elsewhere. They are mostly the nearest copies.")
    if key == "before":
        return ("03", "SAME FRAME, OLD CODE VS NEW",
                "The old add-on (master 239e3a9) cuts this exact frame: stats only, not rendered.",
                "Same camera, same error budget, same off-screen rule (Coarsen).")
    if key == "offscreen":
        mode = stat(f, "offscreen", "COARSEN")
        l2 = {"COARSEN": "Coarsen: copies outside the view stay, at the coarsest detail.",
              "SHADOWS": "Sun Shadows: dropped, unless their sun shadow can reach the view.",
              "CULL": "Cull: every copy outside the view is dropped."}[mode]
        return ("04", "NEW: OFF-SCREEN MODES", "The cut is frozen at the camera; we fly up to look at it.", l2)
    if key == "payoff":
        return ("05", "LOOKING AWAY FROM A LOW SUN",
                "Cull drops the statues behind the camera, and their long shadows with them.",
                "Sun Shadows keeps just those, and still draws fewer triangles than Coarsen.")
    return None


MODE_NAMES = (("COARSEN", "COARSEN"), ("SHADOWS", "SUN SHADOWS"), ("CULL", "CULL"))


def mode_tabs(d, f, a, x1, y):
    """The three Off-screen modes with what each draws on this frame; the one on screen highlighted."""
    s = STATS[f]
    modes = s.get("modes") if s else None
    if not modes:
        return y
    cur = stat(f, "offscreen", "COARSEN")
    draw_text(d, (x1, y), "TRIANGLES DRAWN BY MODE", FT_LBL, DIM, a * 0.85, "ra")
    y += 24
    for key, name in MODE_NAMES:
        on = key == cur
        col = AMBER if on else DIM
        draw_text(d, (x1 - 150, y), name, FT_LBL, col, a * (1.0 if on else 0.75), "ra")
        draw_text(d, (x1, y - 4), fmt(modes[key]), FT_VAL, col if on else WHITE, a * (1.0 if on else 0.7), "ra",
                  features=["tnum"])
        y += 34
    return y


def stats_panel(d, f, t, a, shot):
    ST = scene_tris()
    drawn = stat(f, "drawnTriangles")
    rows = [("TRIANGLES IN SCENE", fmt(ST), WHITE),
            ("TRIANGLES DRAWN" + ("  ·  FROZEN" if stat(f, "frozen", False) else ""), fmt(drawn), ACCENT)]
    if shot == "flight" and f in BEFORE:
        hi = window(t, 40.2, 47.0, 0.8, 0.5)
        rows.append(("OLD CODE, SAME FRAME", fmt(BEFORE[f]), AMBER if hi > 0.5 else DIM))
    rows += [("MESHLETS STREAMED", fmt(stat(f, "streamedMeshlets")), WHITE),
             ("STREAMED COPIES", f"{fmt(stat(f, 'streamedCopies'))} / {fmt(copies())}", AMBER if 30 <= t < 40 else WHITE),
             ("ERROR BUDGET", f"{stat(f, 'errorPx', 1.5):.2f} px", WHITE)]
    x1, y = W - 64, 46
    for lbl, val, col in rows:
        draw_text(d, (x1, y), lbl, FT_LBL, DIM, a * 0.85, "ra")
        draw_text(d, (x1, y + 19), val, FT_VAL, col, a, "ra", features=["tnum"])
        y += 58
    if shot in ("frozen", "payoff"):
        mode_tabs(d, f, a, x1, y + 8)


def tags(d, f, a):
    """Small labels above the streamed copies on screen: nearness rank, distance, triangles."""
    s = shown(f)
    if not s:
        return
    vis = [g for g in s.get("tags", []) if g["xy"][2] and 40 < g["xy"][0] < W - 380 and 230 < g["xy"][1] < H - 120]
    vis.sort(key=lambda g: g["dist"])
    boxes = []
    for g in vis:
        x, y = g["xy"][0], g["xy"][1]
        box = (x - 4, y - 46, x + 300, y + 4)
        if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes):
            continue
        if len(boxes) >= 4:
            break
        boxes.append(box)
        col = (*AMBER, int(230 * a))
        d.line([(x, y), (x, y - 22)], fill=col, width=1)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
        what = "statue" if g["instancer"] == "Statues" else "floor patch"
        draw_text(d, (x + 6, y - 40), f"STREAMED  ·  #{g['rank']} NEAREST {what.upper()}", FT_TAG, AMBER, a)
        draw_text(d, (x + 6, y - 22), f"{g['dist']:.0f} m  ·  {fmt(g['tris'])} triangles", FT_TAGV, WHITE, a * 0.9)


def before_after(d, f, t, a):
    """Big old-vs-new comparison, centre-left."""
    if f not in BEFORE:
        return
    now, old = stat(f, "drawnTriangles"), BEFORE[f]
    x, y = 64, 232
    draw_text(d, (x, y), "OLD CODE", FT_LBL, DIM, a * 0.9)
    draw_text(d, (x, y + 18), fmt(old), FT_CMP, DIM, a, features=["tnum"])
    draw_text(d, (x, y + 98), "NOW", FT_LBL, ACCENT, a)
    draw_text(d, (x, y + 116), fmt(now), FT_CMP, WHITE, a, features=["tnum"])
    full = 520.0
    m = max(old, now, 1)
    d.rectangle([x, y + 84, x + full * old / m, y + 90], fill=(*DIM, int(200 * a)))
    d.rectangle([x, y + 182, x + full * now / m, y + 188], fill=(*ACCENT, int(230 * a)))
    if old:
        draw_text(d, (x, y + 204), f"{(1 - now / old) * 100:.0f}% fewer triangles on this frame", FT_DESC, GREEN, a)


def frustum(d, f, af):
    s = STATS[f]
    if af <= 0 or not s or not s.get("frustum"):
        return
    col = (*ACCENT, int(230 * af))
    for x1_, y1_, x2_, y2_ in s["frustum"]:
        d.line([(x1_, y1_), (x2_, y2_)], fill=col, width=2)
    ex, ey, ef = s["eye2d"]
    if ef:
        r = 6
        d.ellipse([ex - r, ey - r, ex + r, ey + r], outline=col, width=2)
        draw_text(d, (ex + 12, ey + 10) if ex < W - 150 else (ex - 12, ey + 10), "CAMERA", FT_TAG, ACCENT, af,
                  "la" if ex < W - 150 else "ra")


def wipe_labels(d, f, t, a):
    """Mode names on either side of a wipe between Off-screen modes."""
    s = STATS[f]
    if not s or not s.get("viewB") or s.get("wipe") is None:
        return
    va, vb = T.VIEWS[s["view"]], T.VIEWS[s["viewB"]]
    if va[1] == vb[1]:
        return
    names = dict(MODE_NAMES)
    front = (-0.2 + 1.4 * s["wipe"]) * W
    if front > 140:
        draw_text(d, (min(front - 24, W - 40), H - 200), names[vb[1]], FT_SECT, AMBER, a, "ra")
    if front < W - 140:
        draw_text(d, (max(front + 24, 40), H - 200), names[va[1]], FT_SECT, WHITE, a)


def overlays(t, f, bar):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    ST = scene_tris()
    s0 = STATS[f] or {}
    shot = s0.get("shot", "")

    a = window(t, 1.6, 8.4, 1.6, 1.2)
    if a > 0:
        ty = H - 88 - 160
        draw_text(d, (W / 2, ty), "SOLID SUZANNE", FT_TITLE, WHITE, a, "mm", spacing=12)
        a2 = window(t, 2.6, 8.4, 1.6, 1.2)
        draw_text(d, (W / 2, ty + 72), f"Virtual geometry in Blender  ·  {ST / 1e9:.2f} billion triangles  ·  one scene",
                  FT_SUB, DIM, a2, "mm")
        draw_text(d, (W / 2, ty + 108), f"Offline EEVEE render, 1280×720  ·  error budget {stat(f, 'errorPx', 1.5):.1f} px",
                  FT_END3, DIM, a2 * 0.85, "mm")

    a = window(t, 9.8, 15.6, 1.0, 1.0)
    if a > 0:
        y = H - bar - 92
        st_tris = (s0.get("assetTris") or {}).get("Statues", 0)
        draw_text(d, (64, y), "Every crack and carving here is geometry.", FT_CAP1, WHITE, a)
        draw_text(d, (64, y + 46), f"{st_tris / 1e6:.2f} million triangles per statue. No normal maps.", FT_CAP2, DIM, a * 0.95)

    for (s0t, s1t, key) in T.SECTIONS:
        a = window(t, s0t + 0.3, s1t + 0.2, 0.9, 0.7)
        if a <= 0:
            continue
        num, title, l1, l2 = section_text(key, f)
        x, y = 64, 58
        d.line([(x, y + 2), (x + 26 * smooth((t - s0t - 0.3) / 0.9), y + 2)], fill=(*ACCENT, int(255 * a)), width=2)
        draw_text(d, (x + 36, y - 9), num, FT_SECNUM, ACCENT, a)
        draw_text(d, (x, y + 18), title, FT_SECT, WHITE, a, spacing=2)
        draw_text(d, (x, y + 78), l1, FT_DESC, WHITE, window(t, s0t + 1.4, s1t + 0.2, 1.0, 0.7) * 0.95)
        draw_text(d, (x, y + 110), l2, FT_DESC, DIM, window(t, s0t + 2.6, s1t + 0.2, 1.0, 0.7) * 0.92)

    if 17.4 <= t < 47.0 or 47.0 <= t < 59.0:   # LOD views
        al = window(t, 18.0, 30.2, 1.0, 0.6) + window(t, 40.6, 58.8, 1.0, 0.8)
        if al > 0:
            legend(d, min(al, 1.0))

    a = window(t, 17.2, 66.4, 1.0, 0.8)
    if a > 0:
        stats_panel(d, f, t, a, shot)

    a = window(t, 31.4, 39.8, 1.0, 0.8)
    if a > 0:
        tags(d, f, a)

    a = window(t, 41.4, 46.8, 1.0, 0.6)
    if a > 0:
        before_after(d, f, t, a)

    frustum(d, f, window(t, 48.6, 58.8, 1.4, 0.8))
    wipe_labels(d, f, t, 1.0)
    if shot in ("frozen", "payoff") and not s0.get("viewB"):
        names = dict(MODE_NAMES)
        am = window(t, 52.0 if shot == "frozen" else 59.4, 58.8 if shot == "frozen" else 66.3, 0.6, 0.6)
        draw_text(d, (64, H - 200), names[stat(f, "offscreen", "COARSEN")], FT_SECT, AMBER, am * 0.95)

    a = window(t, 68.0, T.DURATION + 1.0, 1.4, 0.5)
    if a > 0:
        now, old = avg_now_before()
        draw_text(d, (W / 2, H / 2 - 64), fmt(ST), FT_BIG, WHITE, a, "mm", features=["tnum"])
        draw_text(d, (W / 2, H / 2 + 10), "triangles in the scene", FT_END2, DIM, a, "mm")
        a2 = window(t, 68.9, T.DURATION + 1.0, 1.2, 0.5)
        if old:
            draw_text(d, (W / 2, H / 2 + 58),
                      f"{now / 1e6:.2f} million drawn per frame in the flight, against {old / 1e6:.2f} million with the old code",
                      FT_END2, ACCENT, a2, "mm")
        a3 = window(t, 69.7, T.DURATION + 1.0, 1.0, 0.5)
        pm = payoff_modes()
        if pm:
            draw_text(d, (W / 2, H / 2 + 100),
                      f"Looking away from the sun: Sun Shadows {mtris(pm['SHADOWS'])}, Coarsen {mtris(pm['COARSEN'])}, Cull {mtris(pm['CULL'])}",
                      FT_END3, WHITE, a3 * 0.9, "mm")
        a4 = window(t, 70.4, T.DURATION + 1.0, 1.0, 0.5)
        draw_text(d, (W / 2, H - bar - 30),
                  f"Blender  ·  EEVEE offline render  ·  error budget {stat(f, 'errorPx', 1.5):.1f} px  ·  Solid Suzanne virtual geometry",
                  FT_END3, WHITE, a4 * 0.8, "mm")
    return shadowed(lay)


_scr = None


def apply_scrims(img, t, f):
    global _scr
    if _scr is None:
        l = np.clip(1 - _xx / 900.0, 0, 1) ** 1.4 * np.clip(1 - _yy / 260.0, 0, 1) ** 1.2
        r = np.clip(1 - (W - _xx) / 470.0, 0, 1) ** 1.4 * np.clip(1 - _yy / 620.0, 0, 1) ** 1.1
        b = np.clip(1 - _xx / 700.0, 0, 1) ** 1.2 * np.clip((_yy - (H - 190)) / 190.0, 0, 1) ** 1.2
        c = np.clip(1 - _xx / 760.0, 0, 1) ** 1.3 * np.exp(-((_yy - 360) / 170.0) ** 2)
        _scr = (l.astype(np.float32), r.astype(np.float32), b.astype(np.float32), c.astype(np.float32))
    l, r, b, c = _scr
    al = 0.62 * max(window(t, s0 + 0.2, s1 + 0.3, 0.9, 0.8) for (s0, s1, _k) in T.SECTIONS)
    ar = 0.62 * window(t, 17.2, 66.4, 1.0, 0.8)
    ab = 0.55 * max(window(t, 18.0, 30.2, 1.0, 0.6), window(t, 40.6, 66.3, 1.0, 0.8))
    ac = 0.6 * window(t, 41.4, 46.8, 1.0, 0.6)
    k = (1 - al * l) * (1 - ar * r) * (1 - ab * b) * (1 - ac * c)
    return img * k[..., None]


def end_dim(t):
    return 1 - 0.45 * smooth((t - 67.8) / 1.6)


# ---------------------------------------------------------------- frame
_last = {}


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def frame(f):
    t = (f + 0.5) / FPS
    pa = f"{FR}/{f:04d}_a.jpg"
    if os.path.exists(pa):
        a = load(pa)
        _last["a"] = a
    else:
        a = _last.get("a", np.zeros((H, W, 3), np.float32))
    raw = a
    s = STATS[f]
    img, glow = a, None
    if s and s.get("viewB") and os.path.exists(f"{FR}/{f:04d}_b.jpg"):
        img, glow = wipe(a, load(f"{FR}/{f:04d}_b.jpg"), s["wipe"])
    beauty = (not s) or (T.VIEWS[s.get("view", "beauty")][0] == 0 and not s.get("viewB"))
    img = bloom(img, 0.85 if beauty else 0.35)
    if beauty:
        img = star_flare(img, s, raw)
    img = grade(img, f)
    if glow is not None:
        img = img + glow[..., None] * np.array([90, 190, 255], np.float32) * 0.95
    img = apply_scrims(img, t, f)
    img = img * end_dim(t)
    bar = int(round(letterbox_h(t)))
    if bar > 0:
        img[:bar] = 0
        img[H - bar:] = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    out = Image.alpha_composite(Image.fromarray(img, "RGB").convert("RGBA"), overlays(t, f, bar)).convert("RGB")
    k = fade_black(t)
    if k < 0.999:
        out = Image.fromarray((np.asarray(out, np.float32) * k).astype(np.uint8))
    return out


def encode(name, f0, f1):
    tmp = f"{OUTD}/{name}.tmp.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(frame(f).tobytes())
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, f"{OUTD}/{name}.mp4")
        print("wrote", name, flush=True)
    else:
        print("ffmpeg failed for", name)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--still":
        for fs in sys.argv[2:]:
            frame(int(fs)).save(os.path.join(WORK, f"still_{int(fs):04d}.png"))
        sys.exit(0)
    names = sys.argv[1:] or [s[0] for s in SEGMENTS]
    for name, f0, f1 in SEGMENTS:
        if name in names:
            encode(name, f0, f1)
