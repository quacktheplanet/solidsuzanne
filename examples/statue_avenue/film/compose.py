"""Composites the asteroid-belt frames: wipes, bloom, star flare, letterbox, titles, LOD legend, live HUD,
frozen-frustum overlay, end card. Each segment is piped to ffmpeg and renamed only on success.

Usage: python compose.py [segment ...]       (default: all)
       python compose.py --still <frame> ...  (composited PNGs for review)
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORK = os.path.abspath(os.environ.get("WORK", "work"))
FR = os.path.join(WORK, "frames")
OUTD = os.path.join(WORK, "segments")
os.makedirs(OUTD, exist_ok=True)
W, H, FPS = 1280, 720, 24
FRAMES = 66 * FPS
SEGMENTS = [("s1", 0, 12 * FPS), ("s2", 12 * FPS, 22 * FPS), ("s3", 22 * FPS, 57 * FPS), ("s4", 57 * FPS, FRAMES)]
CUTS = (12.0, 22.0, 57.0)
SECTIONS = [  # (start, end, number, title, line1, line2)
    (23.4, 34.0, "01", "MESHLETS & DETAIL",
     "Each patch is a meshlet: up to 128 triangles.",
     "Colour is its detail level. Near statues fine, far ones coarse."),
    (34.0, 45.6, "02", "WATCH IT SWITCH",
     "Error budget raised 8x so the switches are visible.",
     "Statues refine as the camera nears, then coarsen as it passes."),
    (45.6, 56.8, "03", "FROZEN CUT",
     "The cut is frozen at the camera. We fly up to inspect it.",
     "In view and near: full detail. Far or off-screen: coarsest."),
]
DIAL = (37.0, 44.8)
FROZEN = (45.5, 57.0)
LOD_STOPS = [
    [0.97, 0.99, 1.00], [0.45, 0.93, 1.00], [0.20, 0.75, 1.00], [0.12, 0.50, 1.00], [0.25, 0.32, 1.00],
    [0.48, 0.26, 1.00], [0.70, 0.22, 0.95], [0.90, 0.20, 0.80], [1.00, 0.22, 0.55], [1.00, 0.26, 0.30],
    [1.00, 0.40, 0.15], [1.00, 0.56, 0.10], [1.00, 0.72, 0.12], [1.00, 0.86, 0.30],
]

F = os.environ.get("FONT_DIR", "/usr/share/fonts/opentype/inter/")   # Inter / Inter Display
def font(name, size):
    try:
        return ImageFont.truetype(F + name, size)
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
FT_SECT = font("InterDisplay-Medium.otf", 42)
FT_DESC = font("Inter-Regular.otf", 25)
FT_LBL = font("Inter-SemiBold.otf", 15)
FT_VAL = font("Inter-Medium.otf", 27)
FT_BIG = font("InterDisplay-Light.otf", 96)
FT_END2 = font("Inter-Regular.otf", 28)
FT_END3 = font("Inter-Regular.otf", 19)
FT_TAG = font("Inter-SemiBold.otf", 17)

ACCENT = (120, 214, 255)
WHITE = (245, 242, 236)
DIM = (200, 198, 196)


def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def window(t, a, b, fin=0.8, fout=0.8):
    return smooth((t - a) / fin) * (1 - smooth((t - (b - fout)) / fout))


def fmt(n):
    return f"{int(round(n)):,}"


# ---------------------------------------------------------------- stats
STATS = []
for f in range(FRAMES):
    p = f"{FR}/{f:04d}.json"
    STATS.append(json.load(open(p)) if os.path.exists(p) else None)


def stat(f, key, default=0):
    s = STATS[f]
    return s.get(key, default) if s else default


def scene_tris():
    for s in STATS:
        if s and s.get("sceneTris"):
            return s["sceneTris"]
    return 0


def instances():
    for s in STATS:
        if s and s.get("statues"):
            return s["statues"] + s["patches"]
    return 0


def avg_drawn():
    vals = [s["drawnTriangles"] for s in STATS if s and s.get("view") == "beauty" and not s.get("viewB")
            and not s.get("frozen") and abs(s.get("errorPx", 1.5) - 1.5) < 1e-3]
    return sum(vals) / max(1, len(vals))


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
    full = 88
    if t < 22.0:
        return full
    if t < 23.6:
        return full * (1 - smooth((t - 22.0) / 1.4))
    if t < 57.0:
        return 0
    return full


def fade_black(t):
    k = smooth(t / 2.0)
    for cut in CUTS:
        if cut - 0.45 <= t < cut:
            k *= 1 - smooth((t - (cut - 0.45)) / 0.45)
        elif cut <= t < cut + 0.6:
            k *= smooth((t - cut) / 0.6)
    k *= 1 - smooth((t - 64.4) / 1.6)
    return k


# ---------------------------------------------------------------- text
def draw_text(d, xy, s, ft, fill, alpha, anchor="la", spacing=0, features=None):
    if alpha <= 0.003:
        return
    col = (*fill, int(255 * clamp01(alpha)))
    if spacing:
        x, y = xy
        total = sum(d.textlength(ch, font=ft) for ch in s) + spacing * (len(s) - 1)
        if anchor[0] == "m":
            x -= total / 2
        for ch in s:
            d.text((x, y), ch, font=ft, fill=col, anchor="l" + anchor[1])
            x += d.textlength(ch, font=ft) + spacing
    else:
        d.text(xy, s, font=ft, fill=col, anchor=anchor, features=features)


def shadowed(layer, radius=6, strength=150):
    a = layer.split()[-1].filter(ImageFilter.GaussianBlur(radius))
    a = a.point(lambda v: min(255, int(v * strength / 255 * 1.8)))
    sh = Image.new("RGBA", (W, H), (4, 5, 8, 0))
    sh.putalpha(a)
    return Image.alpha_composite(sh, layer)


def legend(d, t, a):
    """Ordered detail-level legend, bottom-left."""
    x0, y0, sw, hh = 64, H - 92, 30, 14
    draw_text(d, (x0, y0 - 26), "DETAIL LEVEL", FT_LBL, DIM, a * 0.9)
    for i, c in enumerate(LOD_STOPS):
        col = tuple(int(v * 255) for v in c) + (int(255 * a),)
        d.rectangle([x0 + i * sw, y0, x0 + (i + 1) * sw - 2, y0 + hh], fill=col)
    draw_text(d, (x0, y0 + hh + 8), "FULL DETAIL", FT_LBL, WHITE, a * 0.95)
    draw_text(d, (x0 + len(LOD_STOPS) * sw - 2, y0 + hh + 8), "COARSEST", FT_LBL, WHITE, a * 0.95, "ra")


def overlays(t, f, bar):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    ST, INST = scene_tris(), instances()

    a = window(t, 2.0, 9.8, 1.6, 1.4)
    if a > 0:
        ty = H - 88 - 160
        draw_text(d, (W / 2, ty), "SOLID SUZANNE", FT_TITLE, WHITE, a, "mm", spacing=12)
        a2 = window(t, 3.0, 9.8, 1.6, 1.4)
        draw_text(d, (W / 2, ty + 72), f"Virtual geometry in Blender  ·  {ST / 1e9:.2f} billion triangles  ·  one scene",
                  FT_SUB, DIM, a2, "mm")

    a = window(t, 13.2, 21.4, 1.0, 1.0)
    if a > 0:
        y = H - bar - 92
        draw_text(d, (64, y), "Every crack and carving here is geometry.", FT_CAP1, WHITE, a)
        draw_text(d, (64, y + 46), f"{stat(f, 'monoTris', 1428316) / 1e6:.2f} million triangles per statue. No normal maps.", FT_CAP2, DIM, a * 0.95)

    for (s0, s1, num, title, l1, l2) in SECTIONS:
        a = window(t, s0 + 0.3, s1 + 0.2, 0.9, 0.7)
        if a <= 0:
            continue
        x, y = 64, 58
        d.line([(x, y + 2), (x + 26 * smooth((t - s0 - 0.3) / 0.9), y + 2)], fill=(*ACCENT, int(255 * a)), width=2)
        draw_text(d, (x + 36, y - 9), num, FT_SECNUM, ACCENT, a)
        draw_text(d, (x, y + 18), title, FT_SECT, WHITE, a, spacing=3)
        ad = window(t, s0 + 1.6, s1 + 0.2, 1.0, 0.7)
        draw_text(d, (x, y + 80), l1, FT_DESC, WHITE, ad * 0.95)
        ad2 = window(t, s0 + 3.0, s1 + 0.2, 1.0, 0.7)
        draw_text(d, (x, y + 114), l2, FT_DESC, DIM, ad2 * 0.92)

    # legend while any breakdown view is on screen
    al = window(t, 24.4, 56.8, 1.0, 0.8)
    if al > 0:
        legend(d, t, al)

    # live stats
    a = window(t, 23.2, 56.8, 1.0, 0.8)
    if a > 0:
        drawn, meshlets, vis = stat(f, "drawnTriangles"), stat(f, "streamedMeshlets"), stat(f, "streamedCopies")
        err = stat(f, "errorPx", 1.25)
        dial = DIAL[0] - 0.3 <= t <= DIAL[1] + 0.6
        frozen = FROZEN[0] + 1.2 <= t
        rows = [
            ("TRIANGLES IN SCENE", fmt(ST), WHITE),
            ("TRIANGLES DRAWN" + ("  ·  FROZEN" if frozen else ""), fmt(drawn), ACCENT if (dial or frozen) else WHITE),
            ("MESHLETS STREAMED", fmt(meshlets), WHITE),
            ("COPIES AT FULL DETAIL", f"{fmt(vis)} / {fmt(INST)}", WHITE),
            ("ERROR BUDGET", f"{err:.2f} px", ACCENT if dial else WHITE),
        ]
        x1, y = W - 64, 50
        for lbl, val, col in rows:
            draw_text(d, (x1, y), lbl, FT_LBL, DIM, a * 0.85, "ra", features=["tnum"])
            draw_text(d, (x1, y + 19), val, FT_VAL, col, a, "ra", features=["tnum"])
            y += 60
        draw_text(d, (x1, y + 2), f"1 IN {fmt(ST / max(1, drawn))} DRAWN", FT_LBL, ACCENT, a * 0.95, "ra", features=["tnum"])

    # frozen frustum
    s = STATS[f]
    af = window(t, 47.2, 56.8, 1.4, 0.8)
    if af > 0 and s and s.get("frustum"):
        col = (*ACCENT, int(230 * af))
        for x1_, y1_, x2_, y2_ in s["frustum"]:
            d.line([(x1_, y1_), (x2_, y2_)], fill=col, width=2)
        ex, ey, ef = s["eye2d"]
        if ef:
            r = 6
            d.ellipse([ex - r, ey - r, ex + r, ey + r], outline=col, width=2)
            if ex > W - 150:
                draw_text(d, (ex - 12, ey + 10), "CAMERA", FT_TAG, ACCENT, af, "ra")
            else:
                draw_text(d, (ex + 12, ey + 10), "CAMERA", FT_TAG, ACCENT, af)

    a = window(t, 60.4, 67.0, 1.4, 0.5)
    if a > 0:
        draw_text(d, (W / 2, H / 2 - 46), fmt(ST), FT_BIG, WHITE, a, "mm", features=["tnum"])
        draw_text(d, (W / 2, H / 2 + 28), "triangles in the scene", FT_END2, DIM, a, "mm")
        a2 = window(t, 61.4, 67.0, 1.2, 0.5)
        draw_text(d, (W / 2, H / 2 + 74), f"about {avg_drawn() / 1e6:.1f} million drawn per frame", FT_END2, ACCENT, a2, "mm")
        a3 = window(t, 62.2, 67.0, 1.0, 0.5)
        draw_text(d, (W / 2, H - bar - 30), "Blender  ·  EEVEE  ·  Solid Suzanne virtual geometry", FT_END3, WHITE, a3 * 0.8, "mm")
    return shadowed(lay)


_scr = None


def apply_scrims(img, t):
    global _scr
    if _scr is None:
        l = np.clip(1 - _xx / 820.0, 0, 1) ** 1.4 * np.clip(1 - _yy / 250.0, 0, 1) ** 1.2
        r = np.clip(1 - (W - _xx) / 470.0, 0, 1) ** 1.4 * np.clip(1 - _yy / 420.0, 0, 1) ** 1.1
        b = np.clip(1 - _xx / 640.0, 0, 1) ** 1.2 * np.clip((_yy - (H - 170)) / 170.0, 0, 1) ** 1.2
        _scr = (l.astype(np.float32), r.astype(np.float32), b.astype(np.float32))
    l, r, b = _scr
    al = 0.62 * max(window(t, s0 + 0.2, s1 + 0.3, 0.9, 0.8) for (s0, s1, *_rest) in SECTIONS)
    ar = 0.62 * window(t, 23.2, 56.8, 1.0, 0.8)
    ab = 0.55 * window(t, 24.4, 56.8, 1.0, 0.8)
    k = (1 - al * l) * (1 - ar * r) * (1 - ab * b)
    return img * k[..., None]


def end_dim(t):
    return 1 - 0.45 * smooth((t - 60.2) / 1.6)


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
    beauty = (not s) or (s.get("view") == "beauty" and not s.get("viewB"))
    img = bloom(img, 0.85 if beauty else 0.35)
    if beauty:
        img = star_flare(img, s, raw)
    img = grade(img, f)
    if glow is not None:
        img = img + glow[..., None] * np.array([90, 190, 255], np.float32) * 0.95
    img = apply_scrims(img, t)
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
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
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
