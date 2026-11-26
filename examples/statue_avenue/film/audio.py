"""Synthesizes the soundtrack (60 s, 48 kHz stereo WAV): ambient pad, wipe whooshes, dial ticks, hits."""
import numpy as np
import os
import wave

SR = 48000
DUR = 66.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(3)

WIPES = [(23.4, 25.4), (34.0, 36.0), (45.6, 47.4)]
CUTS = [12.0, 22.0, 57.0]
DIAL = [(37.0, 39.2, +1), (42.6, 44.8, -1)]


def hz(note):
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    n, o = note[:-1], int(note[-1])
    return 440.0 * 2 ** ((names[n] + 12 * (o + 1) - 69) / 12)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def env_window(a, b, fin, fout):
    return smooth((t - a) / fin) * (1 - smooth((t - (b - fout)) / fout))


def one_pole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    acc = 0.0
    # vectorised via lfilter-like recursion in chunks would be faster; this is fine for a few calls
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def fft_lowpass(x, fc, order=2):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    h = 1 / np.sqrt(1 + (f / fc) ** (2 * order))
    return np.fft.irfft(X * (h[:, None] if X.ndim == 2 else h), n=x.shape[0], axis=0)


def fft_bandpass(x, lo, hi, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    h = 1 / np.sqrt(1 + (f / hi) ** (2 * order)) * (1 - 1 / np.sqrt(1 + (f / lo) ** (2 * order)))
    return np.fft.irfft(X * h, n=len(x))


# ---------------------------------------------------------------- pad
CHORDS = [
    (0.0, 12.0, ["C2", "G2", "D3", "E3", "B3"]),
    (12.0, 22.0, ["A1", "E2", "C3", "G3", "B3"]),
    (22.0, 34.0, ["F2", "C3", "A3", "E4", "G4"]),
    (34.0, 45.6, ["G2", "D3", "B3", "F#4", "A4"]),
    (45.6, 57.0, ["E2", "B2", "G3", "D4", "F#4"]),
    (57.0, 66.0, ["C2", "G2", "D3", "E3", "G4"]),
]
pad = np.zeros((N, 2))
for ci, (a, b, notes) in enumerate(CHORDS):
    fin = 3.0 if ci else 4.0
    e = env_window(a - 1.2, b + 1.2, fin, 2.4)
    idx = np.nonzero(e > 1e-4)[0]
    tt = t[idx]
    for ni, nn in enumerate(notes):
        f0 = hz(nn)
        voice = np.zeros((len(idx), 2))
        for vi, cents in enumerate((-7, 0, 6)):
            f = f0 * 2 ** (cents / 1200)
            ph = rng.uniform(0, 2 * np.pi)
            s = np.zeros(len(idx))
            for k in range(1, 9):
                if f * k > 6000:
                    break
                s += np.sin(2 * np.pi * f * k * tt + ph * k) / (k ** 1.6)
            pan = 0.5 + 0.35 * (vi - 1) * (1 if ni % 2 else -1)
            voice[:, 0] += s * (1 - pan)
            voice[:, 1] += s * pan
        # slow breathing per note
        trem = 0.75 + 0.25 * np.sin(2 * np.pi * (0.05 + 0.017 * ni) * tt + ni)
        w = 1.0 if f0 < 200 else 0.7
        pad[idx] += voice * (e[idx] * trem * w)[:, None]
pad = fft_lowpass(pad, 1400, 2)
pad *= 0.11

# sub drone on A1, swelling under the title and the end card
sub = np.sin(2 * np.pi * hz("C2") / 2 * t) * (0.5 + 0.5 * env_window(1.5, 11, 2.5, 3) + 0.6 * env_window(59.5, 66, 2.0, 2.0))
sub *= 0.05 * smooth(t / 3.0)
pad += np.stack([sub, sub], 1)

mix = pad.copy()


# ---------------------------------------------------------------- one-shots
def add(sig, start, gain=1.0, pan=0.5):
    i0 = int(start * SR)
    n = min(len(sig), N - i0)
    if n <= 0:
        return
    if sig.ndim == 1:
        mix[i0:i0 + n, 0] += sig[:n] * gain * (1 - pan) * 2 ** 0.5
        mix[i0:i0 + n, 1] += sig[:n] * gain * pan * 2 ** 0.5
    else:
        mix[i0:i0 + n] += sig[:n] * gain


def boom(dur=3.5, f0=62, f1=34):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-tt * 3.0)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = (1 - np.exp(-tt * 60)) * np.exp(-tt * 1.4)
    body = np.sin(ph) * env
    noise = fft_lowpass(rng.normal(0, 1, n), 300, 2) * np.exp(-tt * 6) * 0.25
    return body + noise


def whoosh(dur, pan_lr=True):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    u = tt / dur
    noise = rng.normal(0, 1, n)
    # time-varying band: render a few bands and crossfade between them
    bands = [(200, 700), (500, 1600), (1200, 3500), (2500, 7000)]
    out = np.zeros(n)
    pos = u * (len(bands) - 1)
    for i, (lo, hi) in enumerate(bands):
        w = np.clip(1 - np.abs(pos - i), 0, 1)
        out += fft_bandpass(noise, lo, hi) * w
    env = np.sin(np.pi * np.clip(u, 0, 1)) ** 1.5
    out *= env
    st = np.zeros((n, 2))
    p = 0.15 + 0.7 * smooth(u) if pan_lr else np.full(n, 0.5)
    st[:, 0] = out * (1 - p)
    st[:, 1] = out * p
    return st


def shimmer(freqs=(1760, 2637, 3520), dur=2.2):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * f * tt + i) / (i + 1) for i, f in enumerate(freqs))
    return s * (1 - np.exp(-tt * 200)) * np.exp(-tt * 2.2)


def tick(f=2400):
    n = int(0.03 * SR)
    tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * tt) * 0.6 + rng.normal(0, 0.4, n)) * np.exp(-tt * 260)


# title + end hits
add(boom(), 2.1, 0.55)
add(boom(4.5, 55, 30), 60.4, 0.6)
# the star clearing the giant's limb: rising shimmer swell
n = int(4.5 * SR); tt = np.arange(n) / SR
swell = sum(np.sin(2 * np.pi * f * tt + i) / (i + 1) for i, f in enumerate((880, 1318.5, 1760, 2637)))
swell *= smooth(tt / 3.2) * (1 - smooth((tt - 3.6) / 0.9)) * 0.5
add(swell, 0.8, 0.06, 0.4)
# cuts
for c in CUTS:
    add(boom(1.4, 90, 45), c - 0.02, 0.26)
# wipes
for a, b in WIPES:
    add(whoosh(b - a + 0.4), a - 0.2, 0.32)
    add(shimmer(), b - 0.25, 0.05, 0.6)
# dial ticks: rate follows the error ramp
for a, b, direction in DIAL:
    tt0 = a
    while tt0 < b:
        u = (tt0 - a) / (b - a)
        k = u if direction > 0 else 1 - u
        add(tick(1600 + 1800 * k), tt0, 0.10, 0.5 + 0.25 * (k - 0.5))
        tt0 += 1 / (9 + 16 * np.sin(np.pi * u))
# freeze: a glassy 'lock' and a held high tone while the camera pulls back
add(shimmer((2349, 3136, 4699), 3.0), 45.55, 0.09, 0.5)
n = int(11.0 * SR); tt = np.arange(n) / SR
hold = (np.sin(2 * np.pi * 1174.7 * tt) + 0.5 * np.sin(2 * np.pi * 1760 * tt + 1.0)) * smooth(tt / 1.5) * (1 - smooth((tt - 9.6) / 1.2))
add(hold * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * tt)), 45.8, 0.025, 0.5)

# ---------------------------------------------------------------- reverb + master
ir_n = int(3.2 * SR)
ir_t = np.arange(ir_n) / SR
ir = rng.normal(0, 1, (ir_n, 2)) * np.exp(-ir_t * 2.1)[:, None]
ir = fft_lowpass(ir, 5000, 1)
ir[: int(0.012 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
L = N + ir_n
wet = np.stack([np.fft.irfft(np.fft.rfft(mix[:, c], L) * np.fft.rfft(ir[:, c], L), L)[:N] for c in (0, 1)], 1)
out = mix * 0.82 + wet * 0.30

fade = smooth(t / 1.4) * (1 - smooth((t - 64.4) / 1.6))
out *= fade[:, None]
out = np.tanh(out * 1.2) / np.tanh(1.2)
peak = np.abs(out).max()
out = out / peak * 10 ** (-1.5 / 20)
rms = np.sqrt((out ** 2).mean())
print(f"peak {20*np.log10(np.abs(out).max()):.1f} dBFS, rms {20*np.log10(rms):.1f} dBFS")

pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
with wave.open(os.path.join(os.path.abspath(os.environ.get("WORK", "work")), "score.wav"), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote score.wav")
