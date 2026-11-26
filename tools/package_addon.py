"""Package addons/vgeo as a Blender extension (.zip) with Blender's own build and validate commands.

    python tools/package_addon.py [--blender PATH] [--out DIR]

Stages addons/vgeo with its built native library (bin/), the web viewer files Export for Web
needs, LICENSE and THIRD_PARTY_NOTICES.md, sets the manifest's platforms from the libraries
present, then runs `blender --command extension build` and `... validate`. The result installs
with Preferences > Get Extensions > Install from Disk, or uploads to extensions.blender.org.

The native library is built per platform (see docs/NATIVE_STREAMING.md); a package carries the
libraries found in addons/vgeo/bin: vgeo_stream.dll -> windows-x64, vgeo_stream.so -> linux-x64,
vgeo_stream.dylib -> macos-arm64 (set VGEO_MACOS_PLATFORM=macos-x64 for an Intel build).
"""

import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "addons", "vgeo")
SKIP_DIRS = {"__pycache__"}
SKIP_EXT = (".pyc", ".pdb", ".ilk", ".exp", ".lib")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def find_blender():
    for p in (arg("--blender"), os.environ.get("BLENDER"),
              r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
              "/Applications/Blender.app/Contents/MacOS/Blender", shutil.which("blender")):
        if p and os.path.exists(p):
            return p
    sys.exit("Blender not found: pass --blender PATH or set BLENDER")


def platforms(bin_dir):
    out = []
    for f in sorted(os.listdir(bin_dir)) if os.path.isdir(bin_dir) else []:
        if f == "vgeo_stream.dll":
            out.append("windows-x64")
        elif f == "vgeo_stream.so":
            out.append("linux-x64")
        elif f == "vgeo_stream.dylib":
            out.append(os.environ.get("VGEO_MACOS_PLATFORM", "macos-arm64"))
    return out


def main():
    out_dir = os.path.abspath(arg("--out", os.path.join(REPO, "dist")))
    plats = platforms(os.path.join(SRC, "bin"))
    if not plats:
        sys.exit("addons/vgeo/bin has no vgeo_stream library: build it first (see README)")
    stage = os.path.join(REPO, "build-package", "vgeo")
    shutil.rmtree(os.path.dirname(stage), ignore_errors=True)
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, SRC)
        os.makedirs(os.path.join(stage, rel), exist_ok=True)
        for f in files:
            if not f.endswith(SKIP_EXT):
                shutil.copy2(os.path.join(root, f), os.path.join(stage, rel, f))
    os.makedirs(os.path.join(stage, "web"), exist_ok=True)
    for f in ("vgeo-viewer.js", "meshopt_decoder.mjs"):        # Export for Web copies these
        shutil.copy2(os.path.join(REPO, "web", f), os.path.join(stage, "web", f))
    for f in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(os.path.join(REPO, f), os.path.join(stage, f))
    manifest = os.path.join(stage, "blender_manifest.toml")
    text = open(manifest, encoding="utf-8").read()
    text = re.sub(r"(?m)^platforms = \[.*\]$", "platforms = [" + ", ".join(f'"{p}"' for p in plats) + "]", text)
    open(manifest, "w", encoding="utf-8", newline="\n").write(text)

    blender = find_blender()
    os.makedirs(out_dir, exist_ok=True)
    cmd = [blender, "--factory-startup", "--command", "extension", "build", "--source-dir", stage,
           "--output-dir", out_dir]
    if len(plats) > 1:
        cmd.append("--split-platforms")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".zip"):
            subprocess.run([blender, "--factory-startup", "--command", "extension", "validate",
                            os.path.join(out_dir, f)], check=True)
            print("built and validated", os.path.join(out_dir, f))


if __name__ == "__main__":
    main()
