#!/usr/bin/env python3
"""Render the app icons from the HTML sources in this folder.

    python tools/make_icons.py

Needs a Chromium-based browser (Edge, Chrome, Brave, Chromium). It renders
headlessly, so nothing opens on screen. Set BROWSER to override the lookup:

    BROWSER=/usr/bin/chromium python tools/make_icons.py

Outputs:
    app/web/icon-192.png, icon-512.png        PWA icons
    android/.../mipmap-*/ic_launcher.png      legacy launcher icon
    android/.../mipmap-*/ic_launcher_round.png
    android/.../mipmap-*/ic_launcher_foreground.png   adaptive foreground

The Android half is skipped when the Capacitor project has not been added yet.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
WEB = ROOT / "app" / "web"
RES = ROOT / "android" / "app" / "src" / "main" / "res"

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
]

# mipmap bucket -> launcher size, and -> adaptive foreground size (108 dp)
DENSITIES = {
    "mdpi": (48, 108),
    "hdpi": (72, 162),
    "xhdpi": (96, 216),
    "xxhdpi": (144, 324),
    "xxxhdpi": (192, 432),
}

# drawable bucket -> splash size in portrait (landscape is the same, rotated)
SPLASHES = {
    "mdpi": (320, 480),
    "hdpi": (480, 800),
    "xhdpi": (720, 1280),
    "xxhdpi": (960, 1600),
    "xxxhdpi": (1280, 1920),
}


def find_browser() -> str:
    env = os.environ.get("BROWSER")
    if env and Path(env).exists():
        return env
    for path in CANDIDATES:
        if Path(path).exists():
            return path
    found = shutil.which("chromium") or shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    sys.exit("No Chromium-based browser found. Set BROWSER=/path/to/chrome.")


def render(browser: str, source: Path, width: int, height: int, out: Path,
           transparent: bool) -> None:
    """Render `source` at exactly `width` x `height` pixels.

    `--window-size` does not reliably size the viewport in newer headless builds
    (the page still lays out at ~800 px and the screenshot crops it), so the
    document is pinned to the target box instead, and the window is only asked to
    be that big.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    html = source.read_text(encoding="utf-8").replace(
        "</head>",
        f"<style>html,body{{width:{width}px;height:{height}px;}}</style></head>",
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html, encoding="utf-8")
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={out}",
            "--virtual-time-budget=1500",
            f"--user-data-dir={tmp}/profile",
        ]
        if transparent:
            cmd.append("--default-background-color=00000000")
        cmd.append(page.as_uri())
        result = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists() or out.stat().st_size == 0:
        sys.exit(f"failed to render {out.name}:\n{result.stderr[-600:]}")
    print(f"  {out.relative_to(ROOT)}  {width}x{height}")


def main() -> None:
    browser = find_browser()
    print(f"browser: {browser}")

    square = TOOLS / "icon-source.html"
    round_ = TOOLS / "icon-round.html"
    fg = TOOLS / "icon-foreground.html"
    splash = TOOLS / "splash-source.html"

    print("PWA icons")
    render(browser, square, 192, 192, WEB / "icon-192.png", transparent=False)
    render(browser, square, 512, 512, WEB / "icon-512.png", transparent=False)

    if not RES.exists():
        print("android/ not present, skipping launcher icons")
        return

    print("Android launcher icons")
    for bucket, (launcher, foreground) in DENSITIES.items():
        base = RES / f"mipmap-{bucket}"
        render(browser, square, launcher, launcher, base / "ic_launcher.png", transparent=False)
        render(browser, round_, launcher, launcher, base / "ic_launcher_round.png", transparent=True)
        render(browser, fg, foreground, foreground,
               base / "ic_launcher_foreground.png", transparent=True)

    print("Android splash screens")
    for bucket, (width, height) in SPLASHES.items():
        render(browser, splash, width, height,
               RES / f"drawable-port-{bucket}" / "splash.png", transparent=False)
        render(browser, splash, height, width,
               RES / f"drawable-land-{bucket}" / "splash.png", transparent=False)


if __name__ == "__main__":
    main()
