#!/usr/bin/env python3
"""
lens_cli — friendly wrapper around Lens font recognition.

Accepts a local image path (any format: png/jpg/avif/heic/webp/tiff/...) OR an
http(s) URL. Handles the annoying parts automatically:
  - downloads remote images regardless of content-type
  - converts anything PIL can open into a clean PNG
  - serves it over a throwaway localhost server (the model only accepts http URLs)
  - prints a readable table of font matches + their .ttf download links

Usage:
  lens ~/Desktop/sample.png
  lens https://example.com/poster.avif
  lens ~/Desktop/sample.png --top-k 10
  lens ~/Desktop/sample.png --json      # raw JSON instead of the table
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402
from lens_inference import run_inference_from_url  # noqa: E402

MODEL_PATH = REPO_ROOT / "model" / "font_classifier.pt"


def load_image_bytes(source: str) -> bytes:
    """Return raw bytes for a local path or an http(s) URL."""
    if source.startswith(("http://", "https://")):
        req = Request(source, headers={"User-Agent": "lens-cli"})
        with urlopen(req, timeout=30) as resp:
            return resp.read()
    path = Path(source).expanduser()
    if not path.exists():
        sys.exit(f"error: no such file: {path}")
    return path.read_bytes()


# The inference pipeline refuses anything over 10MB. Design exports blow past
# that routinely, and the raw failure is a stack trace deep inside the
# downloader, so shrink here instead. OCR finds the largest word either way,
# and the classifier sees a small crop, so downscaling costs no accuracy.
MAX_BYTES = 10 * 1024 * 1024


def to_png(raw: bytes) -> bytes:
    import io

    img = Image.open(io.BytesIO(raw))

    # A transparent PNG converted straight to RGB lands on black, so black
    # artwork disappears and OCR reads nothing. Composite onto whichever of
    # white or black the artwork is not, judged from the visible pixels.
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
        alpha = img.getchannel("A")
        gray = img.convert("L")
        # Mean luminance of the opaque pixels only, straight off the
        # histograms, so light artwork lands on black and dark on white.
        opaque = alpha.point(lambda a: 255 if a > 8 else 0)
        lit = Image.composite(gray, Image.new("L", img.size, 0), opaque)
        hist = lit.histogram()
        n = sum(opaque.histogram()[128:])
        ink = sum(i * c for i, c in enumerate(hist)) / n if n else 128
        bg = 255 if ink < 128 else 0
        flat = Image.new("RGB", img.size, (bg, bg, bg))
        flat.paste(img, mask=alpha)
        img = flat
    else:
        img = img.convert("RGB")

    def encode(im):
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    png = encode(img)
    while len(png) > MAX_BYTES:
        w, h = img.size
        if max(w, h) <= 1400:
            # Already small and still too heavy: fall back to JPEG quality.
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
            return out.getvalue()
        img = img.resize((int(w * 0.7), int(h * 0.7)), Image.LANCZOS)
        png = encode(img)
        print(f"  Downscaled to {img.size[0]}x{img.size[1]} to fit the 10MB limit.")
    return png


def serve_once(png_bytes: bytes):
    """Start a localhost server for the PNG; return (url, shutdown_fn)."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png_bytes)))
            self.end_headers()
            self.wfile.write(png_bytes)

        def log_message(self, *_):  # silence default logging
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/image.png", httpd.shutdown


def print_table(result: dict) -> None:
    word = result.get("word") or ""
    matches = result.get("font_matches") or []
    print(f'\n  Text read by OCR: "{word}"')
    if not word.strip():
        print("  OCR found no word. Crop tighter to a single word of the type,")
        print("  raise the contrast, and try again. No match is worth reading.")
        print()
        return
    if len(word.strip()) < 2:
        print("  OCR reading looks weak. Treat matches as rough guesses.")
    if not matches:
        print("  No font matches returned.\n")
        return
    print(f"\n  Top {len(matches)} font matches:\n")
    for i, m in enumerate(matches, 1):
        name = m.get("name", "?")
        score = m.get("score", 0)
        print(f"    {i:>2}. {name:<28} {score:>5.0%}")
        for f in m.get("fonts", [])[:1]:
            if f.get("url"):
                print(f"        ↳ {f['url']}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Identify fonts in an image.")
    ap.add_argument("image", help="Local image path OR http(s) URL")
    ap.add_argument("--top-k", type=int, default=5, help="Number of matches (default 5)")
    ap.add_argument("--json", action="store_true", help="Print raw JSON")
    args = ap.parse_args()

    raw = load_image_bytes(args.image)
    try:
        png = to_png(raw)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"error: could not open that image ({e})")

    url, shutdown = serve_once(png)
    try:
        result = run_inference_from_url(
            image_url=url, model_path=MODEL_PATH, top_k=args.top_k
        )
    finally:
        shutdown()

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
