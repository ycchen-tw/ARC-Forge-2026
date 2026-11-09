"""Smoke test of a served Qwen3.8-FN checkpoint on all three input paths: text, one image, one short video.

    python smoke_mm.py --url http://127.0.0.1:<port> --model <served name> [--ffmpeg <ffmpeg binary>]

17 * 23 (text), a 4 x 5 cell grid drawn here (3 red cells, a blue bottom-left cell), and an 8-frame mp4 of a red square
moving left to right (needs --ffmpeg with libx264; without it the video case is skipped and says so). Effort low,
production sampling (T=1, top-p 0.95). Prints each answer and its token usage, then ``SMOKE PASS`` or ``SMOKE FAIL``
(exit 1). A broken pruned checkpoint (wrong expert order, router rows, layout) answers garbage on at least one of them.
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


def chat(url, model, content, max_tokens=8192):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "reasoning_effort": "low"}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return (r["choices"][0]["message"].get("content") or "").strip(), r["usage"], round(time.time() - t, 1)


def png_url(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def grid_image():
    img = Image.new("RGB", (5 * 32, 4 * 32), "black")
    d = ImageDraw.Draw(img)
    for c in (1, 2, 3):
        d.rectangle([c * 32, 32, c * 32 + 31, 63], fill=(255, 0, 0))
    d.rectangle([0, 96, 31, 127], fill=(0, 0, 255))
    return img


def video_url(ffmpeg):
    with tempfile.TemporaryDirectory() as td:
        for i in range(8):
            img = Image.new("RGB", (256, 128), "black")
            ImageDraw.Draw(img).rectangle([i * 28, 48, i * 28 + 31, 79], fill=(255, 0, 0))
            img.save(f"{td}/f{i:02d}.png")
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-framerate", "4", "-i", f"{td}/f%02d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{td}/v.mp4"], check=True)
        return "data:video/mp4;base64," + base64.b64encode(Path(f"{td}/v.mp4").read_bytes()).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ffmpeg", help="ffmpeg with libx264 (its lib dir on LD_LIBRARY_PATH); omit to skip the video")
    args = p.parse_args()
    results = {}

    a, u, s = chat(args.url, args.model, "What is 17 * 23? Reply with just the number.")
    results["text"] = {"answer": a, "ok": "391" in a, "usage": u, "s": s}

    a, u, s = chat(args.url, args.model, [
        {"type": "image_url", "image_url": {"url": png_url(grid_image())}},
        {"type": "text", "text": "This image is a grid of square cells (4 rows, 5 columns). How many cells are red, "
                                 "and what color is the single cell in the bottom-left corner? Answer as: <count>, <color>"}])
    results["image"] = {"answer": a, "ok": "3" in a and "blue" in a.lower(), "usage": u, "s": s}

    if args.ffmpeg:
        a, u, s = chat(args.url, args.model, [
            {"type": "video_url", "video_url": {"url": video_url(args.ffmpeg)}},
            {"type": "text", "text": "In this short video a colored square moves. What color is it, and does it move "
                                     "left-to-right or right-to-left? Answer as: <color>, <direction>"}])
        low = a.lower().replace(" to ", "-to-")
        results["video"] = {"answer": a, "ok": "red" in low and "left-to-right" in low, "usage": u, "s": s}
    else:
        print("· video skipped (no --ffmpeg)")

    for name, r in results.items():
        print(name, json.dumps(r, ensure_ascii=False))
    ok = all(r["ok"] for r in results.values())
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
