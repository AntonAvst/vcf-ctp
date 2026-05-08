#!/usr/bin/env python3
"""
debug_osd.py — diagnose OSD timestamp OCR

Usage:
    python3 debug_osd.py --video /path/to/video.mp4 [--frame 0]
"""

import argparse, re, sys
from pathlib import Path
import cv2
import numpy as np

try:
    import pytesseract
    from PIL import Image
except ImportError:
    sys.exit("pytesseract / pillow not installed. Run: pip install pytesseract pillow")

_OSD_RE = re.compile(r"(\d{4})[/\-](\d{2})[/\-](\d{2})\s+(\d{2}):(\d{2}):(\d{2})")


def try_ocr(roi_bgr, scale, threshold):
    roi_up = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    bw = cv2.bitwise_not(bw)
    bw = cv2.copyMakeBorder(bw, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789/:- "
    raw = pytesseract.image_to_string(Image.fromarray(bw), config=cfg).strip()
    matched = bool(_OSD_RE.search(raw))
    return raw, matched, bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Could not read frame")

    H, W = frame.shape[:2]
    print(f"\nFrame size: {W}x{H}\n")

    cv2.imwrite("debug_full_frame.jpg", frame)
    print("Saved: debug_full_frame.jpg")

    # Pixel brightness inspector
    roi_inspect = frame[0:30, 0:250]
    gray_inspect = cv2.cvtColor(roi_inspect, cv2.COLOR_BGR2GRAY)
    print(f"Top-left pixel stats (0:30, 0:250):")
    print(f"  min={gray_inspect.min()}  max={gray_inspect.max()}  "
          f"mean={gray_inspect.mean():.1f}  "
          f"pixels>180: {(gray_inspect>180).sum()}  "
          f"pixels>150: {(gray_inspect>150).sum()}")
    cv2.imwrite("debug_gray_crop.jpg",
                cv2.resize(gray_inspect, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    print("Saved: debug_gray_crop.jpg  (4x upscale, raw grayscale)\n")

    crops = [
        ("c200x22",  0,  0, 200, 22),
        ("c220x24",  0,  0, 220, 24),
        ("c250x26",  0,  0, 250, 26),
        ("c280x28",  0,  0, 280, 28),
    ]
    thresholds = [150, 160, 170, 180, 190, 200]

    found = False
    for (clabel, x1, y1, x2, y2) in crops:
        roi = frame[y1:min(y2,H), x1:min(x2,W)]
        for thr in thresholds:
            raw, matched, bw = try_ocr(roi, 4, thr)
            status = "✓ MATCH" if matched else "✗"
            print(f"  [{status}]  crop=({x1},{y1},{x2},{y2})  thr={thr}  →  '{raw}'")
            if matched and not found:
                found = True
                cv2.imwrite(f"debug_winning_crop.jpg", bw)
                print(f"\n  *** SUCCESS ***")
                print(f"  _OSD_CROP = ({x1}, {y1}, {x2}, {y2})")
                print(f"  Update threshold to {thr} in _ocr_frame_timestamp\n")

    if not found:
        print("\nNo match. Share debug_gray_crop.jpg — it shows raw pixel values of the OSD region.")


if __name__ == "__main__":
    main()