#!/usr/bin/env python3
"""
Trace the outline of every separate green area in an image and save as DXF,
with every turn exactly 90 degrees, measured along each shape's own orientation.
Curved shapes are detected and kept as curves.

Install:  pip install opencv-python ezdxf numpy FreeSimpleGUI

Run the GUI:
    python MAP-EXTRACTOR.py
    python MAP-EXTRACTOR.py --gui

Run from the command line:
    python MAP-EXTRACTOR.py input.png output.dxf
    python MAP-EXTRACTOR.py photo.jpg out.dxf --holes --scale 0.1
    python MAP-EXTRACTOR.py photo.jpg out.dxf --grid 6
    python MAP-EXTRACTOR.py photo.jpg out.dxf --color "#2E9E3A" --hue-tol 15
    python MAP-EXTRACTOR.py photo.jpg out.dxf --angle 30
    python MAP-EXTRACTOR.py photo.jpg out.dxf --angle-step 5
    python MAP-EXTRACTOR.py photo.jpg out.dxf --angle-step 0

Colour options
    --color HEX          Match this colour instead of the hue/saturation/brightness
                         limits, for example "#2E9E3A". The tolerances below set how far
                         from the colour a pixel may be, in OpenCV units.
    --hue-tol N          Hue range is the colour's hue +/- N (0-179 scale, default 20).
    --sat-tol N          Saturation range is the colour's +/- N (0-255, default 100).
    --val-tol N          Brightness range is the colour's +/- N (0-255, default 100).

Angle options
    --angle A            Use A degrees for every shape, exactly as given.
                         Positive values turn counter-clockwise as seen in the DXF.
                         No snapping is applied.
    --angle-step S       Without --angle, the detected orientation of each shape is
                         snapped to the nearest multiple of S degrees (default 15).
                         S = 0 keeps the detected angle unsnapped.
    --angle-tolerance T  Only snap when the detected angle is within T degrees of a
                         multiple of S. Default is S / 2, which always snaps.

Each green blob becomes a closed LWPOLYLINE on layer "GREEN".
With --holes, holes inside blobs are traced too (layer "HOLES").
Curved shapes keep their simplified contour and ignore the angle settings.
Their contour is smoothed first (--curve-smooth) so pixel noise does not show
up as jagged lines, then simplified with --curve-simplify.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass, fields
from typing import Optional

import cv2
import ezdxf
import numpy as np


# ---------------------------------------------------------------------------
# Options shared by the command line and the GUI
# ---------------------------------------------------------------------------

@dataclass
class Options:
    input: str = ""
    output: str = ""
    hue_min: int = 35
    hue_max: int = 85
    sat_min: int = 60
    val_min: int = 40
    color: Optional[str] = None
    hue_tol: int = 20
    sat_tol: int = 100
    val_tol: int = 100
    blur: int = 0
    min_area: float = 50.0
    simplify: float = 2.0
    smooth: int = 7
    min_step: float = 5.0
    grid: int = 0
    angle: Optional[float] = None
    angle_step: float = 15.0
    angle_tolerance: Optional[float] = None
    curve_min_points: int = 8
    curve_angle_tolerance: float = 12.0
    curve_smooth: int = 11
    curve_simplify: float = 1.5
    no_curve_detection: bool = False
    scale: float = 1.0
    holes: bool = False
    save_mask: Optional[str] = None


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def hex_to_rgb(text):
    """Parse '#RRGGBB', 'RRGGBB' or '#RGB' into an (r, g, b) tuple. Raises ValueError."""
    t = str(text).strip().lstrip("#")
    if len(t) == 3:
        t = "".join(ch * 2 for ch in t)
    if len(t) != 6:
        raise ValueError(f"Not a hex colour: {text!r}")
    return tuple(int(t[i:i + 2], 16) for i in (0, 2, 4))


def color_limits(o):
    """
    HSV limits used for the mask, as (hue_lo, hue_hi, sat_lo, sat_hi, val_lo, val_hi).

    With o.color set, the limits are built around that colour using the tolerances.
    Hue may come back below 0 or above 179, which build_mask wraps around.
    """
    if o.color:
        r, g, b = hex_to_rgb(o.color)
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        return (h - o.hue_tol, h + o.hue_tol,
                max(0, s - o.sat_tol), min(255, s + o.sat_tol),
                max(0, v - o.val_tol), min(255, v + o.val_tol))
    return o.hue_min, o.hue_max, o.sat_min, 255, o.val_min, 255


def build_mask(img_bgr, o):
    if o.blur > 0:
        img_bgr = cv2.GaussianBlur(img_bgr, (o.blur | 1, o.blur | 1), 0)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    hl, hh, sl, sh, vl, vh = color_limits(o)

    if hh - hl >= 179:
        hue_ranges = [(0, 179)]
    elif hl < 0:                      # range wraps below 0
        hue_ranges = [(0, hh), (180 + hl, 179)]
    elif hh > 179:                    # range wraps above 179
        hue_ranges = [(hl, 179), (0, hh - 180)]
    else:
        hue_ranges = [(hl, hh)]

    mask = None
    for lo, hi in hue_ranges:
        part = cv2.inRange(hsv, np.array([lo, sl, vl]), np.array([hi, sh, vh]))
        mask = part if mask is None else cv2.bitwise_or(mask, part)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def compute_mask(img, o):
    """Green mask including the optional edge smoothing step."""
    mask = build_mask(img, o)
    if o.smooth > 0:
        k = o.smooth | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def _merge_runs(runs):
    """Merge neighbouring runs (cyclic) that share the same orientation."""
    changed = True
    while changed and len(runs) > 1:
        changed = False
        out = []
        for r in runs:
            if out and out[-1]["o"] == r["o"]:
                a = out[-1]
                w = a["w"] + r["w"]
                a["c"] = (a["c"] * a["w"] + r["c"] * r["w"]) / w if w else a["c"]
                a["w"] = w
                changed = True
            else:
                out.append(dict(r))
        if len(out) > 1 and out[0]["o"] == out[-1]["o"]:
            last = out.pop()
            a = out[0]
            w = a["w"] + last["w"]
            a["c"] = (a["c"] * a["w"] + last["c"] * last["w"]) / w if w else a["c"]
            a["w"] = w
            changed = True
        runs = out
    return runs


def orthogonalize(pts, min_step):
    """
    Turn a closed polygon into a rectilinear one.
    Each edge is snapped to horizontal or vertical (whichever it is closer to),
    consecutive same-direction edges are merged, and corners are placed at the
    intersections. Steps shorter than min_step are removed.
    Returns a list of (x, y) or None if it collapses.
    """
    n = len(pts)
    runs = []
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) >= abs(dy):  # horizontal edge: fixed y
            runs.append({"o": "H", "c": (y0 + y1) / 2.0, "w": abs(dx) + 1e-6})
        else:                   # vertical edge: fixed x
            runs.append({"o": "V", "c": (x0 + x1) / 2.0, "w": abs(dy) + 1e-6})

    runs = _merge_runs(runs)

    # Drop tiny steps, merging their neighbours, while a rectangle-or-bigger remains
    while len(runs) > 4:
        k = min(range(len(runs)), key=lambda i: runs[i]["w"])
        if runs[k]["w"] >= min_step:
            break
        del runs[k]
        runs = _merge_runs(runs)

    if len(runs) < 4 or len(runs) % 2:
        return None

    verts = []
    for i in range(len(runs)):
        a, b = runs[i], runs[(i + 1) % len(runs)]
        h, v = (a, b) if a["o"] == "H" else (b, a)
        verts.append((v["c"], h["c"]))  # corner = (x of vertical, y of horizontal)

    # Remove any duplicate consecutive vertices
    clean = [verts[0]]
    for p in verts[1:]:
        if abs(p[0] - clean[-1][0]) > 1e-9 or abs(p[1] - clean[-1][1]) > 1e-9:
            clean.append(p)
    return clean if len(clean) >= 4 else None


def fix_diagonals(pts):
    """Replace 1-px diagonal corner cuts in a grid-traced contour with true 90-degree corners."""
    n = len(pts)
    diag = [pts[i][0] != pts[(i + 1) % n][0] and pts[i][1] != pts[(i + 1) % n][1]
            for i in range(n)]
    out, consumed = [], set()
    for i in range(n):
        if i in consumed:
            continue
        if diag[i]:
            a, b, prev = pts[i], pts[(i + 1) % n], pts[i - 1]
            out.append((b[0], a[1]) if prev[1] == a[1] else (a[0], b[1]))
            consumed.add((i + 1) % n)
        else:
            out.append(pts[i])
    # drop collinear / duplicate points
    m, clean = len(out), []
    for j in range(m):
        p0, p1, p2 = out[j - 1], out[j], out[(j + 1) % m]
        if p1 == p0:
            continue
        if (p0[0] == p1[0] == p2[0]) or (p0[1] == p1[1] == p2[1]):
            continue
        clean.append(p1)
    return clean


def estimate_angle(approx_pts):
    """Dominant edge direction of a polygon, folded to a 90-degree period (radians, image coords)."""
    n = len(approx_pts)
    sx = sy = 0.0
    for i in range(n):
        x0, y0 = approx_pts[i]
        x1, y1 = approx_pts[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L == 0:
            continue
        t = math.atan2(dy, dx)
        sx += L * math.cos(4 * t)   # 4*theta makes 0/90/180/270 all agree
        sy += L * math.sin(4 * t)
    if sx == 0 and sy == 0:
        return 0.0
    return math.atan2(sy, sx) / 4.0


def is_curved_shape(approx_pts, contour=None, min_points=8, angle_tolerance=12.0):
    """
    Detect genuinely curved/rounded outlines.

    Curved shapes are exempt from the 90-degree rule and keep their
    simplified contour. Normal rectangles/L-shapes remain orthogonal.
    """
    if len(approx_pts) < min_points or len(approx_pts) <= 6:
        return False

    angles = []
    lengths = []
    n = len(approx_pts)
    for i in range(n):
        x0, y0 = approx_pts[i]
        x1, y1 = approx_pts[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)) % 180.0)
        lengths.append(L)

    if len(angles) < min_points:
        return False

    dominant = math.degrees(estimate_angle(approx_pts)) % 180.0
    tol = angle_tolerance

    def angle_distance(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    total = sum(lengths)
    off_axis = sum(
        L for a, L in zip(angles, lengths)
        if min(angle_distance(a, dominant),
               angle_distance(a, (dominant + 90.0) % 180.0)) > tol
    )
    off_ratio = off_axis / total if total else 0.0

    hull_signal = False
    if contour is not None and len(contour) >= 5:
        hull = cv2.convexHull(contour)
        hull_perim = cv2.arcLength(hull, True)
        perim = cv2.arcLength(contour, True)
        if hull_perim > 0:
            hull_signal = (perim / hull_perim) > 1.03

    # A substantial amount of non-orthogonal boundary means the shape is
    # curved. The second condition catches rounded outlines with many points.
    return off_ratio >= 0.20 or (
        len(approx_pts) >= 10 and hull_signal and off_ratio >= 0.10
    )


def smooth_closed(pts, window):
    """Circular moving average over a closed contour. window is in points (pixels)."""
    w = int(window) | 1
    n = len(pts)
    if w < 3 or n < 2 * w:
        return [(float(x), float(y)) for x, y in pts]
    half = w // 2
    arr = np.asarray(pts, dtype=float)
    ext = np.concatenate([arr[-half:], arr, arr[:half]])
    k = np.ones(w) / w
    xs = np.convolve(ext[:, 0], k, mode="valid")
    ys = np.convolve(ext[:, 1], k, mode="valid")
    return list(zip(xs.tolist(), ys.tolist()))


def choose_angle(detected_deg, o):
    """
    Return the orientation to use for one shape, in radians, in image space.

    A fixed angle (o.angle) is used as given. It is entered counter-clockwise
    as seen in the DXF, so it is negated here because image Y points down.
    Otherwise the detected angle is snapped to o.angle_step.
    """
    if o.angle is not None:
        deg = -o.angle
    else:
        deg = detected_deg
        step = o.angle_step
        if step and step > 0:
            snapped = round(deg / step) * step
            tol = step / 2.0 if o.angle_tolerance is None else o.angle_tolerance
            if abs(deg - snapped) <= tol + 1e-9:
                deg = snapped
    return math.radians(deg % 90.0)


def rot(pts, ang, cx, cy):
    """Rotate points by ang about (cx, cy)."""
    c, s_ = math.cos(ang), math.sin(ang)
    return [(cx + (x - cx) * c - (y - cy) * s_, cy + (x - cx) * s_ + (y - cy) * c)
            for x, y in pts]


def grid_shape(rot_pts, g):
    """Staircase a (rotated) polygon onto a g-pixel grid; returns rectilinear points or None."""
    xs = [p[0] for p in rot_pts]
    ys = [p[1] for p in rot_pts]
    ox, oy = math.floor(min(xs)) - 2 * g, math.floor(min(ys)) - 2 * g
    W = int(math.ceil(max(xs)) - ox + 2 * g)
    H = int(math.ceil(max(ys)) - oy + 2 * g)
    m = np.zeros((H, W), np.uint8)
    poly = np.array([[round(x - ox), round(y - oy)] for x, y in rot_pts], np.int32)
    cv2.fillPoly(m, [poly], 255)
    small = cv2.resize(m, (-(-W // g), -(-H // g)), interpolation=cv2.INTER_AREA)
    small = (small >= 127).astype(np.uint8) * 255
    m = np.kron(small, np.ones((g, g), np.uint8))[:H, :W]
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    pts = fix_diagonals([(float(x), float(y)) for x, y in c[:, 0, :]])
    if len(pts) < 4:
        return None
    return [(x + ox, y + oy) for x, y in pts]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(o):
    """
    Run the full conversion.

    Returns (outline count, angles used for right-angle shapes in degrees,
    number of outer shapes kept as curves).
    """
    img = cv2.imread(o.input)
    if img is None:
        raise ValueError(f"Could not read image: {o.input}")
    h = img.shape[0]

    mask = compute_mask(img, o)
    if o.save_mask:
        cv2.imwrite(o.save_mask, mask)

    mode = cv2.RETR_CCOMP if o.holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_NONE)

    doc = ezdxf.new("R2010")
    doc.layers.add("GREEN", color=3)
    doc.layers.add("HOLES", color=1)
    msp = doc.modelspace()

    def approx(c):
        a = cv2.approxPolyDP(c, o.simplify, True) if o.simplify > 0 else c
        return [(float(x), float(y)) for x, y in a[:, 0, :]]

    def approx_curve(c):
        """Smooth the full-resolution contour, then simplify it, for curved shapes."""
        raw = [(float(x), float(y)) for x, y in c[:, 0, :]]
        sm = smooth_closed(raw, o.curve_smooth) if o.curve_smooth >= 3 else raw
        tol = o.curve_simplify if o.curve_simplify > 0 else o.simplify
        arr = np.asarray(sm, dtype=np.float32).reshape(-1, 1, 2)
        a = cv2.approxPolyDP(arr, tol, True) if tol > 0 else arr
        return [(float(x), float(y)) for x, y in a[:, 0, :]]

    def is_hole_at(i):
        return o.holes and hierarchy[0][i][3] != -1

    # Pass 1: orientation, rotation centre and curve flag for every outer shape
    transform = {}
    curved_count = 0
    for i, c in enumerate(contours):
        if is_hole_at(i) or cv2.contourArea(c) < o.min_area:
            continue
        a = approx(c)
        if len(a) < 3:
            continue
        curved = (
            False if o.no_curve_detection else
            is_curved_shape(a, c, o.curve_min_points, o.curve_angle_tolerance)
        )
        detected = 0.0 if o.angle is not None else math.degrees(estimate_angle(a))
        ang = choose_angle(detected, o)
        transform[i] = (
            ang,
            sum(x for x, _ in a) / len(a),
            sum(y for _, y in a) / len(a),
            curved,
        )
        if curved:
            curved_count += 1

    # Pass 2: rectilinearise in each shape's own frame, rotate back, write
    count = 0
    angles = []
    for i, c in enumerate(contours):
        hole = is_hole_at(i)
        key = hierarchy[0][i][3] if hole else i
        if key not in transform or cv2.contourArea(c) < o.min_area:
            continue
        ang, cx, cy, curved = transform[key]
        a = approx(c)
        if len(a) < 3:
            continue

        if curved:
            # Curved shapes are exempt from the 90-degree rule.
            # Keep a smoothed, simplified contour in the original image frame.
            out_pts = approx_curve(c)
            if len(out_pts) < 3:
                out_pts = a
        else:
            r = rot(a, -ang, cx, cy)  # into the shape's own axis-aligned frame
            if o.grid > 0:
                ortho = grid_shape(r, o.grid)
            else:
                ortho = orthogonalize(r, o.min_step)
            if ortho is None:  # fallback: shape's oriented bounding box
                xs, ys = [q[0] for q in r], [q[1] for q in r]
                ortho = [(min(xs), min(ys)), (max(xs), min(ys)),
                         (max(xs), max(ys)), (min(xs), max(ys))]
            out_pts = rot(ortho, ang, cx, cy)  # back to the picture
            if not hole:
                # Report in DXF convention (counter-clockwise positive).
                angles.append((-math.degrees(ang)) % 90.0)

        out = [(x * o.scale, (h - y) * o.scale) for x, y in out_pts]  # flip Y for DXF
        msp.add_lwpolyline(out, close=True,
                           dxfattribs={"layer": "HOLES" if hole else "GREEN"})
        count += 1

    doc.saveas(o.output)
    return count, angles, curved_count


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser():
    d = Options()
    p = argparse.ArgumentParser(
        description="Trace green areas to a DXF with right angles aligned to each shape")
    p.add_argument("input", nargs="?")
    p.add_argument("output", nargs="?")
    p.add_argument("--gui", action="store_true", help="Open the GUI")

    p.add_argument("--hue-min", type=int, default=d.hue_min, help="Min hue 0-179 (default 35)")
    p.add_argument("--hue-max", type=int, default=d.hue_max, help="Max hue 0-179 (default 85)")
    p.add_argument("--sat-min", type=int, default=d.sat_min, help="Min saturation 0-255 (default 60)")
    p.add_argument("--val-min", type=int, default=d.val_min, help="Min brightness 0-255 (default 40)")
    p.add_argument("--color", default=None,
                   help='Match this hex colour, e.g. "#2E9E3A", instead of the '
                        "hue/saturation/brightness limits")
    p.add_argument("--hue-tol", type=int, default=d.hue_tol,
                   help="With --color: hue range is the colour's hue +/- this (default 20)")
    p.add_argument("--sat-tol", type=int, default=d.sat_tol,
                   help="With --color: saturation +/- this (default 100)")
    p.add_argument("--val-tol", type=int, default=d.val_tol,
                   help="With --color: brightness +/- this (default 100)")
    p.add_argument("--blur", type=int, default=d.blur, help="Blur kernel size before thresholding")
    p.add_argument("--min-area", type=float, default=d.min_area,
                   help="Ignore blobs smaller than this many pixels")
    p.add_argument("--simplify", type=float, default=d.simplify,
                   help="Pre-straightening tolerance in pixels (default 2)")
    p.add_argument("--smooth", type=int, default=d.smooth,
                   help="Edge smoothing kernel size (default 7, 0 = off)")
    p.add_argument("--min-step", type=float, default=d.min_step,
                   help="Remove steps shorter than this many pixels (default 5)")
    p.add_argument("--grid", type=int, default=d.grid,
                   help="Staircase mode: snap each shape to N-pixel cells in its own "
                        "rotated frame (default 0 = off)")
    p.add_argument("--angle", type=float, default=None,
                   help="Use this angle in degrees for every shape, with no snapping. "
                        "Positive is counter-clockwise in the DXF.")
    p.add_argument("--angle-step", type=float, default=d.angle_step,
                   help="Snap each detected orientation to multiples of this many "
                        "degrees (default 15, 0 = no snapping)")
    p.add_argument("--angle-tolerance", type=float, default=None,
                   help="Only snap when within this many degrees of a multiple of the "
                        "step (default: half the step)")
    p.add_argument("--curve-min-points", type=int, default=d.curve_min_points,
                   help="Minimum simplified contour points for curved-shape detection (default 8)")
    p.add_argument("--curve-angle-tolerance", type=float, default=d.curve_angle_tolerance,
                   help="Allowed deviation from perpendicular directions when detecting "
                        "curves (default 12 degrees)")
    p.add_argument("--curve-smooth", type=int, default=d.curve_smooth,
                   help="Smoothing window in pixels applied to curved outlines "
                        "(default 11, 0 = off). Raise it to remove jagged lines.")
    p.add_argument("--curve-simplify", type=float, default=d.curve_simplify,
                   help="Simplification tolerance in pixels for curved outlines "
                        "(default 1.5, 0 = use --simplify)")
    p.add_argument("--no-curve-detection", action="store_true",
                   help="Always apply the 90-degree rule, including to curved shapes")
    p.add_argument("--scale", type=float, default=d.scale, help="DXF units per pixel (default 1)")
    p.add_argument("--holes", action="store_true", help="Also trace holes inside green areas")
    p.add_argument("--save-mask", help="Optional path to save the green mask for debugging")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.gui or not argv:
        run_gui(args.input, args.output)
        return

    if not args.input or not args.output:
        parser.error("input and output are required, or use --gui")

    opts = Options(**{f.name: getattr(args, f.name) for f in fields(Options)})
    if opts.color:
        try:
            hex_to_rgb(opts.color)
        except ValueError:
            parser.error(f"--color must be a hex value like #2E9E3A, got {opts.color!r}")

    try:
        count, _, curved = convert(opts)
    except ValueError as e:
        sys.exit(str(e))

    print(f"Wrote {count} outline(s) to {opts.output} ({curved} kept as curves)")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def _import_gui():
    try:
        import FreeSimpleGUI as sg
        return sg
    except ImportError:
        sys.exit("FreeSimpleGUI is not installed. Install it with:\n"
                 "  pip install FreeSimpleGUI")


def _read_options(v):
    """Build Options from the GUI values. Raises ValueError naming the bad field."""

    def num(key, cast, label, allow_blank=False):
        text = str(v[key]).strip()
        if text == "":
            if allow_blank:
                return None
            raise ValueError(f"{label} is empty")
        try:
            return cast(text)
        except ValueError:
            kind = "a whole number" if cast is int else "a number"
            raise ValueError(f"{label} must be {kind}")

    o = Options()

    o.input = v["-IN-"].strip()
    o.output = v["-OUT-"].strip()
    if not o.input:
        raise ValueError("Choose an input image")
    if not o.output:
        raise ValueError("Choose an output DXF file")
    if not o.output.lower().endswith(".dxf"):
        o.output += ".dxf"

    if v["-USECOLOR-"]:
        try:
            r, g_, b = hex_to_rgb(v["-COLOR-"])
        except ValueError:
            raise ValueError("Colour must be a hex value like #00C800")
        o.color = f"#{r:02X}{g_:02X}{b:02X}"
        o.hue_tol = num("-HTOL-", int, "Hue tolerance")
        o.sat_tol = num("-STOL-", int, "Saturation tolerance")
        o.val_tol = num("-VTOL-", int, "Brightness tolerance")
        if min(o.hue_tol, o.sat_tol, o.val_tol) < 0:
            raise ValueError("Tolerances must not be negative")
    else:
        o.hue_min = num("-HMIN-", int, "Hue min")
        o.hue_max = num("-HMAX-", int, "Hue max")
        o.sat_min = num("-SMIN-", int, "Saturation min")
        o.val_min = num("-VMIN-", int, "Brightness min")
    o.blur = num("-BLUR-", int, "Blur")
    o.min_area = num("-AREA-", float, "Min area")
    o.simplify = num("-SIMP-", float, "Simplify")
    o.smooth = num("-SMOOTH-", int, "Smooth")
    o.min_step = num("-STEP-", float, "Min step")
    o.grid = num("-GRID-", int, "Grid")
    o.scale = num("-SCALE-", float, "Scale")
    o.holes = bool(v["-HOLES-"])

    if v["-FIXED-"]:
        o.angle = num("-ANGLE-", float, "Angle")
    else:
        o.angle_step = num("-ASTEP-", float, "Snap step")
        o.angle_tolerance = num("-ATOL-", float, "Snap tolerance", allow_blank=True)

    o.no_curve_detection = not v["-CURVES-"]
    if v["-CURVES-"]:
        o.curve_min_points = num("-CPTS-", int, "Curve min points")
        o.curve_angle_tolerance = num("-CTOL-", float, "Curve angle tolerance")
        o.curve_smooth = num("-CSM-", int, "Curve smooth")
        o.curve_simplify = num("-CSIMP-", float, "Curve simplify")

    o.save_mask = v["-MASK-"].strip() or None

    if not o.color and o.hue_min > o.hue_max:
        raise ValueError("Hue min must not exceed hue max")
    if o.scale <= 0:
        raise ValueError("Scale must be greater than 0")
    if o.grid < 0 or o.blur < 0 or o.smooth < 0:
        raise ValueError("Grid, blur and smooth must not be negative")
    if o.angle is None and o.angle_step < 0:
        raise ValueError("Snap step must not be negative")

    return o


def _safe_convert(o):
    try:
        count, angles, curved = convert(o)
    except Exception as e:  # shown in the GUI log
        return False, f"Error: {e}"

    lines = [f"Wrote {count} outline(s) to {o.output}"]
    if curved:
        lines.append(f"{curved} shape(s) kept as curves")
    if angles:
        used = sorted({round(a, 2) for a in angles})
        shown = ", ".join(f"{a:g}" for a in used[:12])
        more = " ..." if len(used) > 12 else ""
        lines.append(f"Angles used (degrees, mod 90): {shown}{more}")
    if o.color:
        hl, hh, sl, sh, vl, vh = color_limits(o)
        lines.append(f"Colour {o.color}: hue {hl} to {hh}, saturation {sl} to {sh}, "
                     f"brightness {vl} to {vh}")
    if count == 0:
        lines.append("No green areas found. Open the mask preview and adjust the hue, "
                     "saturation or brightness limits.")
    return True, "\n".join(lines)


def _preview_window(sg, o):
    img = cv2.imread(o.input)
    if img is None:
        sg.popup_error(f"Could not read image: {o.input}")
        return

    mask = compute_mask(img, o)

    # Keep selected pixels in colour and dim everything else.
    shown = img.copy()
    shown[mask == 0] = shown[mask == 0] // 5

    h, w = shown.shape[:2]
    f = min(1.0, 900.0 / w, 650.0 / h)
    if f < 1.0:
        shown = cv2.resize(shown, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".png", shown)
    if not ok:
        sg.popup_error("Could not build the preview")
        return

    layout = [
        [sg.Image(data=buf.tobytes())],
        [sg.Text("Bright areas will be traced. Dim areas are ignored.")],
        [sg.Button("Close")],
    ]
    sg.Window("Mask preview", layout, modal=True, finalize=True).read(close=True)


# ---------------------------------------------------------------------------
# Scrolling number fields
# ---------------------------------------------------------------------------

def _step_once(val, spec, d):
    """Move val one step up (d = 1) or down (d = -1) according to its field spec."""
    kind = spec.get("kind", "float")

    if kind == "odd":
        # Kernel sizes: 0 means off, otherwise an odd number of at least 3.
        n = int(round(val))
        if d > 0:
            n = 3 if n < 3 else (n + 2 if n % 2 else n + 1)
        else:
            n = 0 if n <= 3 else (n - 2 if n % 2 else n - 1)
        val = n
    elif kind == "mul":
        val = val * (1.1 if d > 0 else 1 / 1.1)
    else:
        val = val + d * spec.get("step", 1)

    lo, hi = spec.get("lo"), spec.get("hi")
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


def bump_text(text, spec, d, big=False, blank=None):
    """
    Return the field text after one scroll step, or None if text is not a number.
    big moves ten steps at once. blank is a function giving the starting text for an
    empty field.
    """
    raw = text.strip()
    if raw == "":
        raw = blank() if blank else str(spec.get("lo") or 0)
    try:
        val = float(raw)
    except ValueError:
        return None

    for _ in range(10 if big else 1):
        val = _step_once(val, spec, d)

    kind = spec.get("kind", "float")
    if kind in ("int", "odd"):
        return str(int(round(val)))
    if kind == "mul":
        return f"{val:.4g}"
    return f"{round(val, 4):g}"


def install_scroll(window, specs, blanks=None):
    """
    Make the numeric Input fields in specs change with the mouse wheel and the
    Up/Down keys. Hold Shift for ten steps at a time.

    specs maps an element key to {"kind": "int" | "float" | "odd" | "mul",
    "step": n, "lo": n, "hi": n}.
    """
    blanks = blanks or {}
    root = window.TKroot
    by_path = {}

    def apply(key, d, big):
        widget, spec = by_path[key]
        if str(widget.cget("state")) != "normal":
            return
        new = bump_text(widget.get(), spec, d, big, blanks.get(key))
        if new is not None:
            widget.delete(0, "end")
            widget.insert(0, new)

    for key, spec in specs.items():
        widget = window[key].Widget
        path = str(widget)
        by_path[path] = (widget, spec)
        by_path[key] = (widget, spec)

        for seq, d in (("<Up>", 1), ("<Down>", -1)):
            widget.bind(seq, lambda e, k=key, d=d: (apply(k, d, bool(e.state & 1)), "break")[1])

    def on_wheel(event):
        num = event.num if isinstance(event.num, int) else None
        if num == 4:
            d = 1
        elif num == 5:
            d = -1
        elif event.delta:
            d = 1 if event.delta > 0 else -1
        else:
            return None

        target = root.winfo_containing(event.x_root, event.y_root)
        item = by_path.get(str(target))
        if item is None:
            return None
        widget, spec = item
        key = next(k for k, (w, _) in by_path.items() if w is widget and k.startswith("-"))
        apply(key, d, bool(event.state & 1))
        return "break"

    root.bind_all("<MouseWheel>", on_wheel)
    root.bind_all("<Button-4>", on_wheel)
    root.bind_all("<Button-5>", on_wheel)


def _pick_from_image(sg, path):
    """Show the image and let the user click a pixel. Returns '#RRGGBB' or None."""
    img = cv2.imread(path)
    if img is None:
        sg.popup_error(f"Could not read image: {path}")
        return None

    h, w = img.shape[:2]
    f = min(1.0, 900.0 / w, 600.0 / h)
    shown = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA) if f < 1.0 else img
    sh, sw = shown.shape[:2]
    ok, buf = cv2.imencode(".png", shown)
    if not ok:
        sg.popup_error("Could not build the image view")
        return None

    layout = [
        [sg.Graph((sw, sh), (0, sh), (sw, 0), key="-G-", enable_events=True,
                  drag_submits=True)],
        [sg.Text("Click the image to sample a colour."),
         sg.Text("      ", key="-SW-", background_color="#000000"),
         sg.Input("", key="-HEX-", size=(9, 1), disabled=True)],
        [sg.Button("Use colour", key="-USE-", disabled=True), sg.Button("Cancel")],
    ]
    win = sg.Window("Pick a colour from the image", layout, modal=True, finalize=True)
    win["-G-"].draw_image(data=buf.tobytes(), location=(0, 0))

    picked = None
    result = None
    while True:
        event, values = win.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            break
        if event == "-G-":
            x, y = values["-G-"]
            ox = min(w - 1, max(0, int(x / f)))
            oy = min(h - 1, max(0, int(y / f)))
            # Median of a 5x5 patch so one noisy pixel does not decide the colour.
            patch = img[max(0, oy - 2):oy + 3, max(0, ox - 2):ox + 3].reshape(-1, 3)
            b, g_, r = (int(c) for c in np.median(patch, axis=0))
            picked = f"#{r:02X}{g_:02X}{b:02X}"
            win["-SW-"].update(background_color=picked)
            win["-HEX-"].update(picked)
            win["-USE-"].update(disabled=False)
        elif event == "-USE-":
            result = picked
            break
    win.close()
    return result


def run_gui(prefill_in=None, prefill_out=None):
    sg = _import_gui()
    d = Options()

    sg.theme("LightGrey1")

    label_w = 22
    in_w = 9

    def field(label, key, default, tip, w=None):
        return [
            sg.Text(label, size=(w or label_w, 1)),
            sg.Input(str(default), key=key, size=(in_w, 1), tooltip=tip),
        ]

    image_types = (("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                   ("All files", "*.*"))

    files = sg.Frame("Files", [
        [sg.Text("Input image", size=(label_w, 1)),
         sg.Input(prefill_in or "", key="-IN-", size=(48, 1), enable_events=True),
         sg.FileBrowse("Browse", file_types=image_types)],
        [sg.Text("Output DXF", size=(label_w, 1)),
         sg.Input(prefill_out or "", key="-OUT-", size=(48, 1)),
         sg.SaveAs("Browse", file_types=(("DXF", "*.dxf"),))],
        [sg.Text("Save mask (optional)", size=(label_w, 1)),
         sg.Input("", key="-MASK-", size=(48, 1)),
         sg.SaveAs("Browse", file_types=(("PNG", "*.png"),))],
    ])

    default_hex = "#00C800"

    target = sg.Frame("Target colour", [
        [sg.Checkbox("Match this colour instead of the limits", key="-USECOLOR-",
                     default=False, enable_events=True)],
        [sg.Text("Hex", size=(6, 1)),
         sg.Input(default_hex, key="-COLOR-", size=(10, 1), enable_events=True,
                  tooltip="For example #00C800"),
         sg.Text("      ", key="-SWATCH-", background_color=default_hex, relief=sg.RELIEF_SUNKEN),
         sg.Button("Choose...", key="-CHOOSE-"),
         sg.Button("Pick from image", key="-PICK-")],
        [sg.Text("Hue +/-", size=(14, 1)),
         sg.Input(str(d.hue_tol), key="-HTOL-", size=(in_w, 1), disabled=True,
                  tooltip="0-179 scale. How far the hue may differ from the colour")],
        [sg.Text("Saturation +/-", size=(14, 1)),
         sg.Input(str(d.sat_tol), key="-STOL-", size=(in_w, 1), disabled=True,
                  tooltip="0-255 scale")],
        [sg.Text("Brightness +/-", size=(14, 1)),
         sg.Input(str(d.val_tol), key="-VTOL-", size=(in_w, 1), disabled=True,
                  tooltip="0-255 scale")],
    ])

    colour = sg.Frame("Green detection", [
        field("Hue min (0-179)", "-HMIN-", d.hue_min, "Lowest hue counted as green"),
        field("Hue max (0-179)", "-HMAX-", d.hue_max, "Highest hue counted as green"),
        field("Saturation min", "-SMIN-", d.sat_min, "0-255. Raise to reject pale colours"),
        field("Brightness min", "-VMIN-", d.val_min, "0-255. Raise to reject dark pixels"),
        field("Min area (px)", "-AREA-", d.min_area, "Ignore blobs smaller than this"),
    ])

    smoothing = sg.Frame("Smoothing (raise to remove jagged lines)", [
        field("Blur before mask (px)", "-BLUR-", d.blur,
              "Blurs the image before thresholding. Helps with noisy photos. 0 = off", w=28),
        field("Mask edge smooth (px)", "-SMOOTH-", d.smooth,
              "Smooths the edges of the green mask. 0 = off", w=28),
        field("Simplify, straight (px)", "-SIMP-", d.simplify,
              "Tolerance used before straightening right-angle shapes", w=28),
        field("Curve smooth window (px)", "-CSM-", d.curve_smooth,
              "Smooths curved outlines and their holes. 0 = off", w=28),
        field("Curve simplify (px)", "-CSIMP-", f"{d.curve_simplify:g}",
              "Tolerance for curved outlines. 0 = use Simplify, straight", w=28),
    ])

    shape = sg.Frame("Outline cleanup", [
        field("Min step (px)", "-STEP-", d.min_step, "Remove steps shorter than this"),
        field("Grid cell (px)", "-GRID-", d.grid, "Staircase mode. 0 = off"),
        field("DXF units per px", "-SCALE-", d.scale, "Output scale"),
        [sg.Checkbox("Trace holes", key="-HOLES-", default=d.holes)],
    ])

    angle = sg.Frame("Angle", [
        [sg.Radio("Detect per shape and snap", "ANG", key="-AUTO-", default=True,
                  enable_events=True)],
        [sg.Text("      Snap step (deg)", size=(label_w, 1)),
         sg.Input(f"{d.angle_step:g}", key="-ASTEP-", size=(in_w, 1),
                  tooltip="Multiples of this are allowed. 0 = keep detected angle")],
        [sg.Text("      Snap tolerance (deg)", size=(label_w, 1)),
         sg.Input("", key="-ATOL-", size=(in_w, 1),
                  tooltip="Blank = half the step, which always snaps")],
        [sg.Radio("Fixed angle for all shapes", "ANG", key="-FIXED-",
                  enable_events=True)],
        [sg.Text("      Angle (deg)", size=(label_w, 1)),
         sg.Input("0", key="-ANGLE-", size=(in_w, 1), disabled=True,
                  tooltip="Counter-clockwise as seen in the DXF, used exactly")],
    ])

    curves = sg.Frame("Curved shapes", [
        [sg.Checkbox("Detect curves (untick to force 90-degree angles on every shape)", key="-CURVES-",
                     default=not d.no_curve_detection, enable_events=True)],
        field("Min points", "-CPTS-", d.curve_min_points,
              "Minimum simplified contour points for curve detection"),
        field("Angle tolerance (deg)", "-CTOL-", f"{d.curve_angle_tolerance:g}",
              "Allowed deviation from perpendicular edges"),
    ])

    log = sg.Multiline("", key="-LOG-", size=(90, 5), disabled=True, autoscroll=True)

    layout = [
        [files],
        [sg.Column([[colour]], vertical_alignment="top"),
         sg.Column([[target]], vertical_alignment="top"),
         sg.Column([[smoothing]], vertical_alignment="top")],
        [sg.Column([[shape]], vertical_alignment="top"),
         sg.Column([[angle]], vertical_alignment="top"),
         sg.Column([[curves]], vertical_alignment="top")],
        [sg.Button("Preview mask", key="-PREVIEW-"),
         sg.Button("Convert", key="-RUN-"),
         sg.Button("Exit"),
         sg.Text("Scroll the mouse wheel over a number to change it. "
                 "Shift = bigger steps. Up and Down keys also work.")],
        [log],
    ]

    window = sg.Window("MAP-EXTRACTOR", layout, finalize=True)

    int_field = lambda lo, hi: {"kind": "int", "step": 1, "lo": lo, "hi": hi}
    install_scroll(
        window,
        {
            "-HMIN-": int_field(0, 179),
            "-HMAX-": int_field(0, 179),
            "-SMIN-": int_field(0, 255),
            "-VMIN-": int_field(0, 255),
            "-AREA-": {"kind": "float", "step": 10, "lo": 0},
            "-HTOL-": int_field(0, 179),
            "-STOL-": int_field(0, 255),
            "-VTOL-": int_field(0, 255),
            "-BLUR-": {"kind": "odd"},
            "-SMOOTH-": {"kind": "odd"},
            "-SIMP-": {"kind": "float", "step": 0.5, "lo": 0},
            "-CSM-": {"kind": "odd"},
            "-CSIMP-": {"kind": "float", "step": 0.5, "lo": 0},
            "-STEP-": {"kind": "float", "step": 1, "lo": 0},
            "-GRID-": int_field(0, None),
            "-SCALE-": {"kind": "mul", "lo": 1e-6},
            "-ASTEP-": {"kind": "float", "step": 1, "lo": 0, "hi": 90},
            "-ATOL-": {"kind": "float", "step": 0.5, "lo": 0, "hi": 45},
            "-ANGLE-": {"kind": "float", "step": 1, "lo": -360, "hi": 360},
            "-CPTS-": int_field(1, None),
            "-CTOL-": {"kind": "float", "step": 1, "lo": 0, "hi": 45},
        },
        blanks={"-ATOL-": lambda: str(float(window["-ASTEP-"].get() or 15) / 2)},
    )

    def say(text):
        window["-LOG-"].update(text)

    def use_color(hex_text):
        """Put a colour in the hex field, update the swatch and switch colour matching on."""
        r, g_, b = hex_to_rgb(hex_text)
        text = f"#{r:02X}{g_:02X}{b:02X}"
        window["-COLOR-"].update(text)
        window["-SWATCH-"].update(background_color=text)
        window["-USECOLOR-"].update(True)
        sync_fields()

    def sync_fields():
        fixed = bool(window["-FIXED-"].get())
        window["-ANGLE-"].update(disabled=not fixed)
        window["-ASTEP-"].update(disabled=fixed)
        window["-ATOL-"].update(disabled=fixed)
        use = bool(window["-USECOLOR-"].get())
        for key in ("-HMIN-", "-HMAX-", "-SMIN-", "-VMIN-"):
            window[key].update(disabled=use)
        for key in ("-HTOL-", "-STOL-", "-VTOL-"):
            window[key].update(disabled=not use)
        curves_on = bool(window["-CURVES-"].get())
        window["-CPTS-"].update(disabled=not curves_on)
        window["-CTOL-"].update(disabled=not curves_on)
        window["-CSM-"].update(disabled=not curves_on)
        window["-CSIMP-"].update(disabled=not curves_on)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event in ("-AUTO-", "-FIXED-", "-CURVES-", "-USECOLOR-"):
            sync_fields()

        elif event == "-COLOR-":
            try:
                r, g_, b = hex_to_rgb(values["-COLOR-"])
                window["-SWATCH-"].update(background_color=f"#{r:02X}{g_:02X}{b:02X}")
            except ValueError:
                pass  # still typing

        elif event == "-CHOOSE-":
            from tkinter import colorchooser
            try:
                r, g_, b = hex_to_rgb(values["-COLOR-"])
                start = f"#{r:02X}{g_:02X}{b:02X}"
            except ValueError:
                start = None
            _, chosen = colorchooser.askcolor(color=start, parent=window.TKroot,
                                              title="Choose the colour to match")
            if chosen:
                use_color(chosen)

        elif event == "-PICK-":
            if not values["-IN-"].strip():
                say("Error: choose an input image first")
                continue
            chosen = _pick_from_image(sg, values["-IN-"].strip())
            if chosen:
                use_color(chosen)

        elif event == "-IN-":
            path = values["-IN-"].strip()
            if path and not values["-OUT-"].strip():
                window["-OUT-"].update(os.path.splitext(path)[0] + ".dxf")

        elif event == "-PREVIEW-":
            try:
                o = _read_options(dict(values, **{"-OUT-": values["-OUT-"] or "x.dxf"}))
            except ValueError as e:
                say(f"Error: {e}")
                continue
            _preview_window(sg, o)

        elif event == "-RUN-":
            try:
                o = _read_options(values)
            except ValueError as e:
                say(f"Error: {e}")
                continue
            say("Working...")
            window["-RUN-"].update(disabled=True)
            window.perform_long_operation(lambda o=o: _safe_convert(o), "-DONE-")

        elif event == "-DONE-":
            ok, message = values["-DONE-"]
            say(message)
            window["-RUN-"].update(disabled=False)

    window.close()


if __name__ == "__main__":
    main()