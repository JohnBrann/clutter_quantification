
# # python3 ./distance_calculations_no_border.py   --input-dir ../pybullet/scene01/scene/   --out-dir scene01_out2d   --occlusion-csv ../pybullet/scene01_out/per_object_occlusion.csv   --colors-csv ../pybullet/scene01_out/per_object_colors.csv   --occlusion-threshold 50.0

#!/usr/bin/env python3
"""
Batch 2D distance metric (perimeter-corrected) with occlusion-based filtering.

This script processes ALL segmented scene images in an input folder and writes
per-image outputs (visualization PNG, per-point CSV, metrics CSV) into an
output directory. Optionally, objects that are heavily occluded (per a provided
occlusion CSV) are removed from consideration by color before computing
connections.

New: excluded objects are blacked-out in the visualization PNG so you can see
which objects were removed while the distance computation uses only the
remaining objects. Also writes a single summary CSV with connection counts.

Usage example:
  python batch_distances_with_occlusion_filter.py \
    --input-dir scenes \
    --out-dir distances_out \
    --occlusion-csv /path/to/per_object_occlusion.csv \
    --colors-csv /path/to/per_object_colors.csv \
    --occlusion-threshold 50.0
"""

import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import os
from glob import glob
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
import re
import math

# Optional robust contour extractor
try:
    from skimage import measure as skmeasure  # type: ignore
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


@dataclass
class Segment:
    color: Tuple[int, int, int]
    name: str
    mask: np.ndarray
    boundary_coords: np.ndarray  # Nx2 array of (y, x) coords (int32)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ---------- contour / sampling helpers (unchanged) ----------

def order_points_nearest_chain(pts: np.ndarray) -> np.ndarray:
    if pts is None or len(pts) == 0:
        return pts.astype(np.float64)

    ptsf = pts.astype(np.float64)
    N = len(ptsf)
    if N == 1:
        return ptsf.copy()

    tree = cKDTree(ptsf[:, ::-1])  # query with (x,y)
    visited = np.zeros(N, dtype=bool)
    order = [0]
    visited[0] = True

    for _ in range(1, N):
        last = order[-1]
        k = min(20, N)
        dists, idxs = tree.query(ptsf[last, ::-1], k=k)
        idxs_arr = np.atleast_1d(idxs)
        found = False
        for idx in idxs_arr:
            if not visited[int(idx)]:
                order.append(int(idx))
                visited[int(idx)] = True
                found = True
                break
        if not found:
            unvisited_idx = np.nonzero(~visited)[0]
            if unvisited_idx.size == 0:
                break
            unvisited_pts = ptsf[unvisited_idx]
            diffs = unvisited_pts - ptsf[last]
            d2 = np.sum(diffs ** 2, axis=1)
            a = int(unvisited_idx[int(np.argmin(d2))])
            order.append(a)
            visited[a] = True

    ordered = ptsf[order]
    return ordered


def sample_points_along_perimeter(coords: np.ndarray, spacing_px: float) -> np.ndarray:
    if coords is None or len(coords) == 0:
        return np.empty((0, 2), dtype=np.int32)

    pts = coords.astype(np.float64)

    if spacing_px <= 0:
        return np.unique(np.round(pts).astype(np.int32), axis=0)

    try:
        ordered = order_points_nearest_chain(pts)
        if ordered.shape[0] < 3:
            raise Exception("too few points for NN ordering")
    except Exception:
        centroid = pts.mean(axis=0)
        angs = np.arctan2(pts[:, 0] - centroid[0], pts[:, 1] - centroid[1])
        order = np.argsort(angs)
        ordered = pts[order]

    if not np.allclose(ordered[0], ordered[-1]):
        closed = np.vstack([ordered, ordered[0]])
    else:
        closed = ordered.copy()

    seg_dists = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
    cumdist = np.concatenate(([0.0], np.cumsum(seg_dists)))
    total_perim = cumdist[-1]

    if total_perim <= 0:
        pts_int = np.round(ordered).astype(np.int32)
        uniq = np.unique(pts_int, axis=0)
        return uniq

    num_samples = max(1, int(np.floor(total_perim / spacing_px)) + 1)
    sample_ds = np.linspace(0.0, total_perim, num=num_samples)

    xs = closed[:, 1]
    ys = closed[:, 0]

    samp_x = np.interp(sample_ds, cumdist, xs)
    samp_y = np.interp(sample_ds, cumdist, ys)

    samp_pts = np.stack([samp_y, samp_x], axis=1)
    samp_pts_int = np.round(samp_pts).astype(np.int32)
    if samp_pts_int.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    uniq = np.unique(samp_pts_int, axis=0)
    return uniq


def extract_ordered_contour_from_mask(mask: np.ndarray) -> np.ndarray:
    if HAVE_SKIMAGE:
        try:
            contours = skmeasure.find_contours(mask.astype(np.uint8), level=0.5)
            if contours:
                best = max(contours, key=lambda c: c.shape[0])
                pts = np.round(best).astype(np.int32)
                uniq = np.unique(pts, axis=0)
                return uniq
        except Exception:
            pass

    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary = mask & (~eroded)
    ys, xs = np.nonzero(boundary)
    if len(ys) == 0:
        ys, xs = np.nonzero(mask)
    pts = np.stack([ys, xs], axis=1).astype(np.int32)
    pts_unique = np.unique(pts, axis=0)
    return pts_unique


def find_segments(img: np.ndarray, min_pixels: int = 5, spacing_px: float = 8.0) -> List[Segment]:
    H, W, C = img.shape
    assert C == 3, "Expected RGB image"

    flat = img.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)

    # Determine background by most-common color
    color_counts = {tuple(map(int, c)): int(n) for c, n in zip(colors, counts)}
    background_rgb = max(color_counts.items(), key=lambda x: x[1])[0]

    segments: List[Segment] = []
    for color in colors:
        color_tuple = tuple(map(int, color))
        if color_tuple == background_rgb:
            continue

        mask = np.all(img == color, axis=2)
        if mask.sum() < min_pixels:
            continue

        contour_pts = extract_ordered_contour_from_mask(mask)
        if contour_pts is None or len(contour_pts) == 0:
            continue

        sampled_coords = sample_points_along_perimeter(contour_pts, spacing_px)
        segments.append(Segment(
            color=color_tuple,
            name=rgb_to_hex(color_tuple),
            mask=mask,
            boundary_coords=sampled_coords
        ))

    # stable sort by name to keep deterministic ordering
    segments.sort(key=lambda s: s.name)
    return segments


def find_border_segment(img: np.ndarray, min_pixels: int = 10, spacing_px: float = 8.0) -> Optional[Segment]:
    white_rgb = (255, 255, 255)
    H, W, _ = img.shape
    mask = np.all(img == np.array(white_rgb, dtype=img.dtype), axis=2)

    if mask.sum() < min_pixels:
        return None

    contour_pts = extract_ordered_contour_from_mask(mask)
    sampled_coords = sample_points_along_perimeter(contour_pts, spacing_px)

    return Segment(
        color=white_rgb,
        name="border",
        mask=mask,
        boundary_coords=sampled_coords
    )


def compute_all_connections_every_point(
    segments: List[Segment],
    border: Optional[Segment]
) -> List[Tuple[int, int, float, Tuple[int, int], Tuple[int, int]]]:
    results: List[Tuple[int, int, float, Tuple[int, int], Tuple[int, int]]] = []

    border_pts = np.array(border.boundary_coords) if (border is not None) else np.empty((0, 2), dtype=np.int32)
    border_tree = None
    if border is not None and len(border_pts) > 0:
        border_tree = cKDTree(border_pts[:, ::-1])  # use x,y for KD-tree

    all_obj_pts = []
    all_obj_owner = []
    for j, seg in enumerate(segments):
        for pt in seg.boundary_coords:
            all_obj_pts.append(pt)
            all_obj_owner.append(j)
    all_obj_pts = np.array(all_obj_pts) if len(all_obj_pts) > 0 else np.empty((0, 2), dtype=np.int32)

    global_tree = None
    if len(all_obj_pts) > 0:
        global_tree = cKDTree(all_obj_pts[:, ::-1])

    for i, src_seg in enumerate(segments):
        src_pts = np.array(src_seg.boundary_coords)
        if src_pts.size == 0:
            continue

        for src_pt in src_pts:
            src_y, src_x = int(src_pt[0]), int(src_pt[1])

            if border_tree is not None:
                dist_b, idx_b = border_tree.query([src_x, src_y], k=1)
                tgt_b = tuple(int(x) for x in border_pts[int(idx_b)])
                results.append((i, -1, float(dist_b), (src_y, src_x), (tgt_b[0], tgt_b[1])))

            if global_tree is not None and len(all_obj_pts) > 0:
                Kprobe = min(10, max(1, len(all_obj_pts)))
                dists, idxs = global_tree.query([src_x, src_y], k=Kprobe)
                if Kprobe == 1:
                    dists = np.array([dists])
                    idxs = np.array([idxs])

                for dist_o, idx_o in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                    owner = all_obj_owner[int(idx_o)]
                    if owner != i:
                        tgt_pt = all_obj_pts[int(idx_o)]
                        results.append((i, owner, float(dist_o), (src_y, src_x), (int(tgt_pt[0]), int(tgt_pt[1]))))
                        break

    return results


def draw_visualization(
    img: np.ndarray,
    segments: List[Segment],
    border: Optional[Segment],
    connections: List[Tuple[int, int, float, Tuple[int, int], Tuple[int, int]]],
    gripper_width: float,
    out_path: str
):
    H, W, _ = img.shape
    fig, ax = plt.subplots(figsize=(max(6, W / 50), max(6, H / 50)), dpi=150)
    ax.imshow(img)

    for seg in segments:
        pts = seg.boundary_coords
        if pts is None or len(pts) == 0:
            continue
        ys = pts[:, 0]
        xs = pts[:, 1]
        ax.scatter(xs, ys, s=8, color='blue', alpha=0.9, zorder=6, edgecolors='white', linewidths=0.3)

    border_targets = []
    object_targets = []

    for src_idx, tgt_idx, dist, (src_y, src_x), (tgt_y, tgt_x) in connections:
        color_line = "red" if dist < gripper_width else "lime"
        ax.plot([src_x, tgt_x], [src_y, tgt_y], linewidth=0.8, alpha=0.85, color=color_line, zorder=5)
        if tgt_idx == -1:
            border_targets.append((tgt_x, tgt_y))
        else:
            object_targets.append((tgt_x, tgt_y))

    if border is not None and border_targets:
        bt = np.array(border_targets, dtype=np.int32)
        bt_unique = np.unique(bt, axis=0)
        ax.scatter(bt_unique[:, 0], bt_unique[:, 1], s=18, marker='s', color='magenta', zorder=7, edgecolors='white', linewidths=0.4)

    if object_targets:
        ot = np.array(object_targets, dtype=np.int32)
        ot_unique = np.unique(ot, axis=0)
        ax.scatter(ot_unique[:, 0], ot_unique[:, 1], s=14, marker='o', color='orange', zorder=7, edgecolors='white', linewidths=0.4)

    border_note = "Magenta squares: border targets (nearest)\n" if border is not None else ""
    ax.text(
        10, 20,
        f"Gripper width: {gripper_width:.1f} px\n"
        f"Blue: sampled source points\n"
        f"{border_note}"
        f"Orange circles: object targets (nearest)\n"
        f"Red lines: blocked (<gripper)\n"
        f"Green lines: clear (>=gripper)",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="none", alpha=0.7)
    )

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


def compute_metrics(connections: List[Tuple], gripper_width: float) -> dict:
    distances = [d for _, _, d, _, _ in connections]

    if len(distances) == 0:
        return {
            "num_connections": 0,
            "blocked_connections": 0,
            "fraction_blocked": 0.0,
            "min_gap_px": float('nan'),
            "mean_gap_px": float('nan'),
            "median_gap_px": float('nan')
        }

    distances = np.array(distances)
    blocked = distances < gripper_width

    return {
        "num_connections": int(len(distances)),
        "blocked_connections": int(np.sum(blocked)),
        "fraction_blocked": float(np.mean(blocked)),
        "min_gap_px": float(np.min(distances)),
        "mean_gap_px": float(np.mean(distances)),
        "median_gap_px": float(np.median(distances))
    }


# ---------- helpers for occlusion/color CSV processing ----------

SCENE_VIEW_RE = re.compile(r"theta(\d+)_phi(\d+)", flags=re.IGNORECASE)


def parse_view_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    m = SCENE_VIEW_RE.search(os.path.basename(filename))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def load_colors_csv(colors_csv_path: str) -> Dict[Tuple[int, int, str], Tuple[int, int, int]]:
    """
    Load per_object_colors.csv that has fields: theta,phi,obj_id,filename,r,g,b
    Returns a dict keyed by (theta,phi,filename) -> (r,g,b).
    """
    mapping: Dict[Tuple[int, int, str], Tuple[int, int, int]] = {}
    if not colors_csv_path or not os.path.isfile(colors_csv_path):
        return mapping
    with open(colors_csv_path, newline="") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            try:
                th = int(row.get("theta", row.get("theta", "")))
                ph = int(row.get("phi", row.get("phi", "")))
                fname = row.get("filename") or row.get("file") or ""
                r = int(row.get("r", 0))
                g = int(row.get("g", 0))
                b = int(row.get("b", 0))
                mapping[(th, ph, fname)] = (r, g, b)
                # also add fallback mapping keyed only by filename (if not present)
                if (0, 0, fname) not in mapping:
                    mapping[(0, 0, fname)] = (r, g, b)
            except Exception:
                continue
    return mapping


def load_occlusion_csv(occl_csv_path: str, threshold: float) -> Dict[Tuple[int, int], Set[str]]:
    """
    Load per_object_occlusion.csv that has fields: level,theta,phi,obj_id,filename,occlusion_pct,...
    Returns dict mapping (theta,phi) -> set(filenames) where occlusion_pct >= threshold.
    """
    excluded_by_view: Dict[Tuple[int, int], Set[str]] = {}
    if not occl_csv_path or not os.path.isfile(occl_csv_path):
        return excluded_by_view
    with open(occl_csv_path, newline="") as of:
        reader = csv.DictReader(of)
        for row in reader:
            try:
                level = row.get("level", "")
                if level.strip().lower() != "object":
                    continue
                th = int(row.get("theta", 0))
                ph = int(row.get("phi", 0))
                fname = row.get("filename", "")
                pct = float(row.get("occlusion_pct", 0.0))
                if pct >= threshold:
                    key = (th, ph)
                    excluded_by_view.setdefault(key, set()).add(fname)
            except Exception:
                continue
    return excluded_by_view


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return math.sqrt((int(c1[0]) - int(c2[0])) ** 2 + (int(c1[1]) - int(c2[1])) ** 2 + (int(c1[2]) - int(c2[2])) ** 2)


# ---------- main processing / I/O ----------

def safe_basename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def process_single_image(img_path: str, out_dir: str, args: argparse.Namespace,
                         excluded_colors_for_view: Optional[Set[Tuple[int, int, int]]] = None,
                         color_tol: int = 0):
    """
    Process one segmented scene image. Returns a summary dict with counts:
      {
        "scene": basename,
        "num_segments_before": int,
        "num_segments_after": int,
        "num_excluded_segments": int,
        "total_connections": int,
        "blocked_connections": int,
        "fraction_blocked": float
      }
    """
    print(f"\nProcessing: {img_path}")
    img = iio.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = img.astype(np.uint8)

    # Find segments from original image
    segments_all = find_segments(img, spacing_px=args.spacing)
    num_before = len(segments_all)
    print(f"  Found {num_before} objects before filtering: {[s.name for s in segments_all]}")

    # Compute explicit white border from the original image (unchanged by filtering)
    border = find_border_segment(img, spacing_px=args.spacing)
    if border is None:
        if args.verbose:
            print("  No explicit white border detected. Border connections will be DISABLED.")
    else:
        if args.verbose:
            print(f"  White border detected; sampled {len(border.boundary_coords)} border points")

    # Apply occlusion-based color filtering (if provided)
    filtered_out = []
    segments = segments_all
    if excluded_colors_for_view:
        keep_segments = []
        for seg in segments_all:
            matched = False
            for ex_col in excluded_colors_for_view:
                if color_tol == 0:
                    if tuple(seg.color) == tuple(ex_col):
                        matched = True
                        break
                else:
                    if color_distance(tuple(seg.color), tuple(ex_col)) <= color_tol:
                        matched = True
                        break
            if matched:
                filtered_out.append((seg, seg.color))
            else:
                keep_segments.append(seg)
        segments = keep_segments
        print(f"  Filtered out {len(filtered_out)} segments by occlusion-color (remaining {len(segments)}).")
        if len(filtered_out) > 0 and args.verbose:
            print("   Filtered colors:", [c for (_, c) in filtered_out])

    num_after = len(segments)
    num_excluded = num_before - num_after

    if len(segments) == 0:
        print("  No objects remain after filtering — skipping outputs.")
        # still write a masked image (all-black) for completeness
        basename = safe_basename_no_ext(img_path)
        masked_img = img.copy()
        for seg in segments_all:
            try:
                masked_img[seg.mask] = np.array([0, 0, 0], dtype=np.uint8)
            except Exception:
                ys, xs = np.nonzero(seg.mask)
                masked_img[ys, xs] = np.array([0, 0, 0], dtype=np.uint8)
        masked_out_path = os.path.join(out_dir, f"{basename}_masked.png")
        try:
            iio.imwrite(masked_out_path, masked_img)
        except Exception:
            pass

        return {
            "scene": basename,
            "num_segments_before": num_before,
            "num_segments_after": num_after,
            "num_excluded_segments": num_excluded,
            "total_connections": 0,
            "blocked_connections": 0,
            "fraction_blocked": 0.0
        }

    # Create a masked image that blacks-out excluded segments (for visualization)
    masked_img = img.copy()
    if filtered_out:
        for seg, _ in filtered_out:
            # seg.mask is boolean mask; set those pixels to black
            try:
                masked_img[seg.mask] = np.array([0, 0, 0], dtype=np.uint8)
            except Exception:
                ys, xs = np.nonzero(seg.mask)
                masked_img[ys, xs] = np.array([0, 0, 0], dtype=np.uint8)

    # Save masked raw image (no overlay) so user can inspect raw blacked-out input
    basename = safe_basename_no_ext(img_path)
    masked_out_path = os.path.join(out_dir, f"{basename}_masked.png")
    try:
        iio.imwrite(masked_out_path, masked_img)
        if args.verbose:
            print(f"  Saved masked raw image (excluded objects blacked out): {masked_out_path}")
    except Exception as e:
        print(f"[warn] failed to write masked image {masked_out_path}: {e}")

    # Compute connections on *remaining* segments (border computed earlier from original image)
    connections = compute_all_connections_every_point(segments, border)
    total_connections = len(connections)
    blocked_connections = int(sum(1 for (_, _, d, _, _) in connections if d < args.gripper_width))
    fraction_blocked = (blocked_connections / total_connections) if total_connections > 0 else 0.0
    print(f"  Total connections found: {total_connections} (blocked: {blocked_connections})")

    viz_out = os.path.join(out_dir, f"{basename}_distances_viz.png")
    csv_out = os.path.join(out_dir, f"{basename}_distances.csv")
    metrics_out = os.path.join(out_dir, f"{basename}_metrics.csv")

    # Write per-point CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_obj", "source_idx", "target_obj", "target_idx",
            "distance_px", "src_x", "src_y", "tgt_x", "tgt_y",
            "is_blocked", "clearance_px"
        ])
        for src_idx, tgt_idx, dist, (src_y, src_x), (tgt_y, tgt_x) in connections:
            src_name = segments[src_idx].name
            if tgt_idx == -1:
                tgt_name = "border"
                tgt_idx_out = -1
            else:
                tgt_name = segments[tgt_idx].name
                tgt_idx_out = int(tgt_idx)

            is_blocked = 1 if dist < args.gripper_width else 0
            clearance = dist - args.gripper_width
            writer.writerow([
                src_name, int(src_idx), tgt_name, tgt_idx_out,
                f"{dist:.4f}", int(src_x), int(src_y), int(tgt_x), int(tgt_y),
                is_blocked, f"{clearance:.4f}"
            ])
    if args.verbose:
        print(f"  Saved per-point distances to: {csv_out}")

    # Write metrics CSV (use compute_metrics for consistency)
    metrics = compute_metrics(connections, args.gripper_width)
    with open(metrics_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
    if args.verbose:
        print(f"  Saved metrics to: {metrics_out}")
        print(f"  Metrics: {metrics}")

    # Create visualization PNG drawn on the masked image so excluded objects appear blacked out.
    # The draw_visualization function draws sampled points and lines only for 'segments' (remaining).
    draw_visualization(masked_img, segments, border, connections, args.gripper_width, viz_out)
    if args.verbose:
        print(f"  Saved visualization to: {viz_out}")

    return {
        "scene": basename,
        "num_segments_before": num_before,
        "num_segments_after": num_after,
        "num_excluded_segments": num_excluded,
        "total_connections": total_connections,
        "blocked_connections": blocked_connections,
        "fraction_blocked": fraction_blocked
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch compute 2D perimeter-corrected distances for segmented images in a folder."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Folder containing segmented images (PNG).")
    parser.add_argument("--glob", default="*.png",
                        help="Glob pattern to select images in input-dir (default '*.png').")
    parser.add_argument("--out-dir", default="distances_out",
                        help="Directory to write per-image outputs (created if missing).")
    parser.add_argument("--gripper-width", type=float, default=32.0,
                        help="Gripper width in pixels (threshold for blocked/clear).")
    parser.add_argument("--spacing", type=float, default=16.0,
                        help="Distance between sampled boundary points in pixels (default: 16.0).")
    parser.add_argument("--min-pixels", type=int, default=5,
                        help="Minimum pixels for an object segment to be considered (default: 5).")
    parser.add_argument("--skip-patterns", default="_distances,_metrics,_masked",
                        help="Comma-separated substrings; files containing any will be skipped (default: '_distances,_metrics,_masked').")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    # occlusion/color filtering args
    parser.add_argument("--occlusion-csv", default=None,
                        help="Path to per-object occlusion CSV (level,theta,phi,obj_id,filename,occlusion_pct,...). Optional.")
    parser.add_argument("--colors-csv", default=None,
                        help="Path to per-object colors CSV (theta,phi,obj_id,filename,r,g,b). Optional.")
    parser.add_argument("--occlusion-threshold", type=float, default=50.0,
                        help="Occlusion threshold pct: objects with occlusion_pct >= this are excluded (default 50.0).")
    parser.add_argument("--occlusion-color-tol", type=int, default=0,
                        help="Color distance tolerance (Euclidean) used when comparing segment color to excluded colors (default 0 = exact match).")
    args = parser.parse_args()

    input_dir = args.input_dir
    pattern = args.glob
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load color mapping (if provided)
    color_map = load_colors_csv(args.colors_csv)  # keyed by (theta,phi,filename) and fallback (0,0,filename)
    if args.colors_csv:
        print(f"Loaded colors CSV entries: {len(color_map)} (including filename fallbacks)")

    # Load occlusion CSV and build set of filenames to exclude per view
    excluded_by_view = load_occlusion_csv(args.occlusion_csv, args.occlusion_threshold)
    if args.occlusion_csv:
        print(f"Loaded occlusion CSV; views with exclusions: {len(excluded_by_view)}")

    # Build mapping of excluded colors per (theta,phi)
    excluded_colors_map: Dict[Tuple[int, int], Set[Tuple[int, int, int]]] = {}
    for (th, ph), fnames in excluded_by_view.items():
        ex_colors: Set[Tuple[int, int, int]] = set()
        for fname in fnames:
            # try exact keyed mapping first
            if (th, ph, fname) in color_map:
                ex_colors.add(color_map[(th, ph, fname)])
            elif (0, 0, fname) in color_map:
                ex_colors.add(color_map[(0, 0, fname)])
            else:
                # attempt to search any key with that filename
                found = False
                for (kth, kph, kfname), rgb in color_map.items():
                    if kfname == fname:
                        ex_colors.add(rgb)
                        found = True
                        break
                if not found:
                    print(f"[warn] Could not find color for excluded filename '{fname}' (theta{th}_phi{ph}) in colors CSV; skipping exclusion for that file.")
        if ex_colors:
            excluded_colors_map[(th, ph)] = ex_colors

    if excluded_colors_map:
        total_ex_colors = sum(len(s) for s in excluded_colors_map.values())
        print(f"Built excluded-colors map for {len(excluded_colors_map)} views (total colors: {total_ex_colors}).")
    else:
        if args.occlusion_csv or args.colors_csv:
            print("No excluded colors found (check occlusion/colors CSV paths and thresholds).")

    # Build list of files to process
    search_path = os.path.join(input_dir, pattern)
    files = sorted(glob(search_path))
    skip_subs = [s.strip() for s in args.skip_patterns.split(",") if s.strip()]

    if not files:
        raise FileNotFoundError(f"No files found matching: {search_path}")

    print(f"Found {len(files)} image files to consider in: {input_dir}")
    processed = 0

    # accumulator for per-scene summary rows
    summary_rows = []

    for fpath in files:
        base = os.path.basename(fpath)
        # Skip files that look like outputs (contain skip substrings)
        if any(sub in base for sub in skip_subs):
            if args.verbose:
                print(f"Skipping (matches skip pattern): {base}")
            continue

        # Determine view key for this scene by parsing its filename
        view = parse_view_from_filename(base)
        excluded_colors_for_view = None
        if view is not None and view in excluded_colors_map:
            excluded_colors_for_view = excluded_colors_map[view]
            if args.verbose:
                print(f"Applying {len(excluded_colors_for_view)} excluded colors for view theta{view[0]}_phi{view[1]}")
        else:
            if args.verbose and (args.occlusion_csv or args.colors_csv):
                print(f"No exclusions for scene {base} (view {view})")

        try:
            info = process_single_image(fpath, out_dir, args, excluded_colors_for_view=excluded_colors_for_view, color_tol=args.occlusion_color_tol)
            processed += 1
            if info:
                summary_rows.append(info)
        except Exception as e:
            print(f"[error] Failed processing {fpath}: {e}")

    # Write consolidated summary CSV
    summary_csv_path = os.path.join(out_dir, "connections_summary.csv")
    if summary_rows:
        fieldnames = ["scene", "num_segments_before", "num_segments_after", "num_excluded_segments",
                      "total_connections", "blocked_connections", "fraction_blocked"]
        try:
            with open(summary_csv_path, "w", newline="") as sf:
                writer = csv.DictWriter(sf, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary_rows:
                    # ensure fraction_blocked is formatted
                    out_row = {k: row.get(k, "") for k in fieldnames}
                    if out_row.get("fraction_blocked") is not None:
                        out_row["fraction_blocked"] = f"{float(out_row['fraction_blocked']):.6f}"
                    writer.writerow(out_row)
            print(f"\nWrote consolidated connections summary to: {summary_csv_path}")
        except Exception as e:
            print(f"[error] Failed to write summary CSV {summary_csv_path}: {e}")
    else:
        print("\nNo per-scene summary rows to write.")

    print("\nBatch complete.")
    print(f"Processed {processed} images. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()

