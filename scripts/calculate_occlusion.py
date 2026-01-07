import argparse
import os
import re
import csv
from glob import glob
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def parse_args():
    p = argparse.ArgumentParser(description="Multi-view occlusion report using SCENE images as overlay (+ CSV).")
    p.add_argument("--input-dir", type=str, default=".", help="Folder containing object PNGs and the 'scene' subfolder.")
    p.add_argument("--threshold", type=int, default=0, help="RGB threshold [0..255]: pixel is object if any channel > threshold.")
    p.add_argument("--out-dir", type=str, default="scene_occlusion_out", help="Where to write per-viewpoint report figures and CSV.")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI.")
    p.add_argument("--max_cols", type=int, default=6, help="Max columns before wrapping rows.")
    return p.parse_args()


OBJ_RE   = re.compile(r"^theta(\d+)_phi(\d+)_obj(\d+)_color\.png$")
SCENE_RE = re.compile(r"^theta(\d+)_phi(\d+)_scene\.png$")

def find_objects_by_view(input_dir):
    """
    Scan input_dir for object images. Group by (theta, phi).
    Returns dict: {(theta, phi): [(obj_id:int, path:str), ...]}
    """
    out = defaultdict(list)
    for path in glob(os.path.join(input_dir, "*.png")):
        base = os.path.basename(path)
        m = OBJ_RE.match(base)
        if not m:
            continue
        th, ph, obj = m.group(1), m.group(2), int(m.group(3))
        out[(th, ph)].append((obj, path))
    # sort each view's list by obj id (bottom->top)
    for key in out:
        out[key].sort(key=lambda t: t[0])
    return dict(out)

def find_scene_map(input_dir):
    """
    Index scene images by (theta, phi). These are REQUIRED
    because we display them as the overlay.
    """
    scene_dir = os.path.join(input_dir, "scene")
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Required 'scene' subfolder not found in: {input_dir}")
    scenes = {}
    for p in glob(os.path.join(scene_dir, "*.png")):
        m = SCENE_RE.match(os.path.basename(p))
        if m:
            th, ph = m.group(1), m.group(2)
            scenes[(th, ph)] = p
    return scenes

# ----------------------
# Image processing
# ----------------------

def load_rgb_and_mask(path, threshold=0):
    """
    Load an image as RGB (uint8) and a boolean mask where any channel > threshold.
    Assumes black background, colored object.
    """
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    mask = (rgb > threshold).any(axis=2)
    return rgb, mask

def compute_occlusion_masks(masks):
    """
    Given a list of boolean masks in bottom->top order,
    return list occluded_masks where occluded_masks[i] is True where object i is
    occluded by any object above it (j > i).
    """
    H, W = masks[0].shape
    M = len(masks)
    occluded = [np.zeros((H, W), dtype=bool) for _ in range(M)]
    above_cover = np.zeros((H, W), dtype=bool)
    for j in reversed(range(M)):
        m = masks[j]
        occluded[j] = m & above_cover
        above_cover |= m
    return occluded

def representative_color_of_image(rgb: np.ndarray, background_threshold=0):
    """
    Determine a deterministic representative RGB color for a colored object image.
    - rgb: HxWx3 uint8 array
    - background_threshold: any pixel with all channels <= threshold is treated as background.
    Returns an (r,g,b) tuple of ints. If no object pixels found returns (0,0,0).
    """
    flat = rgb.reshape(-1, 3)
    if background_threshold > 0:
        mask = (flat > background_threshold).any(axis=1)
    else:
        # treat pure black as background
        mask = np.any(flat != 0, axis=1)

    pixels = flat[mask]
    if pixels.size == 0:
        return (0, 0, 0)

    # encode colors to single ints then choose the most common (deterministic)
    codes = (pixels[:,0].astype(np.uint32) << 16) | (pixels[:,1].astype(np.uint32) << 8) | pixels[:,2].astype(np.uint32)
    # use numpy unique with return_counts for speed and determinism
    vals, counts = np.unique(codes, return_counts=True)
    mode_code = int(vals[np.argmax(counts)])
    r = (mode_code >> 16) & 0xFF
    g = (mode_code >> 8) & 0xFF
    b = mode_code & 0xFF
    return (int(r), int(g), int(b))


# ----------------------
# Reporting (per-view)
# ----------------------

def make_report_figure(rgb_images, filenames, occlusion_pct, scene_overlay_rgb,
                       out_path, dpi=140, max_cols=6):
    """
    Build a grid figure:
      - panels 1..M: each object's original image (black bg + colored object),
        with a title "filename – Occlusion: YY.YY%"
      - panel M+1: SCENE overlay (ground-truth composite)
    """
    M = len(rgb_images)
    panels = M + 1

    # choose grid (wrap rows if many objects)
    ncols = min(max_cols, panels)
    nrows = (panels + ncols - 1) // ncols

    # Heuristic figure size based on the first image
    H, W, _ = rgb_images[0].shape
    base_inches_w = max(2.5, W / 300.0)
    base_inches_h = max(2.5, H / 300.0)

    fig_w = ncols * base_inches_w
    fig_h = nrows * base_inches_h

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    # Plot each object
    for i, (rgb, fname, occ) in enumerate(zip(rgb_images, filenames, occlusion_pct)):
        ax = axes[i]
        ax.imshow(rgb)
        ax.set_title(f"{os.path.basename(fname)}\nOcclusion: {occ:.2f}%", fontsize=10)
        ax.axis('off')

    # Overlay from scene
    ax_overlay = axes[M]
    ax_overlay.imshow(scene_overlay_rgb)
    ax_overlay.set_title("Scene overlay (ground truth)", fontsize=10)
    ax_overlay.axis('off')

    # Hide any leftover empty axes
    for j in range(M + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def process_view(view_key, obj_list, scene_path, threshold, out_dir, dpi, max_cols):
    """
    Process a single viewpoint:
      - obj_list is [(obj_id, path), ...] in bottom->top order
      - scene_path is the ground-truth scene image to show as overlay
      - returns (per_object_pcts, per_view_average, per_view_std, num_objects, figure_path, per_object_rows, view_row)
    """
    files = [p for _, p in obj_list]

    # Load RGB & masks for objects
    rgb_images, masks = [], []
    for fp in files:
        rgb, m = load_rgb_and_mask(fp, threshold=threshold)
        rgb_images.append(rgb)
        masks.append(m)

    # Consistency check (objects)
    H, W, _ = rgb_images[0].shape
    for i, (rgb, m) in enumerate(zip(rgb_images, masks)):
        if rgb.shape != (H, W, 3) or m.shape != (H, W):
            raise ValueError(f"All images in view {view_key} must be same size. "
                             f"{files[i]} has {rgb.shape}, mask {m.shape}, expected {(H, W, 3)}")

    # Load scene overlay
    scene_img = Image.open(scene_path).convert("RGB")
    scene_rgb = np.asarray(scene_img, dtype=np.uint8)
    if scene_rgb.shape != (H, W, 3):
        raise ValueError(
            f"Scene image size mismatch for view {view_key}. "
            f"Scene {os.path.basename(scene_path)} has {scene_rgb.shape}, expected {(H, W, 3)}"
        )

    # Determine a representative RGB color for each object image (most common non-black)
    obj_colors = []
    for rgb in rgb_images:
        pixels = rgb.reshape(-1, 3)
        mask_nonblack = np.any(pixels != 0, axis=1)
        if not mask_nonblack.any():
            # no colored pixels found (empty object image)
            obj_colors.append(np.array([0, 0, 0], dtype=np.uint8))
            continue
        nonblack_pixels = pixels[mask_nonblack]
        # sample if very large for speed
        if nonblack_pixels.shape[0] > 200000:
            idx = np.random.choice(nonblack_pixels.shape[0], 200000, replace=False)
            sample = nonblack_pixels[idx]
        else:
            sample = nonblack_pixels
        colors, counts = np.unique(sample.reshape(-1, 3), axis=0, return_counts=True)
        mode_color = colors[np.argmax(counts)].astype(np.uint8)
        obj_colors.append(mode_color)

    # Optional color tolerance for anti-aliased edges; set tol=0 for exact match
    tol = 0

    scene_rgb_int = scene_rgb.astype(np.int32)
    occlusion_pct = []
    for m_bool, color in zip(masks, obj_colors):
        total = int(m_bool.sum())
        if total == 0:
            occlusion_pct.append(0.0)
            continue

        if tol == 0:
            scene_matches = np.all(scene_rgb == color, axis=2)
        else:
            diff = scene_rgb_int - color.astype(np.int32)
            dist = np.sqrt((diff ** 2).sum(axis=2))
            scene_matches = dist <= tol

        # Occluded: pixels where object would be present (solo mask) but scene does NOT show its color
        occluded_mask = m_bool & (~scene_matches)
        occluded = int(occluded_mask.sum())
        pct = (occluded / total * 100.0)
        occlusion_pct.append(pct)

    # Sanity warning: if an object's color never appears in the scene overlay, print a warning
    for i, color in enumerate(obj_colors):
        if not np.any(np.all(scene_rgb == color, axis=2)):
            print(f"[warn] object file {os.path.basename(files[i])}: color {color.tolist()} not present in scene overlay — check naming/UID consistency.")

    # compute per-view std
    view_avg = float(np.mean(occlusion_pct)) if occlusion_pct else 0.0
    view_std = float(np.std(occlusion_pct)) if occlusion_pct else 0.0

    # Output figure path
    th, ph = view_key
    out_name = f"theta{th}_phi{ph}_occlusion.png"
    fig_path = os.path.join(out_dir, out_name)

    # Report figure (overlay = SCENE)
    make_report_figure(
        rgb_images=rgb_images,
        filenames=files,
        occlusion_pct=occlusion_pct,
        scene_overlay_rgb=scene_rgb,
        out_path=fig_path,
        dpi=dpi,
        max_cols=max_cols
    )

    # Per-view summary
    print(f"[view theta{th}_phi{ph}] Saved report: {fig_path}")
    per_object_rows = []
    for i, (fp, pct) in enumerate(zip(files, occlusion_pct), start=1):
        base = os.path.basename(fp)
        m = OBJ_RE.match(base)
        obj_id = int(m.group(3)) if m else None
        print(f"  [{i:02d}] {base}  occlusion={pct:6.2f}%")
        per_object_rows.append({
            "level": "object",
            "theta": int(th),
            "phi": int(ph),
            "obj_id": obj_id,
            "filename": base,
            "occlusion_pct": round(pct, 6),
            "view_avg_pct": round(view_avg, 6),
            "view_std_pct": round(view_std, 6),
            "figure_path": fig_path
        })
    print(f"  View average occlusion: {view_avg:.2f}%  std: {view_std:.2f}%\n")

    view_row = {
        "level": "view",
        "theta": int(th),
        "phi": int(ph),
        "obj_id": "",
        "filename": "",
        "occlusion_pct": "",
        "view_avg_pct": round(view_avg, 6),
        "view_std_pct": round(view_std, 6),
        "figure_path": fig_path
    }

    return occlusion_pct, view_avg, view_std, len(occlusion_pct), fig_path, per_object_rows, view_row



def main():
    args = parse_args()
    input_dir = args.input_dir
    out_dir = args.out_dir
    threshold = args.threshold

    # Find objects grouped by viewpoint
    objects_by_view = find_objects_by_view(input_dir)
    if not objects_by_view:
        raise FileNotFoundError(f"No object files found in {input_dir} matching pattern theta*_phi*_obj*_color.png")

    # Scenes are REQUIRED now (we use them for the overlay)
    scenes = find_scene_map(input_dir)
    # Check that every viewpoint with objects has a scene
    missing = [k for k in objects_by_view.keys() if k not in scenes]
    if missing:
        missing_str = ", ".join([f"theta{t}_phi{p}" for (t, p) in sorted(missing, key=lambda x: (int(x[0]), int(x[1])) )])
        raise FileNotFoundError(f"Missing required scene image(s) for viewpoint(s): {missing_str}")

    os.makedirs(out_dir, exist_ok=True)

    print("Computing representative colors for each object image...")
    per_object_color_rows = []  # list of dicts to write later
    # objects_by_view: keys are (th, ph) as strings, values are lists [(obj_id, path), ...]
    for (th, ph), obj_list in sorted(objects_by_view.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
        for obj_id, path in obj_list:
            base = os.path.basename(path)
            try:
                img = Image.open(path).convert("RGB")
                rgb_arr = np.asarray(img, dtype=np.uint8)
                r, g, b = representative_color_of_image(rgb_arr, background_threshold=threshold)
            except Exception as e:
                print(f"[warn] Failed to load/parse image for color extraction '{path}': {e}")
                r, g, b = (0, 0, 0)
            per_object_color_rows.append({
                "theta": int(th),
                "phi": int(ph),
                "obj_id": int(obj_id),
                "filename": base,
                "r": int(r),
                "g": int(g),
                "b": int(b)
            })

    # Write per-object colors CSV
    per_object_colors_path = os.path.join(out_dir, "per_object_colors.csv")
    try:
        with open(per_object_colors_path, "w", newline="") as cf:
            fieldnames = ["theta", "phi", "obj_id", "filename", "r", "g", "b"]
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_object_color_rows:
                writer.writerow(row)
        print(f"Wrote per-object color mapping to: {per_object_colors_path}")
    except Exception as e:
        print(f"[error] failed to write per-object colors CSV {per_object_colors_path}: {e}")

    csv_path = os.path.join(out_dir, "occlusion_summary.csv")
    per_object_csv_path = os.path.join(out_dir, "per_object_occlusion.csv")

    # Accumulators for overall stats
    all_pcts = []
    total_objs = 0

    # Collect per-view averages and stds for writing the simplified CSV and plotting
    view_summaries = []  # list of (view_name:str, view_avg:float, view_std:float)

    # *** ADDED: collect all per-object rows across all views
    all_per_object_rows = []

    # Deterministic order over viewpoints (sort by theta, then phi as ints)
    def view_sort_key(k):
        th, ph = k
        return (int(th), int(ph))

    for view_key in sorted(objects_by_view.keys(), key=view_sort_key):
        obj_list = objects_by_view[view_key]  # [(obj_id, path), ...] bottom->top
        scene_path = scenes[view_key]
        pcts, view_avg, view_std, n, fig_path, per_object_rows, view_row = process_view(
            view_key, obj_list, scene_path, threshold, out_dir, args.dpi, args.max_cols
        )

        # Record view summary
        th, ph = view_key
        view_name = f"theta{th}_phi{ph}"
        view_summaries.append((view_name, view_avg, view_std))

        # *** ADDED: extend global per-object rows
        all_per_object_rows.extend(per_object_rows)

        all_pcts.extend(pcts)
        total_objs += n

    overall_avg = float(np.mean(all_pcts)) if all_pcts else 0.0
    overall_std = float(np.std(all_pcts)) if all_pcts else 0.0

    print("=" * 60)
    print(f"Processed {len(objects_by_view)} viewpoints, {total_objs} objects total.")
    print(f"Average occlusion across ALL objects from ALL viewpoints: {overall_avg:.2f}%")
    print(f"Standard deviation of occlusion across ALL objects: {overall_std:.2f}%")
    print("=" * 60)

    # Write simplified CSV: one row per viewpoint (viewpoint, occlusion), final row full_scene
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["viewpoint", "occlusion"])
        for view_name, view_avg, _ in view_summaries:
            writer.writerow([view_name, f"{view_avg:.6f}"])
        # final overall row
        writer.writerow(["full_scene", f"{overall_avg:.6f}"])

    print(f"CSV written to: {csv_path}")

    # Use fieldnames matching the dict keys produced in per_object_rows
    if all_per_object_rows:
        fieldnames = ["level", "theta", "phi", "obj_id", "filename",
                      "occlusion_pct", "view_avg_pct", "view_std_pct"]
        try:
            with open(per_object_csv_path, "w", newline="") as outf:
                writer = csv.DictWriter(outf, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_per_object_rows:
                    # Ensure we only write the fields in fieldnames order (and keep types simple)
                    out_row = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(out_row)
            print(f"Per-object CSV written to: {per_object_csv_path}")
        except Exception as e:
            print(f"[error] failed to write per-object CSV {per_object_csv_path}: {e}")
    else:
        print("[info] No per-object rows to write to per_object_occlusion.csv")

if __name__ == "__main__":
    main()
