import argparse
import os
import re
import csv
from glob import glob
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Multi-view occlusion report using SCENE images as overlay (+ CSV).")
    p.add_argument("--input-dir", type=str, default=".", help="Folder containing object PNGs and the 'scene' subfolder.")
    p.add_argument("--threshold", type=int, default=0, help="RGB threshold [0..255]: pixel is object if any channel > threshold.")
    p.add_argument("--out-dir", type=str, default="scene_occlusion_out", help="Where to write per-viewpoint report figures and CSV.")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI.")
    p.add_argument("--max-cols", type=int, default=6, help="Max columns before wrapping rows.")
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

# ----------------------
# Main (multi-view)
# ----------------------

# def process_view(view_key, obj_list, scene_path, threshold, out_dir, dpi, max_cols):
#     """
#     Process a single viewpoint:
#       - obj_list is [(obj_id, path), ...] in bottom->top order
#       - scene_path is the ground-truth scene image to show as overlay
#       - returns (per_object_pcts, per_view_average, per_view_std, num_objects, figure_path, per_object_rows, view_row)
#     """
#     files = [p for _, p in obj_list]

#     # Load RGB & masks for objects
#     rgb_images, masks = [], []
#     for fp in files:
#         rgb, m = load_rgb_and_mask(fp, threshold=threshold)
#         rgb_images.append(rgb)
#         masks.append(m)

#     # Consistency check (objects)
#     H, W, _ = rgb_images[0].shape
#     for i, (rgb, m) in enumerate(zip(rgb_images, masks)):
#         if rgb.shape != (H, W, 3) or m.shape != (H, W):
#             raise ValueError(f"All images in view {view_key} must be same size. "
#                              f"{files[i]} has {rgb.shape}, mask {m.shape}, expected {(H, W, 3)}")

#     # Load scene overlay
#     scene_img = Image.open(scene_path).convert("RGB")
#     scene_rgb = np.asarray(scene_img, dtype=np.uint8)
#     if scene_rgb.shape != (H, W, 3):
#         raise ValueError(
#             f"Scene image size mismatch for view {view_key}. "
#             f"Scene {os.path.basename(scene_path)} has {scene_rgb.shape}, expected {(H, W, 3)}"
#         )

#     # Occlusion computation
#     occ_masks = compute_occlusion_masks(masks)
#     occlusion_pct = []
#     for m, occ in zip(masks, occ_masks):
#         total = int(m.sum())
#         occluded = int(occ.sum())
#         pct = (occluded / total * 100.0) if total > 0 else 0.0
#         occlusion_pct.append(pct)

#     # compute per-view std
#     view_avg = float(np.mean(occlusion_pct)) if occlusion_pct else 0.0
#     view_std = float(np.std(occlusion_pct)) if occlusion_pct else 0.0

#     # Output figure path
#     th, ph = view_key
#     out_name = f"theta{th}_phi{ph}_occlusion.png"
#     fig_path = os.path.join(out_dir, out_name)

#     # Report figure (overlay = SCENE)
#     make_report_figure(
#         rgb_images=rgb_images,
#         filenames=files,
#         occlusion_pct=occlusion_pct,
#         scene_overlay_rgb=scene_rgb,
#         out_path=fig_path,
#         dpi=dpi,
#         max_cols=max_cols
#     )

#     # Per-view summary
#     print(f"[view theta{th}_phi{ph}] Saved report: {fig_path}")
#     per_object_rows = []
#     for i, (fp, pct) in enumerate(zip(files, occlusion_pct), start=1):
#         base = os.path.basename(fp)
#         m = OBJ_RE.match(base)
#         obj_id = int(m.group(3)) if m else None
#         print(f"  [{i:02d}] {base}  occlusion={pct:6.2f}%")
#         per_object_rows.append({
#             "level": "object",
#             "theta": int(th),
#             "phi": int(ph),
#             "obj_id": obj_id,
#             "filename": base,
#             "occlusion_pct": round(pct, 6),
#             "view_avg_pct": round(view_avg, 6),
#             "view_std_pct": round(view_std, 6),
#             "figure_path": fig_path
#         })
#     print(f"  View average occlusion: {view_avg:.2f}%  std: {view_std:.2f}%\n")

#     view_row = {
#         "level": "view",
#         "theta": int(th),
#         "phi": int(ph),
#         "obj_id": "",
#         "filename": "",
#         "occlusion_pct": "",
#         "view_avg_pct": round(view_avg, 6),
#         "view_std_pct": round(view_std, 6),
#         "figure_path": fig_path
#     }

#     return occlusion_pct, view_avg, view_std, len(occlusion_pct), fig_path, per_object_rows, view_row

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

    # -------------------------
    # Robust occlusion: compare each object's solo mask to the SCENE overlay by color.
    # This is order-independent and relies on each object image being colored with
    # a unique UID color (as your pipeline already does).
    # -------------------------
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

    # We'll write a simple CSV: viewpoint, occlusion
    csv_path = os.path.join(out_dir, "occlusion_summary.csv")

    # Accumulators for overall stats
    all_pcts = []
    total_objs = 0

    # Collect per-view averages and stds for writing the simplified CSV and plotting
    view_summaries = []  # list of (view_name:str, view_avg:float, view_std:float)

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

    # Grid Plot
    try:
        # parse view_summaries into numeric theta, phi arrays and values
        thetas = sorted({int(v.split("_")[0].replace("theta", "")) for v, _, _ in view_summaries})
        phis = sorted({int(v.split("_")[1].replace("phi", "")) for v, _, _ in view_summaries})
        thetas_arr = np.array(thetas)
        phis_arr = np.array(phis)

        grid = np.full((len(thetas_arr), len(phis_arr)), np.nan, dtype=float)

        # fill grid
        for vname, vavg, _ in view_summaries:
            th_str, ph_str = vname.split("_")
            th = int(th_str.replace("theta", ""))
            ph = int(ph_str.replace("phi", ""))

            # NEW: extract scalar indices safely and handle missing entries
            i_idx = np.flatnonzero(thetas_arr == th)
            j_idx = np.flatnonzero(phis_arr == ph)
            if i_idx.size == 0 or j_idx.size == 0:
                # unexpected: this theta/phi wasn't found in the arrays — skip it
                continue
            i = int(i_idx[0])
            j = int(j_idx[0])

            grid[i, j] = vavg

        grid_path = os.path.join(out_dir, "occlusion_grid.png")
        plt.figure(figsize=(max(6, len(phis_arr) * 0.3), max(4, len(thetas_arr) * 0.3)), dpi=args.dpi)
        im = plt.imshow(grid, aspect='auto', origin='lower')

        # invert the y axis so top and bottom are reversed
        ax = plt.gca()
        ax.invert_yaxis()

        plt.colorbar(im, label="Occlusion (%)")
        plt.title("Per-view average occlusion (theta rows, phi cols)")
        plt.xlabel("phi (deg) — columns in ascending order")
        plt.ylabel("theta (deg) — rows in ascending order")

        # annotate axis ticks with actual phi/theta values if reasonable size
        if len(phis_arr) <= 20:
            plt.xticks(range(len(phis_arr)), phis_arr, rotation=90)
        if len(thetas_arr) <= 20:
            # Because we inverted the y-axis, ticks still correspond to row indices.
            plt.yticks(range(len(thetas_arr)), thetas_arr)

        plt.tight_layout()
        plt.savefig(grid_path, bbox_inches="tight")
        plt.close()
        print(f"Saved grid image: {grid_path}")
    except Exception as e:
        print(f"Failed to create grid plot: {e}")



if __name__ == "__main__":
    main()
