#!/usr/bin/env python3
"""
interactive_occlusion_and_2d_grid.py

Interactive occlusion grid viewer that shows:
  - left: occlusion/scene image for hovered viewpoint
  - right: corresponding 2D distance visualization (masked/distances_viz.png)
  - text on the 2D panel: total connections, blocked connections, gripper width threshold,
    and segment counts (before/after/excluded) read from connections_summary.csv

Usage:
  python interactive_occlusion_and_2d_grid.py \
    --csv ./scene01_out/occlusion_summary.csv \
    --scene-dir scene01/scene/ \
    --distances-dir scene01_out/ \
    --connections-summary scene01_out/connections_summary.csv \
    --gripper-width 32.0
"""
import argparse
import csv
import os
from glob import glob
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

SCENE_RE = re.compile(r"theta0*?(\d+)_phi0*?(\d+)_scene", flags=re.IGNORECASE)
DIST_RE = re.compile(r"theta0*?(\d+)_phi0*?(\d+).*_distances_viz", flags=re.IGNORECASE)
VIEWNAME_RE = re.compile(r"^theta(\d+)_phi(\d+)$", flags=re.IGNORECASE)
SCENE_BASENAME_RE = re.compile(r"^(theta0*?(\d+)_phi0*?(\d+))(?:_scene)?$", flags=re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive occlusion + 2D distances grid viewer.")
    p.add_argument("--dataset-name", required=True, help="Dataset folder name under ./data/<dataset_name>/")
    p.add_argument("--gripper-width", type=float, default=None,
                   help="Gripper width to display on the 2D panel (informational). If not provided, shows 'unknown'.")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI for saved preview image.")
    p.add_argument("--max_fig_w", type=float, default=10.0, help="Maximum figure width (inches).")
    p.add_argument("--max_fig_h", type=float, default=7.0, help="Maximum figure height (inches).")
    return p.parse_args()



def read_csv_view_summaries(csv_path):
    """
    Reads the simplified CSV (viewpoint, occlusion). Returns list of (viewpoint_str, occl_float).
    Skips the 'full_scene' final row.
    """
    view_summaries = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV {csv_path} is empty")

    # Handle header detection: first row contains 'viewpoint' probably
    start_idx = 0
    first = [c.strip().lower() for c in rows[0]]
    if len(first) >= 1 and first[0] == "viewpoint":
        start_idx = 1

    for r in rows[start_idx:]:
        if len(r) < 2:
            continue
        vname = r[0].strip()
        val = r[1].strip()
        if vname.lower() == "full_scene":
            continue
        try:
            vavg = float(val)
        except Exception:
            vavg = np.nan
        view_summaries.append((vname, vavg))
    return view_summaries


def index_scenes(scene_dir):
    """
    Return dict mapping (theta_int, phi_int) -> full_path for files matching SCENE_RE.
    Accepts many zero-padding formats.
    """
    scenes = {}
    if not os.path.isdir(scene_dir):
        return scenes
    for p in glob(os.path.join(scene_dir, "*.png")):
        base = os.path.basename(p)
        m = SCENE_RE.search(base)
        if m:
            th = int(m.group(1))
            ph = int(m.group(2))
            scenes[(th, ph)] = p
    return scenes


def index_distances_viz(dist_dir):
    """
    Return dict mapping (theta_int, phi_int) -> full_path for *_distances_viz.png.
    """
    dmap = {}
    if not dist_dir or not os.path.isdir(dist_dir):
        return dmap
    for p in glob(os.path.join(dist_dir, "*_distances_viz.png")):
        base = os.path.basename(p)
        m = DIST_RE.search(base)
        if m:
            th = int(m.group(1))
            ph = int(m.group(2))
            dmap[(th, ph)] = p
        else:
            # fallback: try to extract "thetaX_phiY" anywhere in filename
            m2 = VIEWNAME_RE.search(base)
            if m2:
                th = int(m2.group(1))
                ph = int(m2.group(2))
                dmap[(th, ph)] = p
    return dmap


def _make_view_keys_from_scene_field(scene_field):
    """
    Given a 'scene' value from connections_summary (e.g. 'theta000_phi000_scene' or 'theta000_phi000'),
    yield a set of possible keys for matching:
      - exact basename (scene_field)
      - with/without '_scene'
      - zero-padded and unpadded forms for by_view (theta_int,phi_int)
    Returns (basename_keys, view_keys) where:
      basename_keys -> list of basenames to index into by_basename
      view_keys -> list of (theta_int, phi_int) keys to index into by_view
    """
    bkeys = set()
    vkeys = set()
    base = os.path.splitext(scene_field)[0]
    bkeys.add(base)
    # if endswith _scene, add version without it
    if base.endswith("_scene"):
        bkeys.add(base[:-6])
    else:
        bkeys.add(base + "_scene")

    m = SCENE_BASENAME_RE.match(base)
    if m:
        keyroot = m.group(1)  # e.g. 'theta000_phi000' or 'theta0_phi0'
        # extract ints
        vm = VIEWNAME_RE.match(keyroot)
        if vm:
            try:
                th = int(vm.group(1)); ph = int(vm.group(2))
                vkeys.add((th, ph))
            except Exception:
                pass
        else:
            # attempt to parse numbers inside keyroot
            mm = re.search(r"theta0*?(\d+)_phi0*?(\d+)", keyroot, flags=re.IGNORECASE)
            if mm:
                try:
                    th = int(mm.group(1)); ph = int(mm.group(2))
                    vkeys.add((th, ph))
                except Exception:
                    pass
    return list(bkeys), list(vkeys)


def load_connections_summary(summary_csv_path):
    """
    Read connections_summary.csv into dict keyed by scene basename (no ext)
    and by (theta_int,phi_int). The CSV is expected to contain at least the header:
      scene, num_segments_before, num_segments_after, num_excluded_segments, total_connections, blocked_connections, fraction_blocked
    We create:
      - by_basename[basename] = rowdict
      - by_view[(theta,phi)] = rowdict
    where rowdict stores the original CSV dict (values as strings).
    """
    by_basename = {}
    by_view = {}
    if not summary_csv_path or not os.path.isfile(summary_csv_path):
        return by_basename, by_view

    with open(summary_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            scene = r.get("scene", "")
            if not scene:
                # try to build a key from filename-like fields if present
                # otherwise skip
                continue
            # normalize basename (no extension)
            base = os.path.splitext(scene)[0]
            by_basename[base] = r

            # build alternative keys and index by view ints when possible
            bkeys, vkeys = _make_view_keys_from_scene_field(base)
            for bk in bkeys:
                by_basename[bk] = r
            for vk in vkeys:
                by_view[vk] = r

    return by_basename, by_view


def main():
    args = parse_args()

    base_dir = os.path.join(".", "data", args.dataset_name)

    # derived inputs
    csv_path = os.path.join(base_dir, "occlusion", "occlusion_summary.csv")
    scene_dir = os.path.join(base_dir, "scene_groundtruths")
    dist_dir = os.path.join(base_dir, "distance")
    summary_path = os.path.join(dist_dir, "connections_summary.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Occlusion CSV not found: {csv_path}")

    # output goes to occlusion folder
    out_dir = os.path.join(base_dir, "occlusion")
    os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(out_dir, "occlusion_grid.png")

    view_summaries = read_csv_view_summaries(csv_path)
    if not view_summaries:
        raise ValueError("No viewpoint rows found in CSV")

    scenes = index_scenes(scene_dir)
    if not scenes:
        print(f"[warn] No scene images found in {scene_dir}. Preview images will not appear on hover.")

    dist_viz_map = index_distances_viz(dist_dir)
    if not dist_viz_map:
        print(f"[warn] No distances visualization PNGs found in {dist_dir}. 2D previews will not appear.")

    by_basename, by_view = load_connections_summary(summary_path) if summary_path else ({}, {})
    if summary_path:
        print(f"Loaded connections summary from: {summary_path} (rows: {len(by_basename)})")

    # parse numeric theta/phi sets from CSV viewpoint strings
    thetas = sorted({int(v.split("_")[0].replace("theta", "")) for v, _ in view_summaries})
    phis = sorted({int(v.split("_")[1].replace("phi", "")) for v, _ in view_summaries})
    thetas_arr = np.array(thetas)
    phis_arr = np.array(phis)

    grid = np.full((len(thetas_arr), len(phis_arr)), np.nan, dtype=float)
    scene_image_map = {}

    # map view-name rows to grid indices and scene image paths
    for vname, vavg in view_summaries:
        m = VIEWNAME_RE.match(vname)
        if not m:
            continue
        th = int(m.group(1)); ph = int(m.group(2))
        i_idx = np.flatnonzero(thetas_arr == th)
        j_idx = np.flatnonzero(phis_arr == ph)
        if i_idx.size == 0 or j_idx.size == 0:
            continue
        i = int(i_idx[0]); j = int(j_idx[0])
        grid[i, j] = vavg

        # find scene image by integer key
        if (th, ph) in scenes:
            scene_image_map[(i, j)] = scenes[(th, ph)]
        else:
            # try variations (zero padded keys)
            for (kth, kph), p in scenes.items():
                if kth == th and kph == ph:
                    scene_image_map[(i, j)] = p
                    break

    # Build figure: top row has two previews (scene left, 2D right), bottom is heatmap
    try:
        base_w = max(8, len(phis_arr) * 0.5)
        base_h = max(5, len(thetas_arr) * 0.35 + 1.5)
        fig_w = min(args.max_fig_w, base_w)
        fig_h = min(args.max_fig_h, base_h)

        # Create gridspec: top row with 2 columns for previews, bottom full-width heatmap
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
        gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 0.9], hspace=0.25, wspace=0.25)

        ax_scene = fig.add_subplot(gs[0:2, 0:2])  # left preview (scene)
        ax_2d = fig.add_subplot(gs[0:2, 2:4])    # right preview (2D viz + text)
        ax_heat = fig.add_subplot(gs[2, 0:4])    # heatmap

        TITLE_FS = 10
        SMALL_TITLE_FS = 9
        TICK_FS = 8
        CBAR_FS = 8

        ax_scene.axis('off')
        ax_scene.set_title("Scene (hover a cell)", fontsize=SMALL_TITLE_FS, pad=6)

        ax_2d.axis('off')
        ax_2d.set_title("2D distances (hover a cell)", fontsize=SMALL_TITLE_FS, pad=6)

        import matplotlib as mpl
        cmap = mpl.cm.get_cmap('viridis').reversed()
        im = ax_heat.imshow(grid, aspect='auto', origin='lower', vmin=0, vmax=100,
                           cmap=cmap, interpolation='nearest')
        ax_heat.invert_yaxis()

        cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal', pad=0.10, fraction=0.06)
        cbar.ax.tick_params(labelsize=CBAR_FS)
        cbar.set_label("Occlusion (%)", fontsize=CBAR_FS)

        ax_heat.set_title("Per-view average occlusion (theta rows, phi cols)", fontsize=SMALL_TITLE_FS)
        if len(phis_arr) > 0:
            ax_heat.set_xticks(np.arange(len(phis_arr)))
            ax_heat.set_xticklabels([str(p) for p in phis_arr], rotation=90, fontsize=TICK_FS, ha='center')
            ax_heat.set_xlim(-0.5, len(phis_arr) - 0.5)
        if len(thetas_arr) > 0:
            ax_heat.set_yticks(np.arange(len(thetas_arr)))
            ax_heat.set_yticklabels([str(t) for t in thetas_arr], fontsize=TICK_FS)
            ax_heat.set_ylim(len(thetas_arr) - 0.5, -0.5)

        last_cell = [None]
        highlight_rect = [None]

        def show_no_preview(ax, title):
            ax.clear()
            ax.axis('off')
            ax.set_title(title, fontsize=SMALL_TITLE_FS, pad=6)

        def draw_2d_panel(ax, viz_path, summary_row, gripper_width):
            """
            Draw the 2D viz image on ax and overlay text from summary_row.
            summary_row expected keys (strings) possibly present in CSV:
              num_segments_before, num_segments_after, num_excluded_segments,
              total_connections, blocked_connections, fraction_blocked
            """
            ax.clear()
            ax.axis('off')
            if viz_path and os.path.isfile(viz_path):
                try:
                    img = Image.open(viz_path).convert("RGB")
                    ax.imshow(img)
                except Exception as e:
                    show_no_preview(ax, f"Error loading 2D viz: {e}")
                    return
            else:
                # draw blank background
                ax.set_facecolor((0.05, 0.05, 0.05))
                show_no_preview(ax, "No 2D distances visualization")

            # Extract numeric fields from summary_row if present (CSV values are strings)
            def get_int_field(d, keys, default=None):
                if not d:
                    return default
                for k in keys:
                    v = d.get(k)
                    if v is None or v == "":
                        continue
                    try:
                        return int(float(v))
                    except Exception:
                        continue
                return default

            def get_float_field(d, keys, default=None):
                if not d:
                    return default
                for k in keys:
                    v = d.get(k)
                    if v is None or v == "":
                        continue
                    try:
                        return float(v)
                    except Exception:
                        continue
                return default

            total = get_int_field(summary_row, ["total_connections", "num_connections"], default=None)
            blocked = get_int_field(summary_row, ["blocked_connections"], default=None)
            before = get_int_field(summary_row, ["num_segments_before"], default=None)
            after = get_int_field(summary_row, ["num_segments_after"], default=None)
            excluded = get_int_field(summary_row, ["num_excluded_segments"], default=None)
            # frac_blocked = get_float_field(summary_row, ["fraction_blocked"], default=None)

            threshold_str = f"{gripper_width:.2f} px" if (gripper_width is not None) else "unknown"

            # Compose info lines
            lines = []
            # lines.append(f"Segments before: {before if before is not None else 'N/A'}")
            # lines.append(f"Segments after:  {after if after is not None else 'N/A'}")
            lines.append(f"# Seg objs:  {excluded if excluded is not None else 'N/A'}")
            # lines.append("")  # spacer
            lines.append(f"Connections: {total if total is not None else 'N/A'}")
            lines.append(f"Blocked:     {blocked if blocked is not None else 'N/A'}")
            # if frac_blocked is not None:
            #     lines.append(f"Fraction blocked: {frac_blocked:.3f}")
            lines.append(f"EEF Threshold: {threshold_str}")
            if total is not None and total > 0 and blocked is not None:
                lines.append(f"ratio: {blocked/total:.2f}")
            else:
                lines.append("ratio: N/A")

            # Place text bottom-right
            ax.text(0.98, 0.02, "\n".join(lines),
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9, color='white',
                    bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="none", alpha=0.7))

            ax.set_title("2D distances", fontsize=SMALL_TITLE_FS, pad=6)

        def on_hover(event):
            if event.inaxes != ax_heat:
                return
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            j = int(round(x))
            i = int(round(y))
            if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
                return
            if last_cell[0] == (i, j):
                return
            last_cell[0] = (i, j)

            # remove previous highlight
            if highlight_rect[0] is not None:
                try:
                    highlight_rect[0].remove()
                except Exception:
                    pass
                highlight_rect[0] = None

            from matplotlib.patches import Rectangle
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                             linewidth=1.6, edgecolor='red', facecolor='none', zorder=5)
            ax_heat.add_patch(rect)
            highlight_rect[0] = rect

            # scene image
            if (i, j) not in scene_image_map:
                show_no_preview(ax_scene, "No scene image for this cell")
            else:
                scene_path = scene_image_map[(i, j)]
                try:
                    scene_img = Image.open(scene_path).convert("RGB")
                    ax_scene.clear()
                    ax_scene.imshow(scene_img)
                    ax_scene.axis('off')
                    theta_val = thetas_arr[i]
                    phi_val = phis_arr[j]
                    occ_val = grid[i, j]
                    occ_text = "N/A" if np.isnan(occ_val) else f"{occ_val:.2f}%"
                    ax_scene.set_title(f"theta={theta_val}°, phi={phi_val}° | Occlusion: {occ_text}", fontsize=SMALL_TITLE_FS, pad=6)
                except Exception as e:
                    show_no_preview(ax_scene, f"Error loading scene image: {e}")

            # 2D viz and stats
            # look up distances viz by (theta,phi)
            th = int(thetas_arr[i]); ph = int(phis_arr[j])
            viz_path = dist_viz_map.get((th, ph), None)
            # also try variants if not found
            if viz_path is None:
                for (kth, kph), p in dist_viz_map.items():
                    if kth == th and kph == ph:
                        viz_path = p
                        break

            # find summary row
            # try by_view (int keyed) first, then by_basename (basename forms)
            summary_row = None
            if (th, ph) in by_view:
                summary_row = by_view.get((th, ph))
            else:
                # try several basename variants that may appear in CSV
                candidates = [
                    f"theta{th}_phi{ph}_scene",
                    f"theta{th}_phi{ph}",
                    f"theta{str(th).zfill(3)}_phi{str(ph).zfill(3)}_scene",
                    f"theta{str(th).zfill(3)}_phi{str(ph).zfill(3)}"
                ]
                for c in candidates:
                    if c in by_basename:
                        summary_row = by_basename[c]
                        break

            draw_2d_panel(ax_2d, viz_path, summary_row, args.gripper_width)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        plt.savefig(grid_path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved grid image: {grid_path}")
        print("Displaying interactive plot. Hover over grid cells to view scene images and 2D distances. Close window to exit.")
        plt.show()

    except Exception as e:
        print(f"Failed to create grid plot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#python3 ./create_full_data_grid.py --csv ./scene01_out/occlusion_summary.csv --scene-dir scene01/scene/ --out-dir scene01_out/ --gripper-width 32 --connections-summary ../2d/scene01_out2d/connections_summary.csv --distances-dir ../2d/scene01_out2d/