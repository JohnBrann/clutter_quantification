import argparse
import csv
import os
from glob import glob
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

SCENE_RE = re.compile(r"^theta(\d+)_phi(\d+)_scene\.png$")

def parse_args():
    p = argparse.ArgumentParser(description="Create interactive occlusion grid from CSV + scene images.")
    p.add_argument("--dataset_name", required=True, help="Folder name under ./data/<dataset_name>.")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI for saved preview image.")
    p.add_argument("--max_fig_w", type=float, default=9.0, help="Maximum figure width (inches).")
    p.add_argument("--max_fig_h", type=float, default=6.0, help="Maximum figure height (inches).")
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
    Return dict mapping (theta_str, phi_str) -> full_path for files matching SCENE_RE.
    """
    scenes = {}
    if not os.path.isdir(scene_dir):
        return scenes
    for p in glob(os.path.join(scene_dir, "*.png")):
        m = SCENE_RE.match(os.path.basename(p))
        if m:
            th_s, ph_s = m.group(1), m.group(2)
            scenes[(th_s, ph_s)] = p
    return scenes

def main():
    p = argparse.ArgumentParser(
        description="Create interactive occlusion grid from CSV + scene images."
    )
    p.add_argument("--dataset-name", required=True, help="Folder name under ./data/<dataset_name>.")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI for saved preview image.")
    p.add_argument("--max_fig_w", type=float, default=9.0, help="Maximum figure width (inches).")
    p.add_argument("--max_fig_h", type=float, default=6.0, help="Maximum figure height (inches).")

    args = p.parse_args()

    # derive paths from dataset_name
    base_dir = os.path.join(".", "data", args.dataset_name)
    out_dir = os.path.join(base_dir, "occlusion")
    csv_path = os.path.join(out_dir, "occlusion_summary.csv")
    scene_dir = os.path.join(base_dir, "scene_groundtruths")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(out_dir, "occlusion_grid.png")

    view_summaries = read_csv_view_summaries(csv_path)
    if not view_summaries:
        raise ValueError("No viewpoint rows found in CSV")

    scenes = index_scenes(scene_dir)
    if not scenes:
        print(f"[warn] No scene images found in {scene_dir}. Preview images will not appear on hover.")

    # parse numeric theta/phi sets from CSV viewpoint strings
    thetas = sorted({int(v.split("_")[0].replace("theta", "")) for v, _ in view_summaries})
    phis = sorted({int(v.split("_")[1].replace("phi", "")) for v, _ in view_summaries})
    thetas_arr = np.array(thetas)
    phis_arr = np.array(phis)

    grid = np.full((len(thetas_arr), len(phis_arr)), np.nan, dtype=float)
    scene_image_map = {}

    pad_th = 0
    pad_ph = 0
    if scenes:
        first_key = next(iter(scenes.keys()))
        pad_th = len(first_key[0])
        pad_ph = len(first_key[1])

    for vname, vavg in view_summaries:
        try:
            th_str, ph_str = vname.split("_")
            th = int(th_str.replace("theta", ""))
            ph = int(ph_str.replace("phi", ""))
        except Exception:
            continue

        i_idx = np.flatnonzero(thetas_arr == th)
        j_idx = np.flatnonzero(phis_arr == ph)
        if i_idx.size == 0 or j_idx.size == 0:
            continue
        i = int(i_idx[0])
        j = int(j_idx[0])
        grid[i, j] = vavg

        if pad_th and pad_ph:
            th_pad = str(th).zfill(pad_th)
            ph_pad = str(ph).zfill(pad_ph)
            key = (th_pad, ph_pad)
            if key in scenes:
                scene_image_map[(i, j)] = scenes[key]
                continue
        key2 = (str(th), str(ph))
        if key2 in scenes:
            scene_image_map[(i, j)] = scenes[key2]

    # --- Interactive UI ---
    base_w = max(6, len(phis_arr) * 0.35)
    base_h = max(3.6, len(thetas_arr) * 0.30 + 1.0)
    fig_w = min(args.max_fig_w, base_w)
    fig_h = min(args.max_fig_h, base_h)

    fig, (ax_img, ax_heat) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        dpi=args.dpi,
        gridspec_kw={'height_ratios': [1.4, 0.6]}
    )

    TITLE_FS = 10
    SMALL_TITLE_FS = 9
    TICK_FS = 8
    CBAR_FS = 8

    ax_img.axis('off')
    ax_img.set_title("Hover over grid to view scene image", fontsize=TITLE_FS, pad=8)

    import matplotlib as mpl
    cmap = mpl.cm.get_cmap('viridis').reversed()
    im = ax_heat.imshow(grid, aspect='auto', origin='lower', vmin=0, vmax=100,
                        cmap=cmap, interpolation='nearest')
    ax_heat.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax_heat, label="Occlusion (%)", fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=CBAR_FS)

    ax_heat.set_title("Per-view average occlusion (theta rows, phi cols)", fontsize=SMALL_TITLE_FS)
    ax_heat.set_xlabel("phi (deg)", fontsize=SMALL_TITLE_FS)
    ax_heat.set_ylabel("theta (deg)", fontsize=SMALL_TITLE_FS)

    if len(phis_arr) > 0:
        ax_heat.set_xticks(np.arange(len(phis_arr)))
        ax_heat.set_xticklabels([str(p) for p in phis_arr], rotation=90, fontsize=TICK_FS, ha='center')
        ax_heat.set_xlim(-0.5, len(phis_arr) - 0.5)
    else:
        ax_heat.tick_params(axis='x', labelsize=TICK_FS)

    if len(thetas_arr) > 0:
        ax_heat.set_yticks(np.arange(len(thetas_arr)))
        ax_heat.set_yticklabels([str(t) for t in thetas_arr], fontsize=TICK_FS)
        ax_heat.set_ylim(len(thetas_arr) - 0.5, -0.5)
    else:
        ax_heat.tick_params(axis='y', labelsize=TICK_FS)

    plt.subplots_adjust(top=0.96, bottom=0.12, left=0.06, right=0.99, hspace=0.12)

    last_cell = [None]
    highlight_rect = [None]

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

        if highlight_rect[0] is not None:
            highlight_rect[0].remove()
            highlight_rect[0] = None
            fig.canvas.draw_idle()

        from matplotlib.patches import Rectangle
        rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                         linewidth=1.6, edgecolor='red', facecolor='none', zorder=5)
        ax_heat.add_patch(rect)
        highlight_rect[0] = rect

        if (i, j) not in scene_image_map:
            ax_img.clear()
            ax_img.axis('off')
            ax_img.set_title("No scene image for this cell", fontsize=SMALL_TITLE_FS, pad=6)
            fig.canvas.draw_idle()
            return

        scene_path = scene_image_map[(i, j)]
        try:
            scene_img = Image.open(scene_path).convert("RGB")
            ax_img.clear()
            ax_img.imshow(scene_img)
            ax_img.axis('off')

            theta_val = thetas_arr[i]
            phi_val = phis_arr[j]
            occ_val = grid[i, j]
            occ_text = "N/A" if np.isnan(occ_val) else f"{occ_val:.2f}%"

            ax_img.set_title(
                f"theta={theta_val}°, phi={phi_val}° | Occlusion: {occ_text}",
                fontsize=SMALL_TITLE_FS, pad=6
            )
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error loading scene image {scene_path}: {e}")

    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    plt.savefig(grid_path, bbox_inches="tight", dpi=args.dpi)
    print(f"Saved grid image: {grid_path}")
    print("Displaying interactive plot. Hover over grid cells to view scene images. Close window to exit.")
    plt.show()

if __name__ == "__main__":
    main()
