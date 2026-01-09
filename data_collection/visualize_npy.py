import argparse
from pathlib import Path

import numpy as np
import cv2


# ---------------------------
# Utilities
# ---------------------------

def guess_depth_key(files):
    for k in files:
        lk = k.lower()
        if "depth" in lk:
            return k
    return None

def guess_seg_key(files):
    # prefer full-scene seg if both are present
    for pref in ["seg_imgs", "seg"]:
        if pref in files:
            return pref
    for k in files:
        lk = k.lower()
        if "seg" in lk and "per_obj" not in lk and "mask" not in lk:
            return k
    return None

def guess_extr_key(files):
    for k in files:
        lk = k.lower()
        if "extr" in lk or "pose" in lk:
            return k
    return None

def angle_tag(theta_deg_array, phi_deg_array, i):
    """Return angle tag like 'theta30_phi90' for frame i."""
    if theta_deg_array is None or phi_deg_array is None:
        return f"{i:03d}"
    th = int(round(float(theta_deg_array[i])))
    ph = int(round(float(phi_deg_array[i])))
    return f"theta{th:03d}_phi{ph:03d}"

def load_bundle(input_path: Path):
    """
    Load as much as possible from:
      - .npz: loose key detection for depth/seg/extrinsics (+ per_obj_masks, angles if present)
      - dir : look for *depth*.npy, *seg*.npy, *extr*.npy
      - .npy: ints->seg, floats->depth
    Returns a dict subset of:
      {
        'depth_imgs': (N,H,W) float32,
        'seg':        (N,H,W) uint{16,32},
        'extrinsics': (N,7)   float32,
        'per_obj_masks': (N,K,H,W) uint8,
        'per_obj_seg_uids': (N,K,H,W) int32,
        'uid_color_lut': (K,4) uint8,
        'obj_uids':   (K,) int32,
        'view_theta_deg': (N,) float32,
        'view_phi_deg':   (N,) float32,
        'scene_id': str
      }
    """
    inp = Path(input_path)
    out = {}

    if inp.is_file():
        if inp.suffix == ".npz":
            data = np.load(inp, allow_pickle=True)
            files = list(data.files)

            # Optional direct keys
            if "depth_imgs" in files:
                d = data["depth_imgs"]
                if d.ndim == 2:
                    d = d[None, ...]
                out["depth_imgs"] = d.astype(np.float32)

            if "seg_imgs" in files:
                s = data["seg_imgs"]
                if s.ndim == 2:
                    s = s[None, ...]
                out["seg"] = s

            if "per_obj_masks" in files:
                pom = data["per_obj_masks"]
                # expect (N,K,H,W) or (K,H,W). Normalize to (N,K,H,W)
                if pom.ndim == 3:
                    pom = pom[None, ...]
                out["per_obj_masks"] = pom.astype(np.uint8)

            if "per_obj_seg_uids" in files:
                posu = data["per_obj_seg_uids"]
                if posu.ndim == 3:
                    posu = posu[None, ...]
                out["per_obj_seg_uids"] = posu.astype(np.int32)

            if "uid_color_lut" in files:
                out["uid_color_lut"] = data["uid_color_lut"].astype(np.uint8)

            if "obj_uids" in files:
                out["obj_uids"] = data["obj_uids"].astype(np.int32)

            if "extrinsics" in files:
                out["extrinsics"] = data["extrinsics"].astype(np.float32)

            # Angle metadata for naming
            if "view_theta_deg" in files:
                out["view_theta_deg"] = data["view_theta_deg"].astype(np.float32)
            if "view_phi_deg" in files:
                out["view_phi_deg"] = data["view_phi_deg"].astype(np.float32)

            if "scene_id" in files:
                # may be stored as 0-d array
                sid = data["scene_id"]
                if isinstance(sid, np.ndarray):
                    sid = sid.item()
                out["scene_id"] = str(sid)

            # If some standard keys are missing, try to guess
            if "depth_imgs" not in out:
                dkey = guess_depth_key(files)
                if dkey is not None:
                    d = data[dkey]
                    if d.ndim == 2:
                        d = d[None, ...]
                    out["depth_imgs"] = d.astype(np.float32)

            if "seg" not in out:
                skey = guess_seg_key(files)
                if skey is not None:
                    s = data[skey]
                    if s.ndim == 2:
                        s = s[None, ...]
                    out["seg"] = s

            if "extrinsics" not in out:
                ekey = guess_extr_key(files)
                if ekey is not None:
                    out["extrinsics"] = data[ekey].astype(np.float32)

        elif inp.suffix == ".npy":
            arr = np.load(inp, allow_pickle=True)
            if np.issubdtype(arr.dtype, np.integer):
                if arr.ndim == 2:
                    arr = arr[None, ...]
                out["seg"] = arr
            else:
                if arr.ndim == 2:
                    arr = arr[None, ...]
                out["depth_imgs"] = arr.astype(np.float32)
        else:
            raise ValueError(f"Unsupported file type: {inp.suffix}")

    else:
        d = inp
        depth_paths = [p for p in [
            d / "depth_imgs.npy",
            *d.glob("*depth*.npy"),
            *d.glob("*Depth*.npy"),
        ] if p.exists()]
        seg_paths = [p for p in [
            d / "seg_imgs.npy",
            d / "seg.npy",
            *d.glob("*seg*.npy"),
            *d.glob("*mask*.npy"),
            *d.glob("*Seg*.npy"),
        ] if p.exists()]
        extr_paths = [p for p in [
            d / "extrinsics.npy",
            *d.glob("*extr*.npy"),
            *d.glob("*pose*.npy"),
        ] if p.exists()]
        perobj_paths = [p for p in [
            d / "per_obj_masks.npy",
            *d.glob("*per_object*.npy"),
            *d.glob("*per_obj*.npy"),
        ] if p.exists()]
        theta_paths = [p for p in [
            d / "view_theta_deg.npy",
        ] if p.exists()]
        phi_paths = [p for p in [
            d / "view_phi_deg.npy",
        ] if p.exists()]

        if depth_paths:
            depth = np.load(depth_paths[0])
            if depth.ndim == 2:
                depth = depth[None, ...]
            out["depth_imgs"] = depth.astype(np.float32)
        if seg_paths:
            seg = np.load(seg_paths[0])
            if seg.ndim == 2:
                seg = seg[None, ...]
            out["seg"] = seg
        if extr_paths:
            out["extrinsics"] = np.load(extr_paths[0]).astype(np.float32)
        if perobj_paths:
            pom = np.load(perobj_paths[0])
            if pom.ndim == 3:
                pom = pom[None, ...]
            out["per_obj_masks"] = pom.astype(np.uint8)
        if theta_paths:
            out["view_theta_deg"] = np.load(theta_paths[0]).astype(np.float32)
        if phi_paths:
            out["view_phi_deg"] = np.load(phi_paths[0]).astype(np.float32)

    if "depth_imgs" not in out and "seg" not in out and "per_obj_masks" not in out:
        raise FileNotFoundError("No depth, segmentation, or per-object masks found.")
    return out

def labels_to_color_with_lut(labels_2d: np.ndarray, uid_color_lut: np.ndarray, obj_uids: np.ndarray):
    """
    Colorize a 2D label map using the provided UID color lookup table.
    labels_2d: (H,W) array where values are UIDs (0 = background)
    uid_color_lut: (K,4) RGBA colors
    obj_uids: (K,) UIDs corresponding to rows in uid_color_lut
    """
    h, w = labels_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create UID -> color mapping
    uid_to_color = {}
    for i, uid in enumerate(obj_uids):
        uid_to_color[int(uid)] = uid_color_lut[i, :3]  # RGB only
    
    # Apply colors
    for uid, color in uid_to_color.items():
        mask = (labels_2d == uid)
        rgb[mask] = color
    
    return rgb

def labels_to_color(labels_2d: np.ndarray):
    """
    Fallback: Colorize a 2D label map (0 = background) with random colors.
    Used when uid_color_lut is not available.
    """
    ids = np.unique(labels_2d)
    rng = np.random.RandomState(1234)
    lut = {
        int(i): (
            int(rng.randint(0, 256)),
            int(rng.randint(0, 256)),
            int(rng.randint(0, 256)),
        )
        for i in ids if i != 0
    }
    h, w = labels_2d.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for i in ids:
        if i == 0:
            continue
        rgb[labels_2d == i] = lut[int(i)]
    return rgb

def per_object_to_labels(seg_khw: np.ndarray, obj_uids: np.ndarray = None):
    """
    Convert per-object masks (K,H,W) to a single label image (H,W).
    If obj_uids is provided, use UIDs as labels; otherwise use indices 1..K.
    If multiple objects overlap, earlier indices win.
    """
    seg_khw = (seg_khw != 0)  # ensure boolean
    K, H, W = seg_khw.shape
    labels = np.zeros((H, W), dtype=np.int32)
    
    for k in range(K):
        mask = seg_khw[k]
        if obj_uids is not None:
            label_val = int(obj_uids[k])
        else:
            label_val = k + 1
        labels[(labels == 0) & mask] = label_val
    
    return labels

def per_object_seg_uids_to_labels(seg_uids_khw: np.ndarray):
    """
    Convert per-object UID-labeled masks (K,H,W) to a single label image (H,W).
    Each (K,H,W) slice contains UID values where the object is present, 0 elsewhere.
    Combine by taking max across K dimension (assumes no overlap).
    """
    labels = np.max(seg_uids_khw, axis=0).astype(np.int32)
    return labels


# ---------------------------
# Batch segmentation export
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Create segmentation PNGs (scene + per-object) from a .npz/.npy bundle or directory."
    )
    ap.add_argument(
        "--scene",
        type=Path,
        help="Path to .npz/.npy or a directory (supports *_all.npz-style bundles).",
    )

    args = ap.parse_args()

    scene_dir = Path("../data") / args.scene
    npz_path = next(scene_dir.glob("*.npz"))  

    print(f"Using bundle: {npz_path}")

    bundle = load_bundle(npz_path)

    segs = bundle.get("seg", None)                      # (N,H,W) int or None (full-scene)
    per_obj_masks = bundle.get("per_obj_masks", None)   # (N,K,H,W) uint8 or None
    per_obj_seg_uids = bundle.get("per_obj_seg_uids", None)  # (N,K,H,W) int32 or None
    uid_color_lut = bundle.get("uid_color_lut", None)   # (K,4) uint8 or None
    obj_uids = bundle.get("obj_uids", None)
    theta_deg = bundle.get("view_theta_deg", None)      # (N,) float32 or None
    phi_deg = bundle.get("view_phi_deg", None)          # (N,) float32 or None

    if segs is not None:
        print(f"Loaded scene seg:        {segs.shape} (N,H,W)")
    if per_obj_masks is not None:
        print(f"Loaded per-object masks: {per_obj_masks.shape} (N,K,H,W)")
    if per_obj_seg_uids is not None:
        print(f"Loaded per-object UIDs:  {per_obj_seg_uids.shape} (N,K,H,W)")
    if uid_color_lut is not None:
        print(f"Loaded UID color LUT:    {uid_color_lut.shape} (K,4)")
    if obj_uids is not None:
        print(f"Loaded object UIDs:      {obj_uids.shape} (K,)")

    # Determine frame count (N)
    n_seg = segs.shape[0] if segs is not None else 0
    n_pom = per_obj_masks.shape[0] if per_obj_masks is not None else 0
    n_posu = per_obj_seg_uids.shape[0] if per_obj_seg_uids is not None else 0
    N = max(n_seg, n_pom, n_posu)
    if N == 0:
        print("No segmentation data to export.")
        return

    outdir = scene_dir
    # folder for ground truth segmentation masks of a scene
    scene_dir = outdir / "scene_groundtruths"
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # folder for ground truth segmentation masks of each object
    object_dir = outdir / "object_groundtruths"
    object_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N):
        tag = angle_tag(theta_deg, phi_deg, i)

        # Whole-scene segmentation image for this view: <angle>_scene.png
        if segs is not None and i < n_seg:
            # Use the full-scene segmentation (has correct occlusion)
            scene_labels = segs[i].astype(np.int32)
            if uid_color_lut is not None and obj_uids is not None:
                img = labels_to_color_with_lut(scene_labels, uid_color_lut, obj_uids)
            else:
                img = labels_to_color(scene_labels)
            cv2.imwrite(str(scene_dir / f"{tag}_scene.png"), img)

        elif (
            per_obj_seg_uids is not None
            and i < n_posu
            and uid_color_lut is not None
            and obj_uids is not None
        ):
            # Fallback: compose from per-object UID layers (no occlusion logic)
            scene_labels = per_object_seg_uids_to_labels(per_obj_seg_uids[i])
            img = labels_to_color_with_lut(scene_labels, uid_color_lut, obj_uids)
            cv2.imwrite(str(scene_dir / f"{tag}_scene.png"), img)

        elif per_obj_masks is not None and i < n_pom:
            # Last-resort fallback: binary masks fused in index order (no occlusion logic)
            scene_labels = per_object_to_labels(per_obj_masks[i], obj_uids)
            if uid_color_lut is not None and obj_uids is not None:
                img = labels_to_color_with_lut(scene_labels, uid_color_lut, obj_uids)
            else:
                img = labels_to_color(scene_labels)
            cv2.imwrite(str(scene_dir / f"{tag}_scene.png"), img)

        # Per-object masks for this view: <angle>_obj<UID>_color.png with consistent colors
        if per_obj_masks is not None and i < n_pom:
            K = per_obj_masks.shape[1]
            for k in range(K):
                mask = per_obj_masks[i, k]
                if uid_color_lut is not None and obj_uids is not None:
                    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    colored_mask[mask != 0] = uid_color_lut[k, :3]
                    uid = int(obj_uids[k])
                    cv2.imwrite(str(object_dir / f"{tag}_obj{uid:03d}_color.png"), colored_mask)

    print(f"Saved segmentation PNGs to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
