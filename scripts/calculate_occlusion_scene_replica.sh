#!/usr/bin/env bash
set -euo pipefail

REPLICA_ROOT="../data/replica"

for scene_dir in "$REPLICA_ROOT"/*; do
  [[ -d "$scene_dir" ]] || continue

  scene_num="$(basename "$scene_dir")"
  echo "Occlusion for replica scene: $scene_num"

  python3 calculate_occlusion.py --dataset-name "$scene_num"
done
