#!/usr/bin/env bash
set -euo pipefail

REPLICA_ROOT="../data/replica"

for scene_dir in "$REPLICA_ROOT"/*; do
  [[ -d "$scene_dir" ]] || continue

  echo "Visualizing: $scene_dir"
  python3 visualize_npy.py --scene "$scene_dir"
done
