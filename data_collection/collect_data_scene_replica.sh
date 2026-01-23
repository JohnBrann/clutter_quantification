#!/usr/bin/env bash
set -euo pipefail

META_JSON="./scene_replica_scenes/metadata.json"
SCENE_DIR="./scene_replica_scenes"
OUT_ROOT="../data/replica"

mkdir -p "$OUT_ROOT"

jq -r 'to_entries | sort_by(.key) | .[] | "\(.key)\t\(.value.layout)"' "$META_JSON" |
while IFS=$'\t' read -r scene_key layout_file; do
  # Strip "scene_" prefix → scene_1 → 1
  scene_id="${scene_key#scene_}"

  python3 ../data_collection/create_scene.py \
    --scene replica \
    --object-set ycb \
    --replica-scene-id "$scene_id"
done
