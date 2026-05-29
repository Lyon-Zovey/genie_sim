#!/bin/bash
set -euo pipefail

# One-click pipeline: Step2 (Isaac Sim collect) -> Step3 -> Step4a/4b/4c
# Usage:
#   ./scripts/run_sceneflow_step2_to_4.sh --task tasks/...json --headless

TASK="tasks/geniesim_2025/sort_fruit/g2/sort_the_fruit_into_the_box_apple_g2.json"
HEADLESS=false
WORKERS=4
CODEC="libx265"
BITS=10
CRF=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"; shift 2;;
    --headless)
      HEADLESS=true; shift;;
    --workers)
      WORKERS="$2"; shift 2;;
    --codec)
      CODEC="$2"; shift 2;;
    --bits)
      BITS="$2"; shift 2;;
    --crf)
      CRF="$2"; shift 2;;
    --help|-h)
      echo "Usage: ./scripts/run_sceneflow_step2_to_4.sh [--task PATH] [--headless] [--workers N] [--codec libx265] [--bits 10] [--crf 0]"
      exit 0;;
    *)
      echo "Unknown option: $1"; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DC_DIR/../.." && pwd)"

cd "$DC_DIR"

if [[ ! -f "$TASK" ]]; then
  echo "Task file not found: $TASK"
  exit 1
fi

TASK_NAME=$(python3 -c "import json; d=json.load(open('$TASK')); print(d.get('task',''))" 2>/dev/null || true)
if [[ -z "$TASK_NAME" ]]; then
  TASK_NAME=$(basename "$TASK" .json)
fi
TASK_NAME=$(echo "$TASK_NAME" | sed 's/[^a-zA-Z0-9_-]/_/g')

CAMERA_DATA_ROOT="$DC_DIR/rbs_data/$TASK_NAME/camera_data"

echo "=========================================="
echo "SceneFlow Step2-4 pipeline"
echo "task: $TASK"
echo "task_name: $TASK_NAME"
echo "camera_data_root: $CAMERA_DATA_ROOT"
echo "=========================================="

# Step 2
RUN_ARGS=(--task "$TASK")
if [[ "$HEADLESS" == "true" ]]; then
  RUN_ARGS+=(--headless)
fi

echo "[Step2] collecting trajectories..."
"$DC_DIR/scripts/run_data_collection.sh" "${RUN_ARGS[@]}"

if [[ ! -d "$CAMERA_DATA_ROOT" ]]; then
  echo "Step2 finished but camera_data root not found: $CAMERA_DATA_ROOT"
  exit 1
fi

# Ensure write permission for post-processing outputs (files written by container user 1234:1234)
# sudo chmod -R 777 "$DC_DIR/rbs_data/$TASK_NAME"

cd "$REPO_ROOT"

# Step 3
echo "[Step3] convert_camera_depths..."
python rbs_scripts/traj2sceneflow/convert_camera_depths.py \
  "source/data_collection/rbs_data/$TASK_NAME/camera_data" \
  --workers "$WORKERS"

# Step 4a
echo "[Step4a] flow_compress..."
python rbs_scripts/traj2sceneflow/flow_compress.py compress \
  --out_root "source/data_collection/rbs_data/$TASK_NAME/camera_data" \
  --codec "$CODEC" --bits "$BITS" --crf "$CRF" \
  --delete_npy

# Step 4b
echo "[Step4b] point_compress(depth)..."
python rbs_scripts/traj2sceneflow/point_compress.py \
  --mode compress \
  --root "source/data_collection/rbs_data/$TASK_NAME/camera_data" \
  --delete-existing

# Step 4c
echo "[Step4c] seg_compress..."
python rbs_scripts/traj2sceneflow/seg_compress.py compress \
  --root "source/data_collection/rbs_data/$TASK_NAME/camera_data" \
  --method b2nd \
  --delete-source

echo "=========================================="
echo "Pipeline completed: Step2-4"
echo "Output: source/data_collection/rbs_data/$TASK_NAME/camera_data"
echo "=========================================="
