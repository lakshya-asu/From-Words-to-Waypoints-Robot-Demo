#!/usr/bin/env bash
set -euo pipefail

IN_DIR="/root/graph_eqa/spatial_experiment/evals/out"
OUT_DIR="/root/graph_eqa/spatial_experiment/evals/compare_out"
mkdir -p "$OUT_DIR"

python compare_methods.py \
  --input "msp_point=$IN_DIR/eval_msp_point.csv" \
  --input "msp_object=$IN_DIR/eval_msp_object.csv" \
  --out_dir "$OUT_DIR"

echo "[OK] compare outputs in: $OUT_DIR"
ls -lh "$OUT_DIR"