#!/usr/bin/env bash
# Run Demo Analysis — analyze all three paper TEM images at once
#
# Uses the enright_rods preset (tuned cluster splitting + rod classification).
# Overlays show green=rod, blue=dot, orange dashed=reject.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs/demo

run_one() {
  local image=$1
  local bar_px=$2
  echo "=== Analyzing $image ==="
  tem-rods analyze \
    --preset enright_rods \
    --image "$image" \
    --scale-bar-nm 20 \
    --scale-bar-pixels "$bar_px" \
    --output-dir outputs/demo
}

run_one data/curated/s2_A_starting_rods.png 48
run_one data/curated/s2_B_30min.png 48
run_one data/curated/s2_D_65min.png 52

echo ""
echo "Done. Results in outputs/demo/"
echo "  open outputs/demo/s2_A_starting_rods_overlay.png"
