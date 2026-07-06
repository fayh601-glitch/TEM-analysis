#!/usr/bin/env bash
# =============================================================================
# analyze_my_image.sh — guided TEM analysis (no coding required)
#
# Usage:
#   bash scripts/analyze_my_image.sh
#   bash scripts/analyze_my_image.sh /path/to/my_tem.png
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "First-time setup: creating Python environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -q -r requirements.txt
  pip install -q -e .
else
  source .venv/bin/activate
fi

IMAGE_PATH="${1:-}"
if [[ -z "$IMAGE_PATH" ]]; then
  echo ""
  echo "=== TEM Nanorod Analyzer ==="
  echo "Drag your image into Terminal, or type the full path:"
  read -r IMAGE_PATH
  IMAGE_PATH="${IMAGE_PATH//\'/}"
  IMAGE_PATH="${IMAGE_PATH//\"/}"
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Error: file not found: $IMAGE_PATH"
  exit 1
fi

echo ""
echo "What type of particles are in this image?"
echo "  1) Nanorods only  (recommended for Enright S2A starting rods)"
echo "  2) Round dots only"
echo "  3) Both rods and dots"
read -r -p "Choice [1]: " MODE_CHOICE
MODE_CHOICE="${MODE_CHOICE:-1}"
case "$MODE_CHOICE" in
  1) MODE="rods"; PRESET="enright_rods" ;;
  2) MODE="dots"; PRESET="dots_only" ;;
  3) MODE="both"; PRESET="screenshot" ;;
  *) MODE="rods"; PRESET="enright_rods" ;;
esac

echo ""
read -r -p "Scale bar length in nm [20]: " BAR_NM
BAR_NM="${BAR_NM:-20}"

echo ""
echo "Measure the white scale-bar LINE in pixels (ImageJ line tool or estimate)."
read -r -p "Scale bar length in pixels: " BAR_PX
if [[ -z "$BAR_PX" ]]; then
  echo "Error: scale bar pixels are required for nm measurements."
  exit 1
fi

STEM="$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')"
OUT_DIR="$ROOT/outputs/user_runs/${STEM}_$(date +%Y%m%d)"
mkdir -p "$OUT_DIR"

echo ""
echo "Running analysis..."
echo "  Image:  $IMAGE_PATH"
echo "  Mode:   $MODE"
echo "  Preset: $PRESET"
echo "  Output: $OUT_DIR"
echo ""

tem-rods analyze \
  --image "$IMAGE_PATH" \
  --preset "$PRESET" \
  --mode "$MODE" \
  --scale-bar-nm "$BAR_NM" \
  --scale-bar-pixels "$BAR_PX" \
  --output-dir "$OUT_DIR"

echo ""
echo "=== Done ==="
echo "Open these files:"
echo "  Overlay:  $OUT_DIR/${STEM}_overlay.png"
echo "  CSV:      $OUT_DIR/${STEM}_measurements.csv"
if [[ -f "$OUT_DIR/${STEM}_segments_debug.png" ]]; then
  echo "  Debug:    $OUT_DIR/${STEM}_segments_debug.png"
fi
echo ""
echo "Green = rod, blue = dot, orange dashed = reject (ambiguous / filtered out)."
