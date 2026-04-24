#!/usr/bin/env bash
# Run Step 7: generate SAM-Med3D prompts from nnU-Net coarse masks.

set -euo pipefail

MASK_ARG=""
IMAGE_ARG=""
OUT_ARG=""
LOG_DIR="logs"
MAX_POS=4
MAX_NEG=4
MIN_COMP=10
NEG_ITERS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mask-dir)              MASK_ARG="--mask-dir $2"; shift 2 ;;
        --image-dir)             IMAGE_ARG="--image-dir $2"; shift 2 ;;
        --output-dir)            OUT_ARG="--output-dir $2"; shift 2 ;;
        --max-positive-points)   MAX_POS="$2"; shift 2 ;;
        --max-negative-points)   MAX_NEG="$2"; shift 2 ;;
        --min-component-voxels)  MIN_COMP="$2"; shift 2 ;;
        --negative-shell-iters)  NEG_ITERS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -f .env ]] && { set -a; source .env; set +a; echo "[prompts] Loaded .env"; } \
              || echo "[prompts] WARNING: .env not found"

mkdir -p "$LOG_DIR" prompts

echo "━━━ Step 7  SAM-Med3D Prompt Generation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# shellcheck disable=SC2086
python scripts/07_generate_sam_prompts.py \
    $MASK_ARG \
    $IMAGE_ARG \
    $OUT_ARG \
    --max-positive-points "$MAX_POS" \
    --max-negative-points "$MAX_NEG" \
    --min-component-voxels "$MIN_COMP" \
    --negative-shell-iters "$NEG_ITERS" \
    --log-dir "$LOG_DIR"

echo ""
echo "━━━ Prompt generation complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
