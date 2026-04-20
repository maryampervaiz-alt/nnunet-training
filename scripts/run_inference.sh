#!/usr/bin/env bash
# Run Step 4: batch inference with nnU-Net v2.
#
# Usage:
#   bash scripts/run_inference.sh [options]
#
# Options:
#   --cv              Per-fold CV inference (required before --cv-mode evaluation)
#   --ensemble        Ensemble over all folds on imagesTs (default when no flag given)
#   --input DIR       Custom input directory of *_0000.nii.gz images
#   --output DIR      Custom output directory
#   --folds N [N ...] Fold indices to include in ensemble (default: all)
#   --config NAME     nnU-Net configuration override
#   --cpu             Run on CPU (slow; for debugging only)
#   --disable-tta     Disable test-time augmentation (faster, slightly lower accuracy)
#   --save-prob       Save softmax probability maps
#
# Examples:
#   bash scripts/run_inference.sh --cv
#   bash scripts/run_inference.sh --ensemble
#   bash scripts/run_inference.sh --input /data/new_cases --output inference_outputs/custom

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CV_MODE=false
INPUT_ARG=""
OUTPUT_ARG=""
FOLDS_ARG=""
CONFIG_ARG=""
DEVICE_ARG=""
TTA_ARG=""
PROB_ARG=""
LOG_DIR="logs"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cv)          CV_MODE=true; shift ;;
        --ensemble)    shift ;;   # ensemble is default; flag accepted for clarity
        --input)       INPUT_ARG="--input $2"; shift 2 ;;
        --output)      OUTPUT_ARG="--output $2"; shift 2 ;;
        --folds)       shift; FOLDS_ARG="--folds"; while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do FOLDS_ARG="$FOLDS_ARG $1"; shift; done ;;
        --config)      CONFIG_ARG="--configuration $2"; shift 2 ;;
        --cpu)         DEVICE_ARG="--device cpu"; shift ;;
        --disable-tta) TTA_ARG="--disable-tta"; shift ;;
        --save-prob)   PROB_ARG="--save-probabilities"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Load environment ──────────────────────────────────────────────────────────
[[ -f .env ]] && { set -a; source .env; set +a; echo "[inference] Loaded .env"; } \
              || echo "[inference] WARNING: .env not found"

mkdir -p "$LOG_DIR" inference_outputs

# ── Run inference ─────────────────────────────────────────────────────────────
if [[ "$CV_MODE" == true ]]; then
    echo "━━━ Inference  (CV per-fold mode) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # shellcheck disable=SC2086
    python scripts/04_inference.py \
        --cv-mode \
        --output inference_outputs/cv \
        $CONFIG_ARG \
        $DEVICE_ARG \
        $TTA_ARG \
        $PROB_ARG \
        --log-dir "$LOG_DIR"
    OUTPUT_DIR="inference_outputs/cv"
else
    echo "━━━ Inference  (Ensemble mode) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # shellcheck disable=SC2086
    python scripts/04_inference.py \
        ${OUTPUT_ARG:-"--output inference_outputs/ensemble"} \
        $INPUT_ARG \
        $FOLDS_ARG \
        $CONFIG_ARG \
        $DEVICE_ARG \
        $TTA_ARG \
        $PROB_ARG \
        --log-dir "$LOG_DIR"
    OUTPUT_DIR="${OUTPUT_ARG#--output }"
    OUTPUT_DIR="${OUTPUT_DIR:-inference_outputs/ensemble}"
fi

echo ""
echo "━━━ Inference complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Predictions : $OUTPUT_DIR"
echo "  Logs        : $LOG_DIR/"
