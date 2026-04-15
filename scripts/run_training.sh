#!/usr/bin/env bash
# Run Steps 1–3: dataset preparation, preprocessing, and 5-fold CV training.
#
# Usage:
#   bash scripts/run_training.sh [options]
#
# Options:
#   --continue          Resume interrupted training from existing checkpoints
#   --folds N [N ...]   Train specific folds only (default: all)
#   --config NAME       nnU-Net configuration (default: 3d_fullres)
#   --seed INT          Global random seed (default: 42 or $NNUNET_SEED)
#   --skip-prepare      Skip Step 1 (data already converted)
#   --skip-preprocess   Skip Step 2 (already preprocessed)
#
# Example:
#   bash scripts/run_training.sh --folds 0 1 --seed 42

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONTINUE_FLAG=""
FOLDS_ARG=""
CONFIG_ARG=""
SEED_ARG=""
SKIP_PREPARE=false
SKIP_PREPROCESS=false
LOG_DIR="logs"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --continue)        CONTINUE_FLAG="--continue-training"; shift ;;
        --folds)           shift; FOLDS_ARG="--folds"; while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do FOLDS_ARG="$FOLDS_ARG $1"; shift; done ;;
        --config)          CONFIG_ARG="--configuration $2"; shift 2 ;;
        --seed)            SEED_ARG="--seed $2"; shift 2 ;;
        --skip-prepare)    SKIP_PREPARE=true; shift ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Load environment ──────────────────────────────────────────────────────────
[[ -f .env ]] && { set -a; source .env; set +a; echo "[train] Loaded .env"; } \
              || echo "[train] WARNING: .env not found"

mkdir -p "$LOG_DIR" metrics checkpoints experiments

# ── Step 1: Dataset preparation ───────────────────────────────────────────────
if [[ "$SKIP_PREPARE" == false ]]; then
    echo "━━━ Step 1/3  Dataset Preparation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/01_prepare_dataset.py --log-dir "$LOG_DIR"
else
    echo "[train] Skipping Step 1 (--skip-prepare)"
fi

# ── Step 2: Preprocessing ─────────────────────────────────────────────────────
if [[ "$SKIP_PREPROCESS" == false ]]; then
    echo "━━━ Step 2/3  Planning & Preprocessing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/02_preprocess.py --log-dir "$LOG_DIR"
else
    echo "[train] Skipping Step 2 (--skip-preprocess)"
fi

# ── Step 3: Training ──────────────────────────────────────────────────────────
echo "━━━ Step 3/3  K-Fold Cross-Validation Training ━━━━━━━━━━━━━━━━━━━━━━━"
# shellcheck disable=SC2086
python scripts/03_train.py \
    $FOLDS_ARG \
    $CONFIG_ARG \
    $SEED_ARG \
    $CONTINUE_FLAG \
    --metrics-dir metrics \
    --checkpoints-dir checkpoints \
    --log-dir "$LOG_DIR"

echo ""
echo "━━━ Training complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Checkpoints : checkpoints/"
echo "  Metrics     : metrics/"
echo "  Logs        : $LOG_DIR/"
