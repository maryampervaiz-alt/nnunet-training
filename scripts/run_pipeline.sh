#!/usr/bin/env bash
# Full end-to-end nnU-Net MEN-RT pipeline (Steps 1–6).
#
# Usage:
#   bash scripts/run_pipeline.sh [options]
#
# Options:
#   --skip-prepare      Skip Step 1 (data already in nnU-Net raw format)
#   --skip-preprocess   Skip Step 2 (already preprocessed)
#   --continue          Resume interrupted training from checkpoints
#   --folds N [N ...]   Train specific folds only (default: all 5)
#   --config NAME       nnU-Net configuration (default: 3d_fullres)
#   --seed INT          Global random seed (default: 42 or $NNUNET_SEED)
#   --latex             Export LaTeX results table after evaluation
#   --no-viz            Skip visualization step
#
# Examples:
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh --skip-prepare --skip-preprocess --folds 0 1
#   bash scripts/run_pipeline.sh --seed 1234 --latex

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SKIP_PREPARE=false
SKIP_PREPROCESS=false
CONTINUE_FLAG=""
FOLDS_ARG=""
CONFIG_ARG=""
SEED_ARG=""
LATEX_FLAG=""
SKIP_VIZ=false
LOG_DIR="logs"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-prepare)    SKIP_PREPARE=true; shift ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --continue)        CONTINUE_FLAG="--continue-training"; shift ;;
        --folds)           shift; FOLDS_ARG="--folds"; while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do FOLDS_ARG="$FOLDS_ARG $1"; shift; done ;;
        --config)          CONFIG_ARG="--configuration $2"; shift 2 ;;
        --seed)            SEED_ARG="--seed $2"; shift 2 ;;
        --latex)           LATEX_FLAG="--latex"; shift ;;
        --no-viz)          SKIP_VIZ=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Load environment ──────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    set -a; source .env; set +a
    echo "[pipeline] Loaded .env"
else
    echo "[pipeline] WARNING: .env not found. Ensure env vars are set manually."
fi

mkdir -p "$LOG_DIR" metrics results visualizations inference_outputs checkpoints experiments

# ── Step 1: Dataset preparation ───────────────────────────────────────────────
if [[ "$SKIP_PREPARE" == false ]]; then
    echo "━━━ Step 1/6  Dataset Preparation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/01_prepare_dataset.py --log-dir "$LOG_DIR"
else
    echo "[pipeline] Skipping Step 1 (--skip-prepare)"
fi

# ── Step 2: Preprocessing ─────────────────────────────────────────────────────
if [[ "$SKIP_PREPROCESS" == false ]]; then
    echo "━━━ Step 2/6  Planning & Preprocessing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/02_preprocess.py --log-dir "$LOG_DIR"
else
    echo "[pipeline] Skipping Step 2 (--skip-preprocess)"
fi

# ── Step 3: Training ──────────────────────────────────────────────────────────
echo "━━━ Step 3/6  K-Fold Cross-Validation Training ━━━━━━━━━━━━━━━━━━━━━━━"
# shellcheck disable=SC2086
python scripts/03_train.py \
    $FOLDS_ARG \
    $CONFIG_ARG \
    $SEED_ARG \
    $CONTINUE_FLAG \
    --metrics-dir metrics \
    --checkpoints-dir checkpoints \
    --log-dir "$LOG_DIR"

# ── Step 4: Fold-wise CV inference ────────────────────────────────────────────
echo "━━━ Step 4/6  Fold-Wise Validation Inference ━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/04_inference.py \
    --cv-mode \
    --cv-output-root inference_outputs/cv \
    --log-dir "$LOG_DIR"

# ── Step 5: Evaluation ────────────────────────────────────────────────────────
echo "━━━ Step 5/6  Cross-Validation Evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# shellcheck disable=SC2086
python scripts/05_evaluate.py \
    --cv-mode \
    --results-dir results \
    --show-rankings \
    $LATEX_FLAG \
    --log-dir "$LOG_DIR"

# ── Step 6: Visualization ─────────────────────────────────────────────────────
if [[ "$SKIP_VIZ" == false ]]; then
    echo "━━━ Step 6/6  Visualization ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/06_visualize.py \
        --all \
        --cv-mode \
        --results-dir results \
        --output-dir visualizations \
        --log-dir "$LOG_DIR"
else
    echo "[pipeline] Skipping Step 6 (--no-viz)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Pipeline complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Predictions    : inference_outputs/cv/"
echo "  Results / CSV  : results/"
[[ -n "$LATEX_FLAG" ]] && echo "  LaTeX table    : results/cv_results_table.tex"
echo "  Visualizations : visualizations/"
echo "  Checkpoints    : checkpoints/"
echo "  Metrics        : metrics/"
echo "  Logs           : $LOG_DIR/"
