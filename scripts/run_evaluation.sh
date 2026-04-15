#!/usr/bin/env bash
# Run Steps 5–6: metric evaluation and visualization.
#
# Prerequisites: inference_outputs/ must exist (run run_inference.sh first).
#
# Usage:
#   bash scripts/run_evaluation.sh [options]
#
# Options:
#   --cv              Evaluate per-fold CV predictions (default when cv outputs exist)
#   --pred DIR        Prediction directory (overrides default)
#   --gt DIR          Ground-truth label directory (overrides default labelsTr)
#   --latex           Export LaTeX results table to results/
#   --no-viz          Skip visualization step
#   --top-n N         Cases shown in best/worst rankings (default: 10)
#
# Examples:
#   bash scripts/run_evaluation.sh --cv --latex
#   bash scripts/run_evaluation.sh --pred inference_outputs/ensemble

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CV_MODE=false
PRED_ARG=""
GT_ARG=""
LATEX_FLAG=""
SKIP_VIZ=false
TOP_N=10
LOG_DIR="logs"
RESULTS_DIR="results"

# Auto-detect CV mode if cv output directory exists
[[ -d "inference_outputs/cv" ]] && CV_MODE=true

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cv)       CV_MODE=true; shift ;;
        --pred)     PRED_ARG="--pred $2"; shift 2 ;;
        --gt)       GT_ARG="--gt $2"; shift 2 ;;
        --latex)    LATEX_FLAG="--latex"; shift ;;
        --no-viz)   SKIP_VIZ=true; shift ;;
        --top-n)    TOP_N="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Load environment ──────────────────────────────────────────────────────────
[[ -f .env ]] && { set -a; source .env; set +a; echo "[evaluate] Loaded .env"; } \
              || echo "[evaluate] WARNING: .env not found"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" visualizations

# ── Step 5: Evaluation ────────────────────────────────────────────────────────
echo "━━━ Step 5  Evaluation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EVAL_CMD=(python scripts/05_evaluate.py
    --results-dir "$RESULTS_DIR"
    --show-rankings
    --top-n "$TOP_N"
    --log-dir "$LOG_DIR"
)

[[ "$CV_MODE" == true ]] && EVAL_CMD+=(--cv-mode)
[[ -n "$PRED_ARG" ]]     && EVAL_CMD+=($PRED_ARG)
[[ -n "$GT_ARG" ]]       && EVAL_CMD+=($GT_ARG)
[[ -n "$LATEX_FLAG" ]]   && EVAL_CMD+=($LATEX_FLAG)

"${EVAL_CMD[@]}"

# ── Step 6: Visualization ─────────────────────────────────────────────────────
if [[ "$SKIP_VIZ" == false ]]; then
    echo "━━━ Step 6  Visualization ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    VIZ_CMD=(python scripts/06_visualize.py
        --all
        --results-dir "$RESULTS_DIR"
        --output-dir visualizations
        --log-dir "$LOG_DIR"
    )

    if [[ "$CV_MODE" == true ]]; then
        VIZ_CMD+=(--cv-mode)
    fi
    [[ -n "$PRED_ARG" ]] && VIZ_CMD+=(--pred-dir "${PRED_ARG#--pred }")

    "${VIZ_CMD[@]}"
fi

echo ""
echo "━━━ Evaluation complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results         : $RESULTS_DIR/"
[[ -n "$LATEX_FLAG" ]] && echo "  LaTeX table     : $RESULTS_DIR/cv_results_table.tex"
echo "  Visualizations  : visualizations/"
echo "  Logs            : $LOG_DIR/"
