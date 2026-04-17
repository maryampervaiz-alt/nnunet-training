#!/bin/bash
# Resume interrupted nnU-Net training from checkpoint_latest.pth
#
# Usage
# -----
#   bash scripts/resume_training.sh                # Resume fold 0
#   bash scripts/resume_training.sh 1              # Resume fold 1
#   bash scripts/resume_training.sh 0 1 2          # Resume folds 0, 1, 2
#
# The script will:
#   1. Check for existing checkpoint_latest.pth
#   2. Re-run training with the --c (continue) flag
#   3. Use the same hyperparameters as the original training

set -e

FOLDS="${@:-0}"  # Default to fold 0 if no args

DATASET_ID="${DATASET_ID:-001}"
CONFIGURATION="${NNUNET_CONFIGURATION:-3d_fullres}"
TRAINER="${NNUNET_TRAINER_CLASS:-nnUNetTrainerEarlyStopping}"
PLANS="${NNUNET_PLANS_IDENTIFIER:-nnUNetPlans}"
NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-50}"
SEED="${NNUNET_SEED:-42}"

nnUNet_results="${nnUNet_results:-.}"

echo ""
echo "========================================================================="
echo "  Resume Training — nnU-Net BraTS MEN-RT"
echo "========================================================================="
echo ""

for FOLD in $FOLDS; do
    echo "[Fold $FOLD] Checking for checkpoint_latest.pth …"

    CKPT_DIR="$nnUNet_results/Dataset${DATASET_ID}_BraTSMENRT/nnUNetTrainerEarlyStopping__${PLANS}__${CONFIGURATION}/fold_${FOLD}"
    LATEST_CKPT="$CKPT_DIR/checkpoint_latest.pth"

    if [ ! -f "$LATEST_CKPT" ]; then
        echo "[Fold $FOLD] ❌ No checkpoint found: $LATEST_CKPT"
        echo "            (Training may not have started, or different path)"
        continue
    fi

    echo "[Fold $FOLD] ✓ Found checkpoint: $LATEST_CKPT"
    echo "[Fold $FOLD] Resuming training with --c flag …"
    echo ""

    nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" \
        -tr "$TRAINER" \
        -p "$PLANS" \
        --num_epochs "$NUM_EPOCHS" \
        --c

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[Fold $FOLD] ✅ Training resumed and completed"
    else
        echo ""
        echo "[Fold $FOLD] ⚠️  Training exited with code $EXIT_CODE"
        exit $EXIT_CODE
    fi
done

echo ""
echo "========================================================================="
echo "  Resume complete. Check logs and checkpoints for details."
echo "========================================================================="
