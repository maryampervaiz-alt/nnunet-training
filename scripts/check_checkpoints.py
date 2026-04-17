#!/usr/bin/env python3
"""Checkpoint validation script.

Verifies that training saved checkpoints correctly for all folds.

Usage
-----
::

    python scripts/check_checkpoints.py
    python scripts/check_checkpoints.py --fold 0
    python scripts/check_checkpoints.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.checkpoint_manager import CheckpointManager
from src.utils.env_utils import load_env, get_env
from src.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Specific fold(s) to check (default: all)",
    )
    p.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Root checkpoint directory",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )
    p.add_argument("--log-dir", default="logs", help="Log directory")
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    log = get_logger(name="check_checkpoints", log_dir=args.log_dir)

    ckpt_mgr = CheckpointManager(root=args.checkpoints_dir)
    num_folds = int(get_env("NUM_FOLDS", default="5"))
    folds_to_check = args.folds if args.folds else list(range(num_folds))

    log.info("=" * 70)
    log.info("  Checkpoint Validation")
    log.info("=" * 70)
    log.info(f"  Checkpoint root : {args.checkpoints_dir}")
    log.info(f"  Folds to check  : {folds_to_check}")
    log.info("=" * 70)

    available = ckpt_mgr.list_available()
    if not available:
        log.warning("No checkpoints found.")
        return

    passed = 0
    failed = 0
    issues: list[str] = []

    for fold in folds_to_check:
        fold_key = f"fold_{fold}"
        if fold_key not in available:
            log.warning(f"[fold_{fold}] MISSING: no checkpoint directory")
            failed += 1
            issues.append(f"Fold {fold}: no checkpoint directory")
            continue

        files = available[fold_key]
        fold_passed = True

        # Check mandatory files
        mandatory = ["best_model.pth", "last_model.pth"]
        for fname in mandatory:
            if fname not in files:
                log.warning(f"[fold_{fold}] MISSING: {fname}")
                fold_passed = False
                issues.append(f"Fold {fold}: missing {fname}")
            elif args.verbose:
                file_path = Path(args.checkpoints_dir) / fold_key / fname
                size_mb = file_path.stat().st_size / 1e6
                log.info(f"[fold_{fold}] OK: {fname} ({size_mb:.1f} MB)")

        # Check metadata
        if "metadata.json" in files:
            meta_path = Path(args.checkpoints_dir) / fold_key / "metadata.json"
            try:
                with meta_path.open() as fh:
                    meta = json.load(fh)
                if "best_val_dice" in meta and args.verbose:
                    log.info(f"[fold_{fold}]     best_val_dice: {meta['best_val_dice']:.4f}")
            except Exception as exc:
                log.warning(f"[fold_{fold}] metadata.json unreadable: {exc}")
                fold_passed = False
                issues.append(f"Fold {fold}: metadata.json error")
        else:
            log.warning(f"[fold_{fold}] MISSING: metadata.json")
            fold_passed = False
            issues.append(f"Fold {fold}: missing metadata.json")

        if fold_passed:
            log.success(f"[fold_{fold}] PASS ✓")
            passed += 1
        else:
            failed += 1

    # Check global best
    if "global_best" in available:
        gb_files = available["global_best"]
        if "best_model.pth" in gb_files:
            log.success("[global_best] PASS ✓")
            passed += 1
        else:
            log.warning("[global_best] MISSING: best_model.pth")
            failed += 1
            issues.append("Global best: missing best_model.pth")
    else:
        log.info("[global_best] not yet available (run full training)")

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info(f"  RESULTS: {passed} passed, {failed} failed")
    log.info("=" * 70)

    if issues:
        log.error("Issues found:")
        for issue in issues:
            log.error(f"  •  {issue}")
        sys.exit(1)
    else:
        log.success("All checkpoints validated successfully!")


if __name__ == "__main__":
    main()
