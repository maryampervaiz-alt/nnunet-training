# nnU-Net for BraTS 2024 MEN-RT: Meningioma GTV Segmentation

Automated gross tumour volume (GTV) segmentation for post-operative meningioma
radiotherapy planning using nnU-Net v2 with 5-fold cross-validation.
Developed for the **BraTS 2024 MEN-RT** challenge.

---

## Method

- **Architecture**: nnU-Net v2 (`3d_fullres`) with automatic configuration
  (patch size, batch size, network topology inferred from the data)
- **Input**: T1-weighted contrast-enhanced MRI (T1c)
- **Target**: Binary GTV mask
- **Training**: 5-fold cross-validation, 1000 epochs per fold
- **Early stopping**: Custom `nnUNetTrainerEarlyStopping` subclass monitors
  pseudo-Dice per epoch; subprocess-level monitor enforces patience
- **Reproducibility**: Global seed propagated to Python, NumPy, PyTorch, and
  CUDA via environment variables

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA support (see installation note below)
- NVIDIA GPU with ≥ 16 GB VRAM (24 GB recommended for `3d_fullres`)

Install dependencies:

```bash
# 1. Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install all other requirements
pip install -r requirements.txt
```

Or install from the project package definition:

```bash
pip install -e .
```

---

## Repository Structure

```
.
├── configs/
│   ├── dataset.yaml          # Dataset metadata (no hardcoded spacing/shape)
│   └── experiment.yaml       # Training, inference, and evaluation settings
├── scripts/
│   ├── 01_prepare_dataset.py # Convert raw data → nnU-Net raw format
│   ├── 02_preprocess.py      # nnUNetv2_plan_and_preprocess
│   ├── 03_train.py           # 5-fold cross-validation training
│   ├── 04_inference.py       # Batch inference (ensemble or per-fold CV)
│   ├── 05_evaluate.py        # Metric computation + LaTeX table export
│   ├── 06_visualize.py       # Overlays, violin plots, training curves
│   ├── 07_generate_sam_prompts.py # nnU-Net masks → SAM-Med3D prompts
│   ├── run_pipeline.sh       # Full end-to-end pipeline
│   ├── run_training.sh       # Steps 1–3 only
│   ├── run_inference.sh      # Step 4 only
│   └── run_evaluation.sh     # Steps 5–6 only
├── src/
│   ├── data/                 # Dataset conversion, integrity checks, splitting
│   ├── evaluation/           # Metrics, evaluator, cross-fold aggregation
│   ├── inference/            # nnUNetv2_predict wrapper
│   ├── training/             # CV orchestrator, early stopping, checkpoint manager
│   ├── utils/                # Env loading, logging, experiment tracking
│   └── visualization/        # Segmentation overlays, metric plots
├── tests/
│   └── test_metrics.py       # Unit tests for metric functions
├── .env.example              # Environment variable template
├── pyproject.toml
└── requirements.txt
```

---

## Setup

### 1. Clone and configure environment

```bash
git clone <repo-url>
cd nnunet-men-rt

cp .env.example .env
# Edit .env — set PROJECT_ROOT and adjust paths to your data
```

### 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Verify GPU

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))
```

---

## Usage

### Full pipeline (recommended)

```bash
bash scripts/run_pipeline.sh
```

### Step by step

```bash
# Step 1 — Convert raw BraTS data to nnU-Net format
python scripts/01_prepare_dataset.py

# Step 2 — nnU-Net planning and preprocessing
python scripts/02_preprocess.py

# Step 3 — 5-fold cross-validation training
python scripts/03_train.py

# Step 4 — Fold-wise validation inference
python scripts/04_inference.py --cv-mode

# Step 5 — Evaluation (Dice, HD95, NSD, precision, recall)
python scripts/05_evaluate.py --cv-mode --latex

# Step 6 — Visualizations (overlays, violin plots, training curves)
python scripts/06_visualize.py --all --cv-mode

# Step 7 — Build SAM-Med3D prompts from nnU-Net coarse masks
python scripts/07_generate_sam_prompts.py \
  --mask-dir inference_outputs/ensemble \
  --output-dir prompts/sammed3d
```

### SAM-Med3D refinement handoff

For prompt-based refinement (point + box guidance), generate prompt artifacts
from nnU-Net coarse predictions:

```bash
bash scripts/run_prompt_generation.sh \
  --mask-dir inference_outputs/ensemble \
  --output-dir prompts/sammed3d
```

Outputs:
- `prompts/sammed3d/cases/<case_id>.json` (per-case prompt payload)
- `prompts/sammed3d/sam_prompt_manifest.json` (global manifest)
- `prompts/sammed3d/sam_prompt_summary.csv` (case-level summary)

Each case JSON includes:
- Connected-component-aware 3D bounding boxes
- Positive and negative prompt points
- Geometry metadata (spacing/origin/direction)

This keeps the refinement stage reproducible and avoids data leakage from GT.

### Training only

```bash
bash scripts/run_training.sh
# Resume interrupted training:
bash scripts/run_training.sh --continue
```

### Inference only

```bash
bash scripts/run_inference.sh
# Ensemble over all folds:
bash scripts/run_inference.sh --ensemble
```

### Evaluation only

```bash
bash scripts/run_evaluation.sh
```

---

## Configuration

All user-facing settings live in `.env`. Model hyperparameters (learning rate,
patch size, batch size, augmentation) are determined automatically by nnU-Net
and are **not overridden**.

Key `.env` variables:

| Variable | Description | Default |
|---|---|---|
| `nnUNet_raw` | Raw dataset root | `$PROJECT_ROOT/nnunet_raw` |
| `nnUNet_preprocessed` | Preprocessed data root | `$PROJECT_ROOT/nnunet_preprocessed` |
| `nnUNet_results` | Model checkpoint root | `$PROJECT_ROOT/checkpoints` |
| `DATASET_ID` | Integer dataset ID | `001` |
| `NNUNET_CONFIGURATION` | nnU-Net config | `3d_fullres` |
| `NUM_FOLDS` | CV folds | `5` |
| `NNUNET_SEED` | Global random seed | `42` |
| `ES_PATIENCE` | Early stopping patience (epochs) | `50` |
| `CUDA_VISIBLE_DEVICES` | GPU index | `0` |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| DSC | Dice Similarity Coefficient |
| HD95 | 95th-percentile Hausdorff Distance (mm) |
| NSD | Normalised Surface Distance at 2 mm tolerance (BraTS 2024 official) |
| Precision | Positive predictive value |
| Recall | Sensitivity / true positive rate |
| Specificity | True negative rate |
| Vol. Sim. | Volume similarity |
| Abs. Vol. Err. | Absolute volume error (ml) |

Results are saved to `results/` as CSV files and an optional LaTeX table
(`results/cv_results_table.tex`).

---

## Reproducibility

All random seeds are fixed through the following chain:

1. `NNUNET_SEED` (env var, default `42`) is read at startup
2. `set_global_seed()` seeds Python `random`, NumPy, PyTorch CPU+CUDA,
   and sets `cudnn.deterministic = True`
3. The seed is forwarded to every training subprocess via `NNUNET_SEED`
   and `PYTHONHASHSEED` environment variables
4. `nnUNetTrainerEarlyStopping` reads and re-applies the seed at init

To reproduce results with a different seed:

```bash
python scripts/03_train.py --seed 1234
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{brats2024menrt,
  title     = {Meningioma Radiotherapy Target Segmentation with nnU-Net},
  author    = {},
  booktitle = {BraTS 2024 Challenge},
  year      = {2024}
}
```

Dataset citation: see `BraTs MEN-RT/CITATION.bib`.

---

## License

MIT License. See `LICENSE` for details.
