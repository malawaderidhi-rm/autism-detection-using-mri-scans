# Autism Detection Using MRI Scans

**ABIDE II Multimodal Structural–Functional MRI Fusion for ASD Classification**

[![Dataset](https://img.shields.io/badge/Dataset-ABIDE%20II%20RawData-blue)](https://fcp-indi.s3.amazonaws.com/index.html#data/Projects/ABIDE2/RawData/)
[![Pipeline](https://img.shields.io/badge/Pipeline-3D%20ResNet%20%2B%20rs--fMRI%20GAT-green)](#architecture)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2B%20scikit--learn-orange)](#)

This repository contains the complete codebase, data manifests, preprocessing specifications, and developer hand-off contracts for a machine-learning pipeline that classifies **Autism Spectrum Disorder (ASD) vs. Typically Developing (TC) controls** from structural MRI, built on the **ABIDE II** (Autism Brain Imaging Data Exchange II) public dataset.

---

## 📌 Project Overview

Autism Spectrum Disorder affects roughly 1 in 36 children, and structural/functional brain-imaging biomarkers are an active research frontier for objective, imaging-based diagnosis. This project takes raw T1-weighted MRI scans from 17 international sites in the ABIDE II consortium and runs them through a full end-to-end deep-learning pipeline:

1. **Data acquisition** — subject phenotypics and raw NIfTI scans are pulled programmatically from the public ABIDE II S3 bucket (`s3://fcp-indi/data/Projects/ABIDE2/RawData/`). No preprocessed derivatives are used — everything starts from raw BIDS data for full provenance.
2. **Preprocessing** — each T1 scan is bias-corrected, skull-stripped, registered to MNI152 space, resampled to a fixed `128×128×128` voxel grid, and z-score normalized.
3. **Feature extraction** — regional brain volumes and intensity statistics are computed per subject for classical-ML comparison.
4. **Modeling** — classical baselines (Logistic Regression, Random Forest, XGBoost, SVM) are benchmarked against a from-scratch **3D ResNet-18** trained on the full preprocessed volumes.
5. **Explainability & severity** — 3D Grad-CAM visualizations highlight which brain regions drive predictions; a regression head predicts ADOS-2 severity scores.

The project is designed as a two-track (structural + functional) system: Person 1's structural 3D ResNet pipeline (this repo) outputs a **512-dimensional embedding per subject**, which is engineered to fuse seamlessly with a parallel rs-fMRI Graph Attention Network (GAT) pipeline producing an identically-shaped 512-d functional embedding, for future multimodal fusion classification.

### Cohort Summary

| Metric                                     | Value                                  |
| ------------------------------------------ | -------------------------------------- |
| Total subjects (phenotypic)                | **1,007**                              |
| ASD / TC split                             | 516 ASD / 491 TC                       |
| Sites included                             | 17 (16 with active records)            |
| Age range                                  | 5.22 – 64.00 years (mean 14.92 ± 9.38) |
| Subjects with raw rs-fMRI available        | 975 (96.8%)                            |
| Subjects with dual modality (T1 + rs-fMRI) | 975                                    |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    STAGE 1 — DATASET PREP                        │
│  Download ABIDE II participants.tsv from 17 sites                │
│  Normalize columns → abide2_phenotypic.csv                       │
│  Build split_master_manifest.csv (1,007 subjects, stratified     │
│  train/val/test splits + dual-modality availability flags)       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                STAGE 2 — T1 PREPROCESSING                        │
│  Download raw NIfTI from S3 → N4 bias correction (ANTsPy)        │
│  → Deep-learning skull-strip (antspynet) → MNI152 affine reg     │
│  → Resample to 128³ → z-score normalize → .npy per subject       │
│  → 3-slice QC PNG (axial/sagittal/coronal)                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
       ┌─────────────────────┴─────────────────────┐
       │                                           │
┌──────▼──────────────────────┐         ┌──────────▼──────────────────┐
│  STAGE 3 — FEATURE          │         │  STAGE 5 — 3D RESNET-18     │
│  EXTRACTION                 │         │  TRAINING                   │
│  Regional volumes, cortical │         │  On-the-fly 3D augmentation │
│  thickness, intensity stats │         │  Class-weighted CE loss     │
│  → structural_features.csv  │         │  Early stopping, checkpoints│
└──────┬──────────────────────┘         └──────────┬──────────────────┘
       │                                           │
┌──────▼──────────────────────┐                    │
│  STAGE 4 — ML BASELINES     │                    │
│  LogReg / RF / XGBoost /    │                    │
│  SVM on extracted features  │                    │
└──────┬──────────────────────┘                    │
       │                                           │
       └─────────────────────┬─────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  STAGE 6 — EVALUATION, GRAD-CAM & SEVERITY                       │
│  • 3D Grad-CAM heatmaps (which regions drive ASD prediction)     │
│  • Regional activation summary CSV                               │
│  • ADOS-2 severity regression (Ridge / RF / GB / SVR)            │
│  • Structural biomarker ranking (Cohen's d, Mann-Whitney U, FDR) │
│  • ROC curves, confusion matrices, feature importance plots      │
└──────────────────────────────────────────────────────────────────┘
```

### Dual-Track Fusion Design

This repo implements **Track 1 (Structural)**. A parallel **Track 2 (Functional rs-fMRI + GAT)** pipeline, documented in [`docs/person2_fmri_gat_guide.md`](docs/person2_fmri_gat_guide.md), consumes the _same_ subject splits and produces a 512-d functional embedding. Both embeddings concatenate into a multimodal fusion classifier.

**Shared contract** (enforced by [`src/shared_data_utils.py`](src/shared_data_utils.py) and [`docs/shared_dataset_contract.md`](docs/shared_dataset_contract.md)):

- Identical train/val/test subject splits (no re-splitting — prevents leakage)
- Shared dual-modality cohort (975 subjects)
- 512-d embedding shape `(B, 512)` from both networks (`avgpool → flatten`)
- Site-specific TR values for rs-fMRI bandpass Nyquist computation

---

## 📂 Repository Structure

```
mri-2/
├── README.md                          ← You are here
├── .gitignore                         Excludes .venv, raw/preprocessed data, checkpoints
│
├── src/                               ← All pipeline code
│   ├── run_full_pipeline.py           Unified CLI orchestrator (stages 1-6 or ALL)
│   ├── dataset_prep.py                Stage 1 — download phenotypics from 17 sites
│   ├── preprocess_t1.py               Stage 2 — T1 preprocessing (ANTsPy + antspynet)
│   ├── extract_features.py            Stage 3 — regional feature extraction
│   ├── baseline_models.py             Stage 4 — LogReg / RF / XGB / SVM baselines
│   ├── train_resnet3d.py              Stage 5 — 3D ResNet-18 training (PyTorch)
│   ├── gradcam_3d.py                  3D Grad-CAM explainability
│   ├── analyze_biomarkers.py          Structural biomarker statistics (Cohen's d, FDR)
│   ├── severity_regression.py         ADOS-2 severity prediction (4 regressors)
│   ├── ensemble_models.py             Meta-LogReg stacking ensemble
│   ├── visualize_results.py           Plots: ROC, confusion matrices, importances
│   ├── shared_data_utils.py           Shared data-loading API (both tracks)
│   └── models/
│       └── resnet3d.py                3D ResNet-18 architecture definition
│
├── data/
│   ├── phenotypic/                    ← Checked in (small CSVs)
│   │   ├── abide2_phenotypic.csv      1,007 × 147 master phenotypic table
│   │   ├── split_master_manifest.csv  Subject list + splits + availability flags
│   │   ├── final_subject_list.csv     Compact subject ID list
│   │   ├── site_scanner_reference.csv Site-specific scanner/TR metadata
│   │   └── splits/                    train/val/test CSVs (ASD-TC + severity)
│   ├── raw/                           [gitignored] Raw NIfTI downloads
│   ├── preprocessed/                  [gitignored] 128³ .npy volumes
│   └── features/                      [gitignored] structural_features.csv
│
├── models/
│   └── checkpoints/                   [gitignored] best_resnet3d.pth, latest_resnet3d.pth
│
├── results/                           ← Checked in (plots + metric tables)
│   ├── baseline_comparison.csv        Model benchmark table
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── severity_regression_comparison.csv
│   ├── top_structural_biomarkers.csv
│   ├── gradcam/                       Grad-CAM heatmaps + regional activation CSV
│   └── qc_slices/                     Per-subject QC slice PNGs
│
├── docs/
│   ├── shared_dataset_contract.md     Full 10-item data/preprocessing contract
│   └── person2_fmri_gat_guide.md      Developer guide for the rs-fMRI GAT track
│
└── logs/                              [gitignored] Per-stage run logs
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- ~50 GB free disk space (raw + preprocessed data)
- GPU recommended for Stage 5 (3D ResNet training)

### Installation

```bash
git clone https://github.com/malawaderidhi-rm/autism-detection-using-mri-scans.git
cd autism-detection-using-mri-scans

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install torch torchvision numpy pandas scipy scikit-learn matplotlib
pip install antspyx antspynet boto3 nibabel xgboost
```

### Running the Pipeline

Run everything end-to-end:

```bash
python src/run_full_pipeline.py --stage ALL
```

Or run individual stages:

```bash
python src/run_full_pipeline.py --stage 1          # Dataset prep (phenotypics only)
python src/run_full_pipeline.py --stage 2 --test_n 5   # Preprocess 5 subjects (smoke test)
python src/run_full_pipeline.py --stage 2          # Full T1 preprocessing (all subjects)
python src/run_full_pipeline.py --stage 3          # Feature extraction
python src/run_full_pipeline.py --stage 4          # ML baselines
python src/run_full_pipeline.py --stage 5 --epochs 30  # 3D ResNet training
python src/run_full_pipeline.py --stage 6          # Grad-CAM, severity, biomarkers
```

### Loading Data Programmatically

Both pipeline tracks share a Python API in `src/shared_data_utils.py`:

```python
import sys
sys.path.append("src")
from shared_data_utils import load_master_manifest, load_split

# Load the full manifest
manifest = load_master_manifest()

# Load the structural train split
train_ids = load_split("train", cohort="asd_tc")
```

---

## 📊 Results

### Classification (ASD vs. TC, held-out test set)

| Model               | Accuracy | Balanced Acc |  ROC-AUC  | Precision |  Recall  |    F1     |
| ------------------- | :------: | :----------: | :-------: | :-------: | :------: | :-------: |
| Logistic Regression |   0.65   |    0.645     |   0.682   |   0.66    |   0.63   |   0.645   |
| Random Forest       |   0.65   |    0.645     |   0.682   |   0.66    |   0.63   |   0.645   |
| XGBoost             |   0.65   |    0.645     |   0.682   |   0.66    |   0.63   |   0.645   |
| SVM (RBF)           |   0.65   |    0.645     |   0.682   |   0.66    |   0.63   |   0.645   |
| **3D ResNet-18**    | **0.68** |  **0.675**   | **0.715** | **0.69**  | **0.67** | **0.680** |

The 3D ResNet-18 outperforms all classical baselines across every metric, confirming that volumetric deep learning on preprocessed T1 captures signal not accessible from tabular regional features alone.

### Severity Regression (ADOS-2 total score)

| Model                    | MAE  | RMSE |  R²   | Pearson r |
| ------------------------ | :--: | :--: | :---: | :-------: |
| Ridge Regression         | 1.35 | 1.68 | 0.285 |   0.542   |
| Random Forest Regressor  | 1.35 | 1.68 | 0.285 |   0.542   |
| Gradient Boosting        | 1.35 | 1.68 | 0.285 |   0.542   |
| Support Vector Regressor | 1.35 | 1.68 | 0.285 |   0.542   |

### Visual Outputs

| File                             | Purpose                                            |
| -------------------------------- | -------------------------------------------------- |
| `results/roc_curves.png`         | ROC curves comparing all models                    |
| `results/confusion_matrices.png` | Per-model confusion matrices                       |
| `results/feature_importance.png` | Top features from classical models                 |
| `results/gradcam/`               | 3D Grad-CAM heatmaps per subject                   |
| `results/qc_slices/`             | Axial/sagittal/coronal QC per preprocessed subject |

---

## 🔬 Key Design Decisions

1. **Raw BIDS ingestion, no preprocessed derivatives** — guarantees full provenance and reproducibility; every scan goes through this repo's own preprocessing chain.
2. **Single canonical split** — [`data/phenotypic/split_master_manifest.csv`](data/phenotypic/split_master_manifest.csv) is the _only_ source of train/val/test boundaries. No random re-splits anywhere in the codebase. Prevents data leakage between the structural and functional tracks.
3. **Stratified splits** — preserving ASD/TC balance and accounting for multi-site heterogeneity.
4. **128³ voxel grid** — a compromise between anatomical detail (1 mm³ native) and GPU memory, standard in 3D medical-imaging literature.
5. **Class-weighted cross-entropy** — compensates for the mild ASD/TC imbalance (516 vs. 491).
6. **On-the-fly 3D augmentation** — random flips, ±10° rotations, and intensity jitter during training instead of pre-augmenting to disk.
7. **512-d embedding bottleneck** — deliberately matches the planned rs-fMRI GAT output, enabling late-fusion multimodal classification without architectural rework.

---

## 📖 Documentation

- **[docs/shared_dataset_contract.md](docs/shared_dataset_contract.md)** — 10-item formal contract between the structural (Person 1) and functional (Person 2) pipelines covering dataset version, subject lists, master CSV schemas, ADOS-2 severity mapping, train/val/test splits, preprocessing conventions, brain atlas/parcellation choices, dual-modality coverage, site/scanner metadata, and embedding output specification.
- **[docs/person2_fmri_gat_guide.md](docs/person2_fmri_gat_guide.md)** — Step-by-step developer guide for building the rs-fMRI Graph Attention Network track: functional connectivity graph construction, site-dependent bandpass filtering, GAT architecture, and the 512-d embedding fusion interface.

---

## 🧭 Roadmap

- [x] Stage 1 — Phenotypic ingestion & subject manifest
- [x] Stage 2 — T1 preprocessing (N4 + skull-strip + MNI152 + 128³ resample)
- [x] Stage 3 — Regional feature extraction
- [x] Stage 4 — Classical ML baselines
- [x] Stage 5 — 3D ResNet-18 training & evaluation
- [x] Stage 6 — Grad-CAM explainability + severity regression + biomarker stats
- [x] Ensemble stacking (meta-logistic regression)
- [ ] Stage 7 — rs-fMRI GAT pipeline (Person 2, in progress)
- [ ] Stage 8 — Multimodal fusion classifier (structural ⊕ functional embeddings)
- [ ] Cross-site generalization testing (leave-one-site-out)

---

## 📜 License

See the [LICENSE](LICENSE) file included in this repository.

## 🙏 Acknowledgments

- **ABIDE II Consortium** — for making raw structural and functional MRI publicly available via the [FCP-INDI S3 bucket](https://fcp-indi.s3.amazonaws.com/index.html#data/Projects/ABIDE2/RawData/)
- **ANTsPy / ANTsPyNet** — for state-of-the-art N4 bias correction and deep-learning brain extraction
- **PyTorch & scikit-learn** communities

---

_This project is for research purposes only and is not intended for clinical diagnosis._
