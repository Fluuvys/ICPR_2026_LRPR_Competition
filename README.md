# ICPR 2026 LR-LPR: Spatio-Temporal SVTR & Ensembling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Competition](https://img.shields.io/badge/Competition-ICPR%202026%20LR--LPR-orange.svg)](https://codalab.lisn.upsaclay.fr/)

Official repository for our solution to the **[ICPR 2026 Competition on Low-Resolution License Plate Recognition (LR-LPR)](https://codalab.lisn.upsaclay.fr/)**.

This competition challenges participants to recognize license plates from heavily degraded, low-resolution dashcam footage — a problem where even state-of-the-art methods struggle to exceed **50–60% accuracy**. Our pipeline ensembles 4 models including Restrans, Mamba, and 2 SVTR variants.
---

## 👥 Team Members

* **Hoang Minh Giang Nguyen** — *Hanoi University of Science and Technology (HUST)*
* **Trong Thai Doan** — *Hanoi University of Science and Technology (HUST)*
* **Dinh Quang Trinh** — *Hanoi University of Science and Technology (HUST)*
* **Tuan Anh Duong** — *Hanoi University of Science and Technology (HUST)*
* **Tuan Anh Vu** — *Hanoi University of Science and Technology (HUST)*

---

## 🏆 Competition Results

![Leaderboard Results](Screenshot%202026-03-04%20at%2009.11.27.png)

| Metric | Score |
|--------|-------|
| **Best Single Model — Test Acc** | `74.00%` (Factorized SVTR 256-Ch) |
| **Ensemble — Test Acc** | `79.23%` |

---

## 🏅 About the Competition

The **ICPR 2026 LR-LPR** competition addresses license plate recognition under authentic, real-world surveillance conditions where images are captured at low resolution or subjected to heavy compression.

| | |
|---|---|
| **Platform** | [Codabench](https://codalab.lisn.upsaclay.fr/) |
| **Primary Metric** | Recognition Rate (Exact Match) |
| **Tie-Breaker** | Confidence Gap (correct vs. incorrect prediction confidence) |
| **Submission Deadline** | March 1, 2026 |
| **Conference** | ICPR 2026, August 2026 |

### Dataset

Each **track** consists of **5 consecutive Low-Resolution (LR) frames** of the same license plate. The training set also provides corresponding High-Resolution (HR) frames to enable super-resolution exploration.

* **Scenario A** — 10,000 tracks, controlled conditions (daylight, no rain)
* **Scenario B** — 10,000 tracks, diverse real-world conditions (varied weather, vehicle types)
* **Public Test Set** — 1,000 tracks from Scenario B (for leaderboard reference)
* **Blind Test Set** — 3,000+ tracks from Scenario B (official final ranking)

Predictions must be made using **only the 5 LR images** per track.

### Submission Format

```
track_00001,ABC1234;0.9876
track_00002,DEF5678;0.6789
track_00003,GHI9012;0.4521
```
Each line: `track_id,plate_text;confidence`

---

## ✨ Key Architectural Features

* **Factorized Temporal Attention:** Replaces standard global attention with a mathematically isolated temporal transformer. Dynamically weights the sharpest frames per spatial slice, reducing complexity from $O((F \times W)^2)$ to $O(F^2)$ and preventing blur-averaging across frames.
* **Spatio-Temporal SVTR Backbone:** A widened 256-channel patch-embedding Vision Transformer that explicitly models horizontal character-to-character relationships.
* **Optically Accurate Augmentation:** Training pipeline using `albumentations` that locks spatial geometry across all 5 frames (preventing artificial "frame earthquakes") and applies optical degradations *prior* to downscaling to accurately simulate physical dashcam lens behaviour.
* **Logit-Level Ensembling:** Extracts raw CTC probability distributions (`log_softmax`) for advanced multi-model ensembling and regex-constrained Beam Search decoding.

---

## 📂 Repository Structure

```text
ICPR_2026_LRPR_Competition/
├── configs/               # Global configuration and hyperparameter files
├── data/                  # Placeholder for train/val/test datasets
├── src/
│   ├── data/              # Dataloaders, multi-frame padding, and augmentations
│   ├── models/            # SVTR, ResTran, Mamba, and Factorized Fusion blocks
│   ├── training/          # Universal trainer and validation loops
│   └── utils/             # CTC decoding and metric calculation
├── ensemble.py            # Logit aggregation and Beam Search decoding script
├── inference.py           # Standalone script for Codabench submission generation
└── train.py               # Main training entry point
```

---

## 🚀 Installation & Setup

**1. Clone the repository:**

```bash
git clone https://github.com/Fluuvys/ICPR_2026_LRPR_Competition.git
cd ICPR_2026_LRPR_Competition
```

**2. Install dependencies:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python albumentations tqdm
```

**3. Data Preparation:**

> ⚠️ Dataset access requires signing a license agreement and registering on Codabench with an institutional email. See the [competition page](https://codalab.lisn.upsaclay.fr/) for instructions.

Place the competition data inside the `data/` directory. The blind test set path must match the value in `configs/config.py`.

---

## 💻 Usage

### Training a Single Model

```bash
python train.py --model new_svtr --experiment-name factorized_svtr_256 --epochs 60
```

*Append `--full-train` to train on 100% of the data for final Codabench submission weights.*

### Standalone Inference (Greedy Decoding)

```bash
python inference.py --model new_svtr --weights results/best_model.pth --output submission.txt
```

### Logit Extraction & Ensembling

```bash
# Step 1: Extract logits from multiple model runs
python inference_logits.py --weights results/model_1.pth --output_name svtr_run1
python inference_logits.py --weights results/model_2.pth --output_name svtr_run2

# Step 2: Fuse logits and decode with Beam Search
python ensemble.py \
  --models results/logits/svtr_run1_logits.npy results/logits/svtr_run2_logits.npy \
  --output final_ensemble_submission.txt
```

---

## 🙏 Acknowledgements

Developed for the **ICPR 2026 Low-Resolution License Plate Recognition Track**. Special thanks to the competition organizers — Rayson Laroca (PUCPR/UFPR), Valfride Nascimento (UFPR), and David Menotti (UFPR) — for curating the challenging multi-frame dashcam dataset and hosting the competition on Codabench.

### Selected References

* V. Nascimento et al., "Toward Advancing License Plate Super-Resolution in Real-World Scenarios," *Journal of the Brazilian Computer Society*, 2025.
* K. Na et al., "MF-LPR2: Multi-frame License Plate Image Restoration and Recognition Using Optical Flow," *CVIU*, 2025.
* R. Laroca et al., "Leveraging Model Fusion for Improved License Plate Recognition," *CIARP*, 2023.
