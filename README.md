# Uncertainty-Aware BEV Perception
## HD Map Quality · Adverse-Weather Calibration · Real-Time Edge Deployment

> *Can lightweight epistemic uncertainty estimation flag low-confidence BEV regions
> caused by weather-induced sensor degradation — enabling automated HD map quality
> control and safer adverse-condition autonomy — without prohibitive inference latency
> on vehicle-class hardware?*

[![arXiv](https://img.shields.io/badge/arXiv-coming%20Week%2014-b31b1b.svg)](#)
[![Docker](https://img.shields.io/badge/docker-compose%20up-2496ED.svg?logo=docker)](./docker-compose.yml)
[![GeoJSON](https://img.shields.io/badge/output-GeoJSON%20map%20features-27ae60.svg)](./outputs/lane_map_sample.geojson)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## What this project builds

A unified BEV perception pipeline that produces **lane segmentation predictions**
alongside **per-pixel epistemic uncertainty maps** via Monte Carlo Dropout, then
exports them as **confidence-gated GeoJSON lane features** for HD map quality
control and flags low-confidence regions caused by weather-induced sensor degradation.

Built as a single codebase targeting two markets simultaneously:
- 🇳🇱 **Netherlands (TomTom / HERE):** GeoJSON export with `mean_variance` per lane
  feature enables automated re-survey prioritisation without per-frame human review.
- 🇸🇪 **Scandinavia (Einride / Zenseact):** CADC winter evaluation reveals a
  calibration failure — model confidence stays high while mIoU collapses under snow —
  the dangerous failure mode for adverse-condition autonomy.

---

## Results

| Experiment | Metric | Value |
|---|---|---|
| nuScenes val lane segmentation | mIoU | — |
| CADC winter (no retraining) | mIoU | — |
| Weather degradation delta | ΔmIoU | — |
| ECE calibration gap (clear vs. winter) | ECE | — |
| TensorRT FP16 throughput gain | ms/frame | — |

*Numbers populate as experiments complete. Results are measured, not projected.*

---

## Repository structure

av-perception-portfolio/
├── src/
│   ├── geometry/        # K matrix, projection, homography, IPM
│   ├── models/          # U-Net, temporal fusion head
│   ├── data/            # nuScenes loader, CADC loader
│   ├── augmentation/    # Poisson snow injection, Koschmieder fog model
│   ├── uncertainty/     # MC Dropout inference, variance maps
│   ├── export/          # GeoJSON export, confidence-gated flagging
│   ├── deployment/      # TensorRT pipeline, ONNX export, latency profiler
│   └── utils/           # logger.py, file_utils.py — shared, never duplicated
├── configs/
│   └── default.yaml
├── ros2_ws/             # ROS2 C++ inference node (Week 13)
├── paper/
│   └── main.tex         # LaTeX draft, opens Week 8, arXiv by Week 14
├── assets/              # Output GIFs and images for README display
├── outputs/             # Gitignored — never committed
├── notebooks/           # Exploratory visualisation only, not primary deliverables
├── docker-compose.yml
├── requirements.txt
└── README.md


---

## Build progress

| Module | Status | Week |
|---|---|---|
| Repo scaffold + engineering config | ✅ Complete | Wk 1 |
| Camera geometry (K matrix, extrinsic T, IPM) | ✅ Complete | Wk 1 |
| PyTorch U-Net from scratch (BCE + Dice) | 🔄 In progress | Wk 2 |
| nuScenes LiDAR-to-camera projection | ⬜ Upcoming | Wk 3 |
| GeoJSON export script | ⬜ Upcoming | Wk 3 |
| Poisson snowflake noise injection | ⬜ Upcoming | Wk 5 |
| Koschmieder fog model | ⬜ Upcoming | Wk 5 |
| CADC winter evaluation | ⬜ Upcoming | Wk 6 |
| Temporal feature fusion | ⬜ Upcoming | Wk 7 |
| MC Dropout uncertainty maps | ⬜ Upcoming | Wk 9 |
| Confidence-gated GeoJSON flagging | ⬜ Upcoming | Wk 10 |
| TensorRT FP16 engine + latency profiler | ⬜ Upcoming | Wk 11 |
| Custom CUDA normalization kernel | ⬜ Upcoming | Wk 12 |
| ROS2 C++ inference node | ⬜ Upcoming | Wk 13 |
| arXiv preprint submission | ⬜ Upcoming | Wk 14 |

---

## Setup

```bash
git clone https://github.com/arpitt14/av-perception-portfolio.git
cd av-perception-portfolio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All scripts are run from the repo root using `python -m module.path` to prevent
import scope errors. Example: `python -m src.geometry.camera_model`

---

## Background

Built by a Mathematics and Computing undergraduate at Delhi Technological University
as part of a structured 16-week roadmap toward production-level AV perception
engineering. Foundation: linear algebra, probability theory, Bayesian inference,
PyTorch, OpenCV. Building toward: CUDA, TensorRT, ROS2, C++ inference.

The commit history is the evidence trail. Every week ships a measurable deliverable.
