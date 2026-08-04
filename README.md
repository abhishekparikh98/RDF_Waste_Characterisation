# ♻️ Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production Using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-ultralytics)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

---

## 📖 Overview

This repository contains the implementation of my **MSc Computing Dissertation** at the **University of Roehampton**.

The project presents a **multi-modal waste characterisation system** that combines **Deep Learning**, **Object Detection**, and **Classical Machine Learning** to classify mixed household waste and estimate its suitability for **Refuse-Derived Fuel (RDF)** production.

The Flask web application runs **two vision models in parallel on every upload** — a fine-tuned **YOLOv8n** multi-object detector and a legacy **Keras CNN** — and stitches their predictions to a **Random Forest** classifier via a hand-curated material feature library. A **Grad-CAM** interpretability layer is produced for the legacy CNN route; when YOLO emits no confident detection, the page routes silently to the CNN and renders the heatmap.

---

## 🎯 Research Objectives

- Build a reproducible image dataset from TrashNet and TACO with a 60-to-6 category mapping.
- Train and evaluate a from-scratch CNN baseline plus MobileNetV2 and ResNet50 transfer-learning backbones.
- Train a Random Forest on synthetic RDF tabular data and validate via 5-fold cross-validation.
- Bridge the image and tabular halves with a hand-curated material feature library.
- Replace the single-label CNN with a YOLOv8n multi-object detector for the deployment route.
- Expose the system through a Flask web application with parallel dual-model routing.
- Provide a Grad-CAM interpretability layer for the CNN route.
- Be honest about the limitations (synthetic RDF data, training-distribution / deployment-distribution mismatch).

---

## 🚀 Features

- ✅ Waste Image Classification (single-label CNN)
- ✅ Multi-object Waste Detection (YOLOv8n, confidence threshold 0.45)
- ✅ RDF Suitability Prediction (Random Forest)
- ✅ BaselineCNN (3-block ConvNet, ~25.8 M params)
- ✅ ResNet50 Transfer Learning (test accuracy 0.888)
- ✅ MobileNetV2 Transfer Learning (test accuracy 0.820)
- ✅ Random Forest RDF classifier (test F1 0.914)
- ✅ Multi-modal Late-Fusion Pipeline
- ✅ Dual-Model Flask Web Interface (YOLO + CNN, parallel)
- ✅ Confidence-Coloured Progress Bar
- ✅ Grad-CAM heatmap for the CNN route
- ✅ "No confident detection" failure mode (declines to answer rather than guess)

---

## 🏗️ Project Architecture

```
                   ┌────────────────────────────┐
                   │     Uploaded waste image    │
                   └─────────────┬──────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
     ┌────────────────────┐              ┌────────────────────┐
     │  YOLOv8n (CPU)     │              │  Legacy CNN (CPU)  │
     │  conf_threshold=   │              │  BaselineCNN /     │
     │  0.45              │              │  ResNet50          │
     └─────────┬──────────┘              └─────────┬──────────┘
               │ bbox list                       │ class + Grad-CAM
               ▼                                 ▼
       ┌──────────────────────────────────────────────────┐
       │       MATERIAL_FEATURE_LIBRARY (6 rows)         │
       │   cardboard  glass  metal  paper  plastic  trash │
       └─────────────────────┬────────────────────────────┘
                             ▼
                ┌────────────────────────────┐
                │  Random Forest (joblib)    │
                │   → RDF Suitable / Not     │
                └────────────────────────────┘
```

Routing logic (`app.py::index()`): YOLO is tried first; if it emits ≥ 1 detection with confidence ≥ 0.45 the multi-object branch renders. Otherwise the legacy CNN branch renders and Grad-CAM is overlaid. The two vision models run independently in parallel and their agreement / disagreement is surfaced in plain English.

---

## 🧠 Machine Learning Models

| Model | Purpose | Test score |
|---|---|---|
| BaselineCNN | Single-label image classification (from scratch) | Accuracy 0.5744 |
| MobileNetV2 | Lightweight transfer learning (ImageNet) | Accuracy 0.8198 |
| ResNet50 | Deep transfer learning (ImageNet) | Accuracy 0.8877 |
| **YOLOv8n** | Multi-object detection (TACO) | mAP@0.5 0.201 (peak 0.225 at epoch 44) |
| Random Forest | RDF suitability (synthetic tabular) | F1 0.9141 |
| Multi-modal fusion | Late-fusion of vision + chemistry | n/a (architectural) |

---

## 🗂 Dataset

| Dataset | Type | Records | Classes | Source |
|---|---|---|---|---|
| TrashNet | Single-object images | 2,527 | 6 | Yang & Thung (2016) |
| TACO | Multi-object images | 1,500 (after Flickr dead-link pruning) | 60 → 6 | Proença & Simões (2020) |
| RDF features | Synthetic tabular | 3,000 | binary | Generated by `src/rdf_preprocessing.py` |

The 60-to-6 TACO category mapping is documented in [`docs/TACO_TO_YOLO_MAPPING.md`](docs/TACO_TO_YOLO_MAPPING.md).

---

## 💻 Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| Flask | Web application (single route) |
| TensorFlow + Keras | Deep learning (CNN, transfer learning) |
| Ultralytics YOLOv8 | Multi-object detection |
| Scikit-learn | Random Forest, GridSearchCV, ColumnTransformer |
| OpenCV | Image processing, Grad-CAM colormap |
| NumPy / Pandas | Numerical and tabular computing |
| Matplotlib | All figures and plots |
| python-docx | Dissertation assembly |
| HTML / CSS | Frontend (Jinja2 template + academic dashboard CSS) |

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/abhishekparikh98/RDF_Waste_Characterisation.git
```

Move into the project directory

```bash
cd RDF_Waste_Characterisation
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application.

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 🔁 Reproducing the Trained Models

```bash
python scripts/preprocess_dataset.py        # produces data/processed/
python scripts/convert_taco_to_yolo.py      # produces data/yolo/
python scripts/train_cnn.py                 # produces models/cnn_baseline_best.h5
python scripts/compare_cnn_mobilenetv2.py   # produces mobilenetv2_best.h5, resnet50_best.h5
python scripts/train_yolo.py                # produces models/yolo_best.pt (~5.4 h on CPU)
python scripts/train_rdf_rf.py              # produces models/rdf_random_forest_pipeline.joblib
python scripts/evaluate_yolo.py             # produces results/yolo_* and reports/yolo_evaluation_report.md
python app.py                               # launches the Flask demo
```

All random seeds are fixed at 42; dependencies are pinned in `requirements.txt`.

---

## 🖼️ Application Workflow

1. Upload a waste image.
2. YOLOv8n and the legacy CNN both run in parallel.
3. The YOLO pipeline emits bounding boxes + per-object RDF verdicts (if any object has confidence ≥ 0.45).
4. The legacy CNN pipeline emits a single class prediction + Grad-CAM heatmap.
5. The Random Forest consumes the chemistry row from `MATERIAL_FEATURE_LIBRARY` and returns a binary RDF verdict.
6. The page renders:
   - Predicted waste class(es) with confidence bars
   - Per-object RDF suitability (YOLO route) or single verdict (CNN route)
   - AI explanation (model agreement / disagreement in plain English)
   - Grad-CAM overlay (CNN route)
   - Material properties table
   - Recommendation
   - Collapsible technical details

---

## 📸 Screenshots

Screenshots of the live Flask demo live under `Screenshot/`. They show the upload form, the CNN-based result page with Grad-CAM overlay, and the YOLO-based multi-object result page.

| Stage | Where |
|---|---|
| Upload form | `Screenshot/Screenshot 2026-07-30 192732.png` |
| CNN result + Grad-CAM | `Screenshot/Screenshot 2026-07-30 192821.png` |
| YOLO multi-object result | `Screenshot/Screenshot 2026-07-30 193018.png` |
| Upload interface (hero) | `Screenshot/Upload Interface.png.png` |

Source architecture and workflow diagrams live in `docs/` (`.drawio` + exported PNGs).

---

## ⚠️ Known Limitations

- **Confidence threshold 0.45**: The trained YOLOv8n model reaches mAP@0.5 ≈ 0.20 on the converted TACO dataset. At the Ultralytics default of 0.25 the model emits many false-positive 'trash' predictions on out-of-distribution phone photographs. The runtime threshold is therefore raised to 0.45 (configured in `src/yolo_detector.py::YOLODetectorConfig`). At this threshold, images that do not yield a confident detection are reported as "no confident detection" rather than guessed.
- **Synthetic RDF lab data**: The RDF tabular dataset is generated by a rule engine, not from real plant chemistry. The reported Random Forest accuracy is an upper bound on what the model can achieve against plant data.
- **Distribution shift**: TrashNet and TACO images differ from real-world phone photographs. The Flask demo handles this gracefully (CNN fallback), but the deployed model would need a conveyor-belt dataset and an active-learning loop.
- **CPU training**: All training was performed on CPU. YOLOv8n training took 5.43 hours; a workstation-class GPU would enable larger backbones (YOLOv8s / YOLOv8m) and longer schedules.

---

## 📈 Experimental Results

| Model | Test score | Report |
|---|---|---|
| Baseline CNN | Accuracy 0.5744 | `reports/cnn_baseline_report.md` |
| MobileNetV2 | Accuracy 0.8198 | `reports/cnn_mobilenetv2_resnet50_evaluation_report.md` |
| ResNet50 | Accuracy 0.8877 | same |
| Random Forest | F1 0.9141 | `reports/rdf_random_forest_report.md` |
| YOLOv8n | mAP@0.5 0.201 (peak 0.225 at epoch 44) | `reports/yolo_evaluation_report.md` |

---

## 🔬 Research Contributions

- A documented 60-to-6 TACO category mapping with explicit treatment of mislabelled parent categories.
- A hand-curated six-row material feature library that bridges detected classes to RDF chemistry features.
- A dual-model Flask demo that runs YOLOv8 and the legacy CNN in parallel on every upload and surfaces their agreement / disagreement.
- A reproducible training and evaluation pipeline with fixed random seeds, pinned dependencies, and committed checkpoints.
- An honest discussion of the limitations (synthetic RDF data, distribution shift, short training budget).

---

## 📌 Future Work

- Real RDF plant-data integration (replace the synthetic table).
- YOLOv8s / YOLOv8m backbones on a workstation-class GPU.
- YOLO-native explainable AI (EigenCAM, EigenGradCAM).
- Active-learning loop for low-confidence detections.
- Real-time webcam / conveyor-belt integration (`stream=True`).
- End-to-end learned bridge between vision features and chemistry features.
- Proper pytest suite (preprocessing, model loaders, pipeline, Flask route).
- Production hardening (Docker, WSGI, auth, rate limiting, monitoring).
- Mobile deployment (TensorFlow Lite / PyTorch Mobile).

---

## 📁 Pre-trained Models

Trained checkpoints (`.h5`, `.pt`, `.joblib`) live under `models/`. The YOLOv8n checkpoint is included; the CNN and Random Forest checkpoints can be regenerated by the training scripts above.

The repository also contains:

- Model architectures
- Training scripts
- Inference code
- Data preprocessing pipeline
- Evaluation reports and figures

---

## 📚 References

- Yang, M. & Thung, G. (2016). *Classification of trash for recyclability status*. CS229 Project Report, Stanford University.
- Proença, P. & Simões, P. (2020). *TACO: Trash annotations in context for litter detection*. arXiv:2003.06975.
- Jocher, G., Chaurasia, A. & Qiu, J. (2023). *Ultralytics YOLO (Version 8.0.0)*. https://github.com/ultralytics/ultralytics
- He, K., Zhang, X., Ren, S. & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. & Chen, L.-C. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks*. CVPR.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. ICCV.
- Breiman, L. (2001). *Random forests*. Machine Learning 45(1), 5–32.
- Geirhos, R. et al. (2020). *Shortcut learning in deep neural networks*. Nature Machine Intelligence 2(11), 665–673.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python*. JMLR 12, 2825–2830.

A full APA-7 reference list is in the final dissertation (`FINAL_MSC_DISSERTATION.pdf`).

---

## 👨‍🎓 Author

**Abhishek Parikh**

MSc Computing

University of Roehampton

United Kingdom

Year: **2026**

---

## ⭐ Acknowledgements

I would like to thank my dissertation supervisor and the University of Roehampton for the guidance and support throughout this research project. The TrashNet, TACO, Ultralytics, TensorFlow, Keras, and Scikit-learn open-source communities made this work possible.
