# ♻️ Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production Using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

---

## 📖 Overview

This repository contains the implementation of my **MSc Computing Dissertation** at the **University of Roehampton**.

The project presents **multi-modal waste characterisation system** that combines **Deep Learning** and **Machine Learning** techniques to classify mixed household waste and estimate its suitability for **Refuse-Derived Fuel (RDF)** production.

The system analyses waste images using Convolutional Neural Networks (CNNs) and Transfer Learning models while integrating additional waste characteristics to improve prediction accuracy for RDF suitability.

---

# 🎯 Research Objectives

- Develop waste classification system using machine learning.
- Improve RDF quality through automated waste characterisation.
- Compare CNN and Transfer Learning models.
- Integrate image-based classification with additional waste characteristics.
- Develop an easy-to-use Flask web application.
- Evaluate the effectiveness of multiple machine learning approaches.

---

# 🚀 Features

- ✅ Waste Image Classification
- ✅ RDF Suitability Prediction
- ✅ CNN Model
- ✅ ResNet50 Transfer Learning
- ✅ MobileNetV2 Transfer Learning
- ✅ Random Forest Classification
- ✅ Multi-modal Machine Learning
- ✅ Flask Web Interface
- ✅ Confidence Score Prediction
- ✅ Grad-CAM Explainability
-    YOLOv8 Object Detection

---

# 🏗️ Project Architecture

```
Waste Image
      │
      ▼
Image Preprocessing
      │
      ▼
CNN / ResNet50 / MobileNetV2
      │
      ▼
Feature Extraction
      │
      ▼
Random Forest
      │
      ▼
Multi-modal Fusion
      │
      ▼
Waste Classification
      │
      ▼
RDF Suitability Prediction
```

---


# 🧠 Machine Learning Models

The project evaluates multiple machine learning models.

| Model | Purpose |
|--------|---------|
| CNN | Baseline Image Classification |
| MobileNetV2 | Lightweight Transfer Learning |
| ResNet50 | Deep Transfer Learning |
| Random Forest | RDF Suitability Classification |
| Multi-modal Fusion | Combined Prediction |


---

# 🗂 Dataset

This project combines two publicly available datasets.

## TrashNet

- Household waste images
- Multiple waste categories
- Image classification dataset

## TACO Dataset

- Trash Annotations in Context
- Real-world waste images
- Object detection and segmentation

---

# 💻 Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Flask | Web Application |
| TensorFlow | Deep Learning |
| Scikit-learn | Machine Learning |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Pandas | Data Analysis |
| Matplotlib | Visualisation |
| HTML/CSS | Frontend |

---

# ⚙️ Installation

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

# ▶️ Running the Application

Start the Flask application.

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

# 🖼️ Application Workflow

1. Upload a waste image.
2. The image is preprocessed.
3. The selected AI model performs classification.
4. The Random Forest model predicts RDF suitability.
5. The system displays:

- Predicted waste class
- Confidence score
- RDF suitability
- Recommendation

---

## 📸 Screenshots

## 🏠 Homepage

The homepage allows users to upload waste images and select the trained model for prediction.

<p align="center">
  <img src="docs/homepage.png" width="900" alt="Homepage">
</p>

---

## 📤 Upload Interface

The upload interface enables users to select an image and initiate waste classification.

<p align="center">
  <img src="docs/upload-interface.png" width="900" alt="Upload Interface">
</p>

---

## 📊 Prediction Result

The prediction page displays the predicted waste class, confidence score, RDF suitability, and recommendation.

<p align="center">
  <img src="docs/prediction-result.png" width="900" alt="Prediction Result">
</p>

---

## 🔥 Grad-CAM Visualization

Grad-CAM highlights the regions of the image that contributed most to the model's prediction, improving interpretability.

<p align="center">
  <img src="docs/gradcam.png" width="900" alt="Grad-CAM Visualization">
</p>

# 📈 Experimental Results

| Model | Status |
|--------|--------|
| CNN | ✅ Implemented |
| MobileNetV2 | ✅ Implemented |
| ResNet50 | ✅ Implemented |
| Random Forest | ✅ Implemented |
| Multi-modal Prediction | ✅ Implemented |
| GradCAM | ✅ Implemented |
| YOLOv8 | Not Implemented |

> **Note:** Final accuracy values will be included after completion of the dissertation experiments.

---

# 🔬 Research Contributions

This project demonstrates:

- waste characterisation using machine learning.
- Transfer Learning for waste classification.
- Multi-modal machine learning integration.
- RDF quality prediction.
- Practical deployment using Flask.

---

# 📌 Future Work

- Vision Transformers (ViT)
- Real-time webcam prediction
- Mobile application
- Cloud deployment
- Industrial-scale RDF optimisation

---

# 📁 Pre-trained Models

The trained model files (`.h5`) are **not included** in this repository because they exceed GitHub's file size limit.

The repository contains:

- Model architectures
- Training scripts
- Inference code
- Data preprocessing pipeline

The trained models can be regenerated using the provided training scripts.

---

# 👨‍🎓 Author

**Abhishek Parikh**

MSc Computing

University of Roehampton

United Kingdom

Year: **2026**

---

# 📚 References

- TrashNet Dataset
- TACO Dataset
- TensorFlow Documentation
- Scikit-learn Documentation
- Flask Documentation

---



# ⭐ Acknowledgements

I would like to thank my dissertation supervisor and the University of Roehampton for their guidance and support throughout this research project.

---
