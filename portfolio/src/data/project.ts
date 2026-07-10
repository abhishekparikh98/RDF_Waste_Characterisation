/**
 * Real project metadata, sourced directly from the analysis of the MSc project
 * at D:\University\Msc Project. Nothing here is invented.
 */

export const project = {
  title: "Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production Using Machine Learning",
  shortTitle: "Multi-Modal Waste Characterisation for RDF Production",
  type: "MSc Computing Dissertation",
  author: {
    name: "Abhishek Parikh",
    email: "abhishekparikh98@gmail.com",
    role: "MSc Computing Candidate",
  },
  year: 2026,
  status: "Completed and Evaluated",
  repository: {
    url: "https://github.com/abhishekparikh98/msc-waste-characterisation",
    branch: "master",
  },
};

export const projectSummary = {
  goal: "Classify waste images and predict the suitability of the material for Refuse-Derived Fuel (RDF) production using a multi-modal machine-learning pipeline.",
  modality: "Computer vision (image) + tabular sensor/material features (Random Forest)",
  pipeline:
    "Image -> Waste Classification (CNN / MobileNetV2 / ResNet50) -> Predicted Class -> Material Feature Mapping -> Random Forest -> RDF Suitable / Not Suitable",
  primaryDataset: "TrashNet (2,527 images, 6 classes)",
  secondaryDatasets: [
    "TACO — Trash Annotations in Context (downloaded but not used for training in this submission)",
    "Synthetic RDF tabular dataset (3,000 samples) generated from domain-informed material profiles",
  ],
  deployment: "Flask web application (app.py) + command-line multimodal inference (scripts/run_multimodal_inference.py)",
};

export const researchObjectives: string[] = [
  "Develop machine learning models capable of classifying waste materials from multiple modalities (images, sensor data).",
  "Combine TrashNet and TACO datasets for comprehensive waste material characterisation.",
  "Evaluate model performance on waste-to-RDF production pipeline optimisation.",
  "Create interpretable models suitable for industrial deployment.",
];

export const researchProblem = {
  context:
    "Refuse-Derived Fuel (RDF) is produced from municipal solid waste after sorting, shredding, and drying. The calorific value, moisture, and contamination of the input material determine the energy content and emissions profile of the resulting fuel.",
  gap:
    "Most RDF production lines still rely on manual sorting, which is slow, inconsistent, and difficult to scale. Pure computer-vision systems also do not consider the material properties that ultimately decide fuel quality.",
  approach:
    "Build a multi-modal ML pipeline that first classifies the waste from an image, then maps the predicted class to RDF-relevant material features, and finally predicts suitability with a tree-based classifier.",
  questions: [
    "How accurately can a from-scratch CNN and two transfer-learning backbones (MobileNetV2, ResNet50) classify TrashNet waste categories?",
    "How well does a Random Forest classifier predict RDF suitability from the mapped material features (moisture, contamination, combustibility, calorific value)?",
    "Can the two stages be chained into a single, deployable multimodal inference pipeline?",
  ],
};

export const literatureReview = [
  {
    title: "TrashNet dataset and waste classification baselines",
    summary:
      "TrashNet (Yang & Thung, 2016) is a six-class waste image dataset widely used as a baseline. It provides ~2,500 images across cardboard, glass, metal, paper, plastic, and trash and is the de-facto reference benchmark for academic waste-classification studies.",
  },
  {
    title: "TACO — Trash Annotations in Context",
    summary:
      "TACO is an open image dataset of litter in real-world environments, with bounding-box and segmentation masks. It is suited to litter detection rather than the clean TrashNet categories, so the present work uses it only as a reference for future extension.",
  },
  {
    title: "Convolutional and transfer-learning image classifiers",
    summary:
      "From-scratch CNNs and ImageNet-pretrained backbones (MobileNetV2, ResNet50, EfficientNet) have been applied to waste classification and consistently outperform shallow baselines on small datasets. The present work follows this two-stage frozen-then-fine-tune transfer-learning recipe.",
  },
  {
    title: "Refuse-Derived Fuel characterisation",
    summary:
      "RDF quality is governed by calorific value, moisture content, contamination, and combustibility. Empirical material profiles for the six TrashNet classes were used to seed the tabular dataset in this work, and a Random Forest was selected for its interpretability and robustness to mixed feature types.",
  },
];
