# Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production Using Machine Learning

## Project Overview

This repository contains the implementation of an MSc Computing dissertation project focused on developing machine learning models for multi-modal waste characterization. The goal is to improve the quality of Refuse-Derived Fuel (RDF) production through automated waste classification and characterization using combined computer vision and sensor data.

## Research Objectives

- Develop machine learning models capable of classifying waste materials from multiple modalities (images, sensor data)
- Combine TrashNet and TACO datasets for comprehensive waste material characterization
- Evaluate model performance on waste-to-RDF production pipeline optimization
- Create interpretable models suitable for industrial deployment

## Project Structure

```
msc-project/
├── src/                      # Source code for models and utilities
│   ├── models/              # ML model implementations
│   ├── preprocessing/       # Data preprocessing pipelines
│   ├── utils/               # Utility functions
│   └── evaluation/          # Evaluation metrics and scripts
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── data/
│   ├── raw/                 # Original TrashNet and TACO datasets
│   └── processed/           # Cleaned and preprocessed datasets
├── models/                  # Trained model checkpoints and weights
├── reports/                 # Generated analysis and figures
│   └── figures/             # Plots, visualizations, diagrams
├── results/                 # Experimental results and outputs
├── docs/                    # Project documentation
├── scripts/                 # Utility scripts for data processing, training
├── requirements.txt         # Python dependencies
├── .gitignore               # Git exclusion rules
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Dataset Information

### TrashNet
- A dataset for waste material classification
- Contains multiple waste categories for machine learning training

### TACO
- "Trash Annotations in Context" dataset
- Provides contextual waste object detection and segmentation

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd msc-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

(To be populated during project development)

## Project Timeline

- **Phase 1**: Data exploration and preprocessing
- **Phase 2**: Model development and training
- **Phase 3**: Evaluation and optimization
- **Phase 4**: Results analysis and reporting

## Contributing Guidelines

This is an academic dissertation project. For any modifications or extensions, please consult with the project supervisor.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Author

**Student**: [Your Name]  
**Institution**: [Your University]  
**Supervisor**: [Supervisor Name]  
**Year**: 2026

## References

- TrashNet Dataset: [Reference]
- TACO Dataset: [Reference]

---

**Status**: Project Initialized | **Last Updated**: June 2026
