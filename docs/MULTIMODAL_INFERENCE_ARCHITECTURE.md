# Multimodal Inference Architecture

## Overview

This pipeline links the trained waste image classifier with the RDF suitability model in a single inference flow:

`Image -> Waste Classification -> Material Features -> RDF Suitability Prediction`

It is intentionally modular and does not include a web interface.

## Components

| Component | File | Responsibility |
|---|---|---|
| Image classifier loader | `src/multimodal_inference.py` | Loads the trained CNN or transfer-learning classifier and predicts waste class |
| Material feature mapper | `src/multimodal_inference.py` | Converts the predicted class into RDF-related tabular features |
| RDF predictor | `src/multimodal_inference.py` | Loads the trained Random Forest pipeline and predicts suitability |
| CLI runner | `scripts/run_multimodal_inference.py` | Executes the full inference chain from the terminal |

## Data Flow

1. A waste image is loaded from disk.
2. The image classifier predicts the waste class and confidence.
3. The predicted class is mapped to material features:
   - `material_type`
   - `moisture_content`
   - `contamination_level`
   - `combustibility`
   - `calorific_value`
4. The RDF model consumes those features and predicts:
   - binary RDF suitability
   - RDF probability

## Why This Design

- **Separation of concerns:** image classification and RDF prediction stay independent.
- **Reusability:** the same RDF predictor can be used later with real tabular features.
- **Configurability:** the image model path and preprocessing mode can be changed without changing the pipeline.
- **Research clarity:** the architecture is easy to describe in a dissertation and easy to validate step by step.

## Default Artifact Paths

- Image classifier: `models/cnn_baseline_best.h5`
- RDF model: `models/rdf_random_forest_pipeline.joblib`

## Inference Output

The pipeline returns:

- predicted waste class
- classifier confidence
- generated material features
- RDF suitability label
- RDF suitability probability

## Extensibility

Future work can replace the material mapping with:

- real measured RDF tabular records
- confidence-weighted feature estimation
- joint multimodal fusion
- batch inference over folders or datasets
