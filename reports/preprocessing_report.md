# Data Preprocessing Report

**Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production Using Machine Learning**

---

## Table of Contents

1. [Overview](#overview)
2. [Preprocessing Steps](#preprocessing-steps)
3. [Dataset Statistics](#dataset-statistics)
4. [Train/Validation/Test Split](#trainvalidationtest-split)
5. [Image Properties](#image-properties)
6. [Data Quality](#data-quality)
7. [Output Structure](#output-structure)
8. [Validation Results](#validation-results)

## Overview

This report documents the preprocessing pipeline applied to the TrashNet dataset for the MSc dissertation project.

**Purpose**: Prepare raw images for machine learning model training by standardizing formats, resizing, normalizing, and creating stratified splits.

**Key Objective**: Maintain data integrity while creating reproducible, balanced dataset splits.

## Preprocessing Steps

The following preprocessing steps were applied to each image:

1. **Image Validation**: Verify all images are readable and not corrupted
2. **Format Conversion**: Convert all images to RGB format (3 channels)
3. **Resizing**: Resize all images to 224x224 pixels using Lanczos interpolation
4. **Normalization**: Normalize pixel values to range [0, 1] by dividing by 255
5. **Stratified Splitting**: Divide dataset into train/validation/test sets while maintaining class distribution
6. **Storage**: Save processed images as PNG files (lossless compression)

## Dataset Statistics

### Original Dataset

- **Total Images**: 2527
- **Total Classes**: 6
- **Classes**: cardboard, glass, metal, paper, plastic, trash

## Train/Validation/Test Split

### Split Ratios

- **Training**: 70%
- **Validation**: 15%
- **Testing**: 15%

### Split Results

**TRAIN**: 1763 images (69.8%)

| Class | Count | Percentage |
|-------|-------|------------|
| cardboard | 281 | 15.94% |
| glass | 350 | 19.85% |
| metal | 286 | 16.22% |
| paper | 415 | 23.54% |
| plastic | 336 | 19.06% |
| trash | 95 | 5.39% |

**VALIDATION**: 381 images (15.1%)

| Class | Count | Percentage |
|-------|-------|------------|
| cardboard | 61 | 16.01% |
| glass | 75 | 19.69% |
| metal | 62 | 16.27% |
| paper | 89 | 23.36% |
| plastic | 73 | 19.16% |
| trash | 21 | 5.51% |

**TEST**: 383 images (15.2%)

| Class | Count | Percentage |
|-------|-------|------------|
| cardboard | 61 | 15.93% |
| glass | 76 | 19.84% |
| metal | 62 | 16.19% |
| paper | 90 | 23.50% |
| plastic | 73 | 19.06% |
| trash | 21 | 5.48% |

## Image Properties

### Target Specifications

- **Resolution**: 224 x 224 pixels
- **Format**: RGB (3 channels)
- **Data Type**: 32-bit floating point (float32)
- **Value Range**: [0.0, 1.0]
- **Interpolation**: Lanczos (high-quality resizing)
- **Storage Format**: PNG (lossless)

## Data Quality

**Invalid/Corrupted Images**: 0

[OK] All images validated successfully

## Output Structure

Processed dataset saved to `data/processed/` with structure:

```
data/processed/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── validation/
│   ├── cardboard/
│   └── ...
└── test/
    ├── cardboard/
    └── ...
```

## Validation Results

| Metric | Result |
|--------|--------|
| Total Images Processed | 2527 |
| Training Images | 1763 |
| Validation Images | 381 |
| Test Images | 383 |
| Invalid Images | 0 |
| Target Resolution | 224x224 RGB |
| Normalization | [0, 1] |

## Key Findings

1. **Class Balance**: Review class distribution for potential imbalance:
   - Smallest class (train): trash (95 images)
   - Largest class (train): paper (415 images)
   - Imbalance ratio: 4.37:1

2. **Data Quality**: All images validated and readable

3. **Reproducibility**: Random seed (42) ensures reproducible splits

## Recommendations

1. **Class Imbalance Handling**: Consider class weighting in model training
2. **Data Augmentation**: Apply augmentation during training for minority classes
3. **Model Input**: Images are ready for deep learning pipelines (e.g., PyTorch, TensorFlow)
4. **Validation Strategy**: Use stratified split for fair cross-validation

---

*Report generated automatically by preprocessing pipeline*
