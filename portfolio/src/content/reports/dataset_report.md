# Dataset Exploration and Validation Report

**Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production Using Machine Learning**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dataset Overview](#dataset-overview)
3. [Detailed Statistics](#detailed-statistics)
4. [Image Analysis](#image-analysis)
5. [Data Quality](#data-quality)
6. [Visualizations](#visualizations)

## Executive Summary

- **Total Images Analyzed**: 2527
- **Total Classes**: 6
- **Datasets Found**: 1

## Dataset Overview

### TrashNet

- **Path**: `C:\Users\Abhi\OneDrive\Desktop\Msc Project\TrashNET Data set`
- **Total Images**: 2527
- **Number of Classes**: 6

## Detailed Statistics

### TrashNet - Class Distribution

| Class | Image Count | Percentage |
|-------|-------------|------------|
| cardboard | 403 | 15.95% |
| glass | 501 | 19.83% |
| metal | 410 | 16.22% |
| paper | 594 | 23.51% |
| plastic | 482 | 19.07% |
| trash | 137 | 5.42% |

**Total**: 2527 images

## Image Analysis

### TrashNet - Image Resolutions

**Width (pixels)**
- Minimum: 512
- Maximum: 512
- Average: 512.00

**Height (pixels)**
- Minimum: 384
- Maximum: 384
- Average: 384.00

## Data Quality

### TrashNet

- **Corrupted Images Detected**: 0
- [OK] No corrupted images detected

## Visualizations

Class distribution charts have been generated and saved to `reports/figures/`

- `trashnet_class_distribution.png`
- `dataset_comparison.png`

## Recommendations

1. **Data Balance**: Review class distribution for imbalance issues
2. **Image Sizes**: Consider standardizing image resolutions
3. **Data Quality**: Investigate and handle any corrupted images
4. **Train/Val/Test Split**: Plan appropriate data splits before modeling

---

*Report generated automatically by dataset exploration script*
