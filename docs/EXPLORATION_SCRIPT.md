# Dataset Exploration Script - Documentation

## Overview

The `scripts/explore_dataset.py` is a comprehensive data exploration and validation tool designed for the Multi-Modal Waste Characterisation MSc dissertation project. It performs automated analysis of waste classification datasets without modifying or preprocessing any data.

---

## Script Architecture

The script is organized into modular components following clean code principles:

### 1. **Logging Configuration Module**
- **Function**: `setup_logging()`
- **Purpose**: Configures both console and file logging with UTF-8 encoding support
- **Windows Compatibility**: Special handling for UTF-8 output on Windows systems
- **Output**: Detailed debug logs to `exploration.log` and info logs to console

### 2. **DatasetDetector Class**
- **Purpose**: Auto-detects available datasets in the project
- **Methods**:
  - `detect_datasets()`: Searches for TrashNet and TACO in multiple locations
- **Functionality**:
  - Searches in project root and `data/raw/` directories
  - Automatically maps dataset paths for later processing
  - Logs detection status for transparency

### 3. **ImageAnalyzer Class**
- **Purpose**: Validates individual images and extracts properties
- **Methods**:
  - `is_valid_image()`: Checks if image is readable and not corrupted
  - `get_image_resolution()`: Extracts width and height of valid images
- **Features**:
  - Detects corrupted/unreadable images without modifying them
  - Handles multiple image formats (JPG, PNG, BMP, GIF, TIFF)
  - Graceful error handling for invalid files

### 4. **DatasetExplorer Class**
- **Purpose**: Core analysis engine for dataset statistics
- **Methods**:
  - `explore()`: Main exploration pipeline
  - `_collect_images_by_class()`: Organizes images by class/category
  - `_calculate_statistics()`: Computes all metrics
  - `_report_findings()`: Logs dataset summary
- **Outputs**:
  - Total image count
  - Class distribution with percentages
  - Resolution statistics (min, max, average)
  - Corrupted image list
  - Detailed logging at each step

### 5. **VisualizationGenerator Class**
- **Purpose**: Creates publication-quality charts
- **Methods**:
  - `generate_class_distribution_chart()`: Creates bar charts for each dataset
  - `generate_combined_comparison_chart()`: Compares multiple datasets visually
- **Features**:
  - High-resolution PNG output (300 DPI)
  - Automatic value labels on bars
  - Professional styling with seaborn
  - Saves to `reports/figures/`

### 6. **ReportGenerator Class**
- **Purpose**: Creates comprehensive markdown documentation
- **Methods**:
  - `generate_markdown_report()`: Generates structured markdown report
- **Content Includes**:
  - Executive summary
  - Dataset overview
  - Detailed statistics tables
  - Image analysis results
  - Data quality assessment
  - Visualization references
  - Recommendations for modeling

### 7. **Main Execution Pipeline**
- **Function**: `main()`
- **Execution Order**:
  1. Initialize logging and set up paths
  2. Auto-detect datasets
  3. Explore each dataset and collect statistics
  4. Generate visualization charts
  5. Generate markdown report
  6. Provide completion summary

---

## Key Features

### ✓ Automated Dataset Detection
- Scans project root and `data/raw/` for TrashNet and TACO
- Supports multiple dataset paths
- Logs detection status

### ✓ Comprehensive Image Analysis
- Counts total images per class
- Detects corrupted/unreadable images
- Extracts image resolutions
- Calculates min/max/average dimensions
- Supports multiple image formats

### ✓ Data Quality Validation
- Validates all images without modification
- Reports corrupted files (if any)
- Provides data integrity assessment

### ✓ Professional Visualizations
- Class distribution bar charts
- Dataset comparison charts
- High-resolution PNG output (300 DPI)
- Automatic legend and labels
- Professional styling

### ✓ Comprehensive Reporting
- Markdown format for version control
- Executive summary
- Detailed statistics tables
- Class distribution percentages
- Image resolution analysis
- Data quality assessment
- Actionable recommendations

### ✓ Clean Code Standards
- Type hints for all functions
- Modular class-based design
- Comprehensive docstrings
- Error handling and validation
- Logging instead of print statements
- UTF-8 encoding support for Windows

---

## Generated Outputs

### 1. **reports/dataset_report.md** (1.85 KB)
**Content**:
- Executive summary with total images (2527) and classes (6)
- Dataset overview with paths
- Detailed statistics table showing class distribution
- Image resolution analysis
- Data quality metrics
- Visualization references
- Recommendations

**Key Findings**:
```
Total Images: 2527
Classes: 6
- cardboard: 403 (15.95%)
- glass: 501 (19.83%)
- metal: 410 (16.22%)
- paper: 594 (23.51%)
- plastic: 482 (19.07%)
- trash: 137 (5.42%)

Image Resolutions: 512x384 pixels (consistent)
Corrupted Images: 0 (perfect data quality)
```

### 2. **reports/figures/trashnet_class_distribution.png** (114.76 KB)
**Content**:
- Bar chart showing images per class
- Total image count in title
- Automatic value labels on each bar
- Professional styling
- High-resolution output (300 DPI)

### 3. **exploration.log** (6.92 KB)
**Content**:
- Complete execution log with timestamps
- Dataset detection details
- Image analysis progress
- Performance metrics
- All warnings and errors

---

## Usage

### Run the Script
```bash
cd "C:\Users\Abhi\OneDrive\Desktop\Msc Project"
python scripts/explore_dataset.py
```

### Output Example
```
2026-06-26 15:41:21 - INFO - Starting dataset exploration
2026-06-26 15:41:21 - INFO - [FOUND] TrashNet detected at: ...
2026-06-26 15:41:21 - INFO - Found 6 classes
2026-06-26 15:41:21 - INFO - Total Images: 2527
2026-06-26 15:41:22 - INFO - [OK] Report saved successfully
```

---

## Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| **Total Images Analyzed** | 2527 |
| **Total Classes** | 6 |
| **Image Format** | JPEG |
| **Image Resolution** | 512x384 pixels (uniform) |
| **Corrupted Images** | 0 |
| **Data Quality** | Excellent |
| **Class Balance** | Moderate imbalance (trash: 5.42%, paper: 23.51%) |

---

## Important Notes

- **No Data Modification**: Script only reads and analyzes images
- **No Data Splitting**: Raw dataset remains intact
- **No Model Training**: Pure exploration and validation only
- **Reproducible Output**: Same results on repeated runs
- **Version Controlled**: All reports are markdown and can be tracked in git

---

## Future Enhancements

The script is designed to be extended:
- Support for additional datasets (TACO images if added)
- Custom class filtering
- Statistical distribution analysis
- Data augmentation recommendations
- Train/validation/test split suggestions

---

*Script created as part of MSc Computing Dissertation - Multi-Modal Waste Characterisation Project*
