# Data Preprocessing Module - Comprehensive Documentation

## Overview

The preprocessing pipeline is a production-grade system for preparing the TrashNet dataset for machine learning model training. It combines multiple modular components designed following clean architecture principles with full type hints and comprehensive logging.

---

## Architecture Overview

The preprocessing system is built on a layered architecture:

```
┌─────────────────────────────────────────────────────┐
│         scripts/preprocess_dataset.py               │
│              (Runner & Orchestration)               │
├─────────────────────────────────────────────────────┤
│         PreprocessingPipeline                       │
│        (Main Orchestrator)                          │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Image        │  │ Image        │  │ Dataset   │ │
│  │ Validator    │  │ Preprocessor │  │ Splitter  │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │ Dataset      │  │ Preprocessing Visualizer     │ │
│  │ Saver        │  │ Report Generator             │ │
│  └──────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Module 1: `src/preprocessing.py`

### Core Classes and Their Responsibilities

#### **ImageValidator**
**Purpose**: Validates image files for readability and integrity before processing.

**Class Methods**:
```python
class ImageValidator:
    def is_valid_image(image_path: Path) -> bool
    def validate_dataset(dataset_path: Path) -> Dict[str, List[Path]]
```

**Functionality**:
- Checks if image files can be opened and read successfully
- Identifies corrupted or unreadable images
- Returns dictionary mapping classes to valid image paths
- Logs invalid images for audit trail

**Key Features**:
- Supports multiple formats: JPG, PNG, BMP, GIF, TIFF
- Uses PIL's verify() method for thorough validation
- No image modifications - purely inspection
- Detailed logging of validation process

**Output**:
```python
{
    'cardboard': [Path(...), Path(...), ...],  # 403 valid images
    'glass': [...],                             # 501 valid images
    # ... etc
}
```

---

#### **ImagePreprocessor**
**Purpose**: Handles the core image processing operations (resizing, normalization).

**Class Methods**:
```python
class ImagePreprocessor:
    def load_and_preprocess_image(image_path: Path) -> np.ndarray
    def preprocess_batch(image_paths: List[Path]) -> Dict[str, np.ndarray]
```

**Preprocessing Pipeline for Each Image**:

1. **Load as RGB**
   - Opens image using PIL
   - Converts to RGB format (3 channels)
   - Handles grayscale, RGBA, and other formats

2. **Resize to 224x224**
   - Uses Lanczos interpolation (high-quality)
   - Maintains aspect ratio consistency
   - Deterministic resizing

3. **Normalize to [0, 1]**
   - Converts pixel values from [0, 255] to [0, 1]
   - Formula: `normalized = pixel / 255.0`
   - Stores as float32 for neural networks

**Code Example**:
```python
# Original image: 512x384, pixel values [0-255]
img = load_and_preprocess_image(path)
# Result: 224x224 RGB, float32, values [0.0-1.0]
print(img.shape)  # (224, 224, 3)
print(img.dtype)  # float32
print(img.min(), img.max())  # ~0.0, ~1.0
```

**Error Handling**:
- Gracefully handles corrupted files
- Returns None on processing failure
- Logs detailed error messages

---

#### **DatasetSplitter**
**Purpose**: Performs reproducible stratified train/validation/test splits.

**Class Methods**:
```python
class DatasetSplitter:
    def stratified_split(valid_images: Dict[str, List[Path]]) -> Dict[str, Dict]
    def get_split_statistics(splits: Dict[str, Dict]) -> Dict
```

**Key Algorithm**:
- Uses scikit-learn's `train_test_split` for stratification
- Splits each class independently to maintain distribution
- Two-stage splitting:
  1. Separate test set (15%)
  2. Split remainder into train (70%) and validation (15%)

**Stratification Guarantee**:
```
Original:  paper 23.5%, glass 19.8%, ... trash 5.4%
           ↓ (stratified split)
Train:     paper 23.54%, glass 19.85%, ... trash 5.39%
Validation: paper 23.36%, glass 19.69%, ... trash 5.51%
Test:      paper 23.50%, glass 19.84%, ... trash 5.48%
```

**Output Structure**:
```python
{
    'train': {
        'cardboard': [Path(...), ...],  # 281 images
        'glass': [...],                  # 350 images
        ...
    },
    'validation': {
        'cardboard': [Path(...), ...],  # 61 images
        ...
    },
    'test': {
        'cardboard': [Path(...), ...],  # 61 images
        ...
    }
}
```

**Statistics**:
```python
stats = {
    'train': {
        'total_images': 1763,
        'num_classes': 6,
        'class_distribution': {...}
    },
    # ... validation and test
}
```

---

#### **DatasetSaver**
**Purpose**: Efficiently saves preprocessed images to disk in organized directory structure.

**Class Methods**:
```python
class DatasetSaver:
    def save_preprocessed_dataset(splits: Dict, preprocessor: ImagePreprocessor) -> Dict[str, int]
```

**Workflow**:
1. Creates directory structure: `data/processed/{train|validation|test}/{class}/`
2. For each image in split:
   - Loads and preprocesses using ImagePreprocessor
   - Converts normalized float32 array back to uint8
   - Saves as PNG (lossless compression)
   - Preserves original filename

**Output Structure**:
```
data/processed/
├── train/
│   ├── cardboard/
│   │   ├── cardboard1.jpg
│   │   ├── cardboard2.jpg
│   │   └── ...
│   ├── glass/
│   └── ...
├── validation/
│   ├── cardboard/
│   └── ...
└── test/
    ├── cardboard/
    └── ...
```

**Storage Format**:
- **Input**: Original JPG files
- **Output**: PNG files (lossless)
- **Reason**: PNG preserves normalized pixel values without quality loss
- **Quality**: Saved with quality=95 for compression

**Performance**: Saves 2527 images in ~18 seconds

---

#### **PreprocessingPipeline**
**Purpose**: Orchestrates the complete preprocessing workflow.

**Main Method**:
```python
class PreprocessingPipeline:
    def run() -> Dict
```

**Execution Steps**:
1. **Validate Dataset** - Check all images are readable
2. **Stratified Split** - Create train/val/test splits
3. **Save Preprocessed** - Process and save images
4. **Generate Summary** - Log comprehensive results
5. **Return Results** - Dictionary with all statistics

**Workflow**:
```python
pipeline = PreprocessingPipeline(
    raw_dataset_path="TrashNET Data set",
    processed_output_path="data/processed",
    target_size=(224, 224),
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)

results = pipeline.run()
# Results contains all statistics and paths
```

**Returns**:
```python
{
    'success': True,
    'valid_images_count': 2527,
    'splits': {
        'train': {'total_images': 1763, ...},
        'validation': {'total_images': 381, ...},
        'test': {'total_images': 383, ...}
    },
    'saved_counts': {
        'train': 1763,
        'validation': 381,
        'test': 383
    },
    'class_distribution': {...}
}
```

---

## Module 2: `scripts/preprocess_dataset.py`

### Runner Script Components

#### **PreprocessingVisualizer**
**Purpose**: Generates publication-quality visualizations of preprocessing results.

**Methods**:
```python
class PreprocessingVisualizer:
    def plot_split_distribution(split_stats: Dict) -> Path
    def plot_class_distribution_comparison(original_dist: Dict, split_stats: Dict) -> Path
```

**Generated Visualizations**:

1. **Split Distribution Plot** (`split_distribution.png`)
   - 3-panel subplot showing train/val/test class distributions
   - Bar charts with value labels
   - Shows absolute counts and percentages

2. **Class Distribution Comparison** (`class_distribution_comparison.png`)
   - Grouped bar chart comparing classes across splits
   - Train (blue), Validation (orange), Test (green)
   - Demonstrates stratification effectiveness

**Visualization Properties**:
- Resolution: 300 DPI (publication quality)
- Format: PNG (lossless)
- Size: ~0.15 MB each
- Style: Professional seaborn styling
- Automatic legend and labels

---

#### **PreprocessingReportGenerator**
**Purpose**: Creates comprehensive markdown documentation of preprocessing results.

**Method**:
```python
class PreprocessingReportGenerator:
    def generate_report(pipeline_results: Dict, output_path: Path) -> Path
```

**Report Sections**:

1. **Overview**
   - Purpose and objectives
   - Key design decisions

2. **Preprocessing Steps**
   - Validation, format conversion, resizing, normalization
   - Storage format and lossless compression

3. **Dataset Statistics**
   - Original and processed counts
   - Classes and distributions

4. **Train/Validation/Test Split**
   - Split ratios (70/15/15)
   - Detailed table with counts and percentages for each split

5. **Image Properties**
   - Target specifications (224x224, RGB, [0,1])
   - Interpolation method and storage format

6. **Data Quality**
   - Invalid/corrupted image count (0 found)
   - Validation status

7. **Output Structure**
   - Directory layout diagram
   - File organization

8. **Validation Results**
   - Metrics table
   - Counts, resolutions, normalization range

9. **Key Findings**
   - Class balance analysis (4.37:1 imbalance ratio)
   - Data quality assessment
   - Reproducibility notes

10. **Recommendations**
    - Class imbalance handling strategies
    - Data augmentation suggestions
    - Model input readiness

---

#### **Main Execution Flow**
**Function**: `main()`

**Execution Steps**:
1. Setup logging with UTF-8 encoding for Windows
2. Verify raw dataset exists
3. Initialize PreprocessingPipeline
4. Run complete pipeline
5. Generate visualizations
6. Generate markdown report
7. Log completion summary

**Logging Output**:
- Console: INFO and above
- File (preprocessing.log): DEBUG and above
- Timestamps on all messages
- Progress indicators for long operations

---

## Key Features and Design Decisions

### 1. **Stratified Splitting**
- **Why**: Ensures class distribution is maintained across splits
- **How**: Uses sklearn's train_test_split with per-class splitting
- **Verification**: Percentages within 0.15% across all splits

### 2. **Lossless Storage (PNG)**
- **Why**: Preserves preprocessed pixel values without degradation
- **Alternative Avoided**: JPEG introduces compression artifacts
- **Trade-off**: Larger file size (~120 MB) vs. data integrity

### 3. **Lanczos Interpolation**
- **Why**: High-quality resizing (better than bilinear)
- **Advantage**: Maintains edge sharpness
- **Limitation**: Slower than bilinear (acceptable for preprocessing)

### 4. **Float32 Normalization**
- **Why**: Optimal for neural networks
- **Range**: [0.0, 1.0] instead of [0, 255]
- **Advantage**: Better numerical stability in computations

### 5. **Random Seed (42)**
- **Why**: Ensures reproducible splits across runs
- **Usage**: All experiments can be repeated identically
- **Documentation**: Seed value logged in report

### 6. **Original Dataset Unchanged**
- **Why**: Preserves raw data integrity
- **Implementation**: Only reads from raw, writes to processed
- **Benefit**: Can re-run preprocessing with different parameters

---

## Processing Statistics

### Performance Metrics
- **Total Images**: 2,527
- **Processing Time**: ~18 seconds
- **Throughput**: ~140 images/second
- **Output Size**: ~120 MB (PNG format)

### Split Results
```
Training:   1,763 images (69.8%)
Validation: 381 images  (15.1%)
Testing:    383 images  (15.2%)
```

### Class Distribution (Training Set)
```
Paper:      415 images (23.54%) ← Largest
Glass:      350 images (19.85%)
Plastic:    336 images (19.06%)
Metal:      286 images (16.22%)
Cardboard:  281 images (15.94%)
Trash:      95 images  (5.39%)  ← Smallest
────────────────────────────────
Total:      1,763 images
```

### Data Quality
- **Corrupted Images**: 0
- **Successfully Validated**: 2,527 (100%)
- **Successfully Processed**: 2,527 (100%)
- **Processing Failures**: 0

---

## Usage Instructions

### Run Preprocessing Pipeline
```bash
cd "C:\Users\Abhi\OneDrive\Desktop\Msc Project"
python scripts/preprocess_dataset.py
```

### Expected Output
```
2026-06-26 15:58:24 - INFO - Starting Preprocessing Pipeline
...
2026-06-26 15:58:42 - INFO - Preprocessing Complete!
2026-06-26 15:58:43 - INFO - Processed dataset: data/processed
2026-06-26 15:58:43 - INFO - Report: reports/preprocessing_report.md
2026-06-26 15:58:43 - INFO - Visualizations: reports/figures
```

### Generated Files
1. **Processed Dataset**: `data/processed/{train|validation|test}/{class}/*.png`
2. **Report**: `reports/preprocessing_report.md`
3. **Visualizations**: 
   - `reports/figures/split_distribution.png`
   - `reports/figures/class_distribution_comparison.png`
4. **Logs**: `preprocessing.log`

---

## Advanced Features

### 1. **Error Recovery**
- Continues processing if individual image fails
- Logs all errors for investigation
- Provides recovery report

### 2. **Detailed Logging**
- DEBUG level: Detailed progress for each image
- INFO level: High-level pipeline steps
- File logging: Complete audit trail

### 3. **Type Hints**
- All functions fully type-hinted
- Enables IDE autocompletion
- Facilitates static type checking

### 4. **Comprehensive Documentation**
- Module docstrings
- Function docstrings with Args, Returns, Raises
- Inline comments for complex logic
- Clear variable names

---

## Integration with Deep Learning Frameworks

### PyTorch
```python
from torchvision import datasets
dataset = datasets.ImageFolder('data/processed/train')
dataloader = DataLoader(dataset, batch_size=32)
```

### TensorFlow
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator()
train_data = gen.flow_from_directory('data/processed/train', target_size=(224, 224))
```

### Direct NumPy Access
```python
from PIL import Image
import numpy as np

img = Image.open('data/processed/train/cardboard/cardboard1.jpg')
array = np.array(img) / 255.0  # Already normalized
```

---

## Reproducibility and Version Control

### Reproducibility Guarantees
- ✓ Fixed random seed (42)
- ✓ Deterministic preprocessing operations
- ✓ Same input always produces identical output
- ✓ Logged random state in report

### Version Control
- Generated files tracked: reports and logs
- Processed dataset NOT tracked (generated, large)
- .gitignore excludes data/processed/
- README includes instructions to regenerate

---

## Future Enhancements

1. **Parallel Processing**: Multi-threaded image loading
2. **Caching**: Persist split assignments for fast re-processing
3. **Custom Augmentation**: Integration with preprocessing pipeline
4. **Dataset Statistics**: Advanced metrics (histogram, edge detection)
5. **Quality Metrics**: Blur detection, contrast analysis
6. **Resume Capability**: Checkpoint and resume large datasets

---

*Documentation created as part of MSc Computing Dissertation - Multi-Modal Waste Characterisation Project*
