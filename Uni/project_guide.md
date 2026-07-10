# Multi-Modal Waste Characterisation for RDF Production — Step-by-Step Guide

## Overview

This guide walks you through the entire project lifecycle: from setting up your environment and acquiring data, through building individual ML pipelines (image classification + RDF prediction), to fusing them into a multi-modal system and delivering a working prototype.

---

## Phase 0: Environment & Project Setup

### 0.1 — Create Project Structure

```
waste-rdf-project/
├── data/
│   ├── raw/                  # Original downloaded datasets
│   │   ├── trashnet/
│   │   └── taco/
│   ├── processed/            # Cleaned, resized, augmented images
│   └── rdf_features/         # Tabular RDF feature CSVs
├── notebooks/                # Jupyter notebooks for EDA & experiments
├── src/
│   ├── data/                 # Data loading & preprocessing
│   │   ├── image_loader.py
│   │   ├── rdf_features.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── cnn_custom.py     # Custom CNN
│   │   ├── mobilenetv2.py    # Transfer learning - MobileNetV2
│   │   ├── resnet50.py       # Transfer learning - ResNet50
│   │   ├── rdf_predictor.py  # Random Forest / XGBoost
│   │   └── multimodal.py     # Fusion model
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualisation.py
│   └── app/                  # Prototype UI (Streamlit/Gradio)
│       └── app.py
├── models/                   # Saved trained models (.h5, .pkl)
├── results/                  # Plots, tables, reports
├── tests/                    # Unit tests
├── requirements.txt
├── README.md
└── config.yaml               # Hyperparameters & paths
```

### 0.2 — Install Dependencies

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# Install core libraries
pip install tensorflow keras torch torchvision   # Pick TF or PyTorch (guide uses TF/Keras)
pip install scikit-learn xgboost
pip install pandas numpy matplotlib seaborn
pip install opencv-python Pillow albumentations
pip install streamlit gradio                      # For prototype UI
pip install jupyter notebook
pip install pyyaml tqdm
```

> [!TIP]
> **Framework choice**: TensorFlow/Keras is slightly easier for transfer learning with MobileNetV2/ResNet50 (built-in `tf.keras.applications`). PyTorch is equally valid — use whichever you're more comfortable with.

---

## Phase 1: Data Acquisition & Understanding

### 1.1 — Download Datasets

#### TrashNet
- **Source**: [TrashNet on GitHub](https://github.com/garythung/trashnet)
- **Contents**: ~2,527 images across 6 classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- **Format**: RGB images, various sizes

```python
# Download TrashNet (or clone the repo)
# git clone https://github.com/garythung/trashnet.git data/raw/trashnet
```

#### TACO (Trash Annotations in Context)
- **Source**: [TACO Dataset](http://tacodataset.org/)
- **Contents**: ~1,500 images with 60 categories (COCO-format annotations)
- **Key advantage**: Real-world, in-context photos (not studio shots)

```python
# Clone TACO
# git clone https://github.com/pedropro/TACO.git data/raw/taco
# python data/raw/taco/download.py
```

> [!IMPORTANT]
> **Dataset Harmonisation**: TrashNet and TACO use different class taxonomies. You must map them into a **unified label set** aligned with RDF-relevant material categories. A recommended unified set:
> 
> | Unified Class | TrashNet Source | TACO Source | RDF Relevance |
> |---|---|---|---|
> | Cardboard | cardboard | Cardboard | High (combustible) |
> | Paper | paper | Paper, Magazine, etc. | High (combustible) |
> | Plastic | plastic | Plastic bag, Bottle, etc. | Medium-High |
> | Metal | metal | Can, Aluminium foil, etc. | Low (non-combustible) |
> | Glass | glass | Glass bottle, jar, etc. | Low (non-combustible) |
> | Organic/Other | trash | Food waste, Cigarette, etc. | Variable |

### 1.2 — Exploratory Data Analysis (EDA)

Create a notebook `notebooks/01_eda.ipynb`:

```python
import os
import matplotlib.pyplot as plt
from collections import Counter

# Count images per class
class_counts = {}
for cls in os.listdir('data/raw/trashnet/dataset-resized'):
    path = f'data/raw/trashnet/dataset-resized/{cls}'
    class_counts[cls] = len(os.listdir(path))

# Visualise distribution
plt.bar(class_counts.keys(), class_counts.values())
plt.title('TrashNet Class Distribution')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/class_distribution.png')
plt.show()
```

**Key EDA tasks:**
- [ ] Class distribution (check for imbalance)
- [ ] Image size distribution (min, max, mean resolution)
- [ ] Sample visualisation (grid of random samples per class)
- [ ] Check for corrupt or unreadable images
- [ ] Assess image quality and variability

---

## Phase 2: Image Classification Pipeline

### 2.1 — Data Preprocessing

```python
# src/data/image_loader.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224  # Standard for MobileNetV2 / ResNet50
BATCH_SIZE = 32

def create_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Create train/validation/test generators with augmentation."""
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 80/20 train/val split
    )
    
    # Validation/Test data — no augmentation, only rescale
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_gen, val_gen
```

> [!TIP]
> **Data split strategy**: Use a 70/15/15 or 80/10/10 train/val/test split. Keep the test set completely unseen until final evaluation. Use stratified splitting to preserve class proportions.

### 2.2 — Model 1: Custom CNN (Baseline)

```python
# src/models/cnn_custom.py
from tensorflow.keras import layers, models

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=6):
    """Baseline custom CNN for waste classification."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier head
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 2.3 — Model 2: MobileNetV2 (Transfer Learning)

```python
# src/models/mobilenetv2.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=6):
    """MobileNetV2 with transfer learning — lightweight & efficient."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 2.4 — Model 3: ResNet50 (Transfer Learning)

```python
# src/models/resnet50.py
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_resnet50(input_shape=(224, 224, 3), num_classes=6):
    """ResNet50 with transfer learning — deeper architecture."""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 2.5 — Training Strategy

```python
# notebooks/02_train_image_models.ipynb
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('models/best_model.h5', save_best_only=True)
]

# Phase 1: Train with frozen base (transfer learning models)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# Phase 2: Fine-tuning — unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Keep early layers frozen
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)
```

> [!IMPORTANT]
> **Fine-tuning is critical** for good transfer learning performance. Always do two-phase training:
> 1. **Phase 1**: Frozen base, train only the new classifier head (higher LR ~1e-3)
> 2. **Phase 2**: Unfreeze top N layers of the base, train end-to-end (lower LR ~1e-5)

---

## Phase 3: RDF Suitability Prediction (Tabular)

### 3.1 — Generate / Prepare RDF Feature Dataset

Since real-world RDF material data may not be publicly available, you'll need to create a **synthetic but realistic** dataset based on domain knowledge from literature.

```python
# src/data/rdf_features.py
import pandas as pd
import numpy as np

def generate_rdf_dataset(n_samples=3000, random_state=42):
    """
    Generate synthetic RDF feature dataset based on material science literature.
    
    Features:
    - material_type: categorical (mapped from image classification)
    - moisture_content: % (0-80)
    - contamination_level: scale 0-10 (0=clean, 10=heavily contaminated)
    - combustibility: scale 0-10
    - calorific_value: MJ/kg (0-46)
    
    Target:
    - rdf_suitable: binary (1=suitable, 0=not suitable)
    - rdf_grade: multi-class (High, Medium, Low, Unsuitable)
    """
    np.random.seed(random_state)
    
    # Material-specific realistic ranges (from waste management literature)
    material_profiles = {
        'cardboard': {
            'moisture': (5, 25), 'contamination': (0, 4),
            'combustibility': (7, 10), 'calorific': (15, 18)
        },
        'paper': {
            'moisture': (5, 30), 'contamination': (0, 5),
            'combustibility': (7, 10), 'calorific': (13, 17)
        },
        'plastic': {
            'moisture': (0, 10), 'contamination': (0, 6),
            'combustibility': (8, 10), 'calorific': (30, 46)
        },
        'metal': {
            'moisture': (0, 5), 'contamination': (0, 3),
            'combustibility': (0, 1), 'calorific': (0, 0.5)
        },
        'glass': {
            'moisture': (0, 5), 'contamination': (0, 3),
            'combustibility': (0, 0), 'calorific': (0, 0)
        },
        'organic': {
            'moisture': (40, 80), 'contamination': (3, 10),
            'combustibility': (2, 6), 'calorific': (3, 8)
        }
    }
    
    records = []
    materials = list(material_profiles.keys())
    
    for _ in range(n_samples):
        mat = np.random.choice(materials)
        profile = material_profiles[mat]
        
        moisture = np.random.uniform(*profile['moisture'])
        contamination = np.random.uniform(*profile['contamination'])
        combustibility = np.random.uniform(*profile['combustibility'])
        calorific = np.random.uniform(*profile['calorific'])
        
        # RDF suitability logic (domain-driven rules with noise)
        # High-quality RDF: high calorific, low moisture, low contamination, high combustibility
        rdf_score = (
            0.35 * (calorific / 46) +
            0.25 * (1 - moisture / 80) +
            0.20 * (combustibility / 10) +
            0.20 * (1 - contamination / 10)
        )
        rdf_score += np.random.normal(0, 0.05)  # Add noise
        rdf_score = np.clip(rdf_score, 0, 1)
        
        rdf_suitable = 1 if rdf_score >= 0.45 else 0
        
        if rdf_score >= 0.7:
            rdf_grade = 'High'
        elif rdf_score >= 0.5:
            rdf_grade = 'Medium'
        elif rdf_score >= 0.35:
            rdf_grade = 'Low'
        else:
            rdf_grade = 'Unsuitable'
        
        records.append({
            'material_type': mat,
            'moisture_content': round(moisture, 2),
            'contamination_level': round(contamination, 2),
            'combustibility': round(combustibility, 2),
            'calorific_value': round(calorific, 2),
            'rdf_score': round(rdf_score, 4),
            'rdf_suitable': rdf_suitable,
            'rdf_grade': rdf_grade
        })
    
    return pd.DataFrame(records)

# Generate and save
df = generate_rdf_dataset(n_samples=3000)
df.to_csv('data/rdf_features/rdf_dataset.csv', index=False)
print(df.head())
print(df['rdf_grade'].value_counts())
```

> [!NOTE]
> **Why synthetic data?** Real industrial RDF characterisation data is rarely public. Using domain-knowledge-based synthetic data is a common and accepted approach in research prototyping. Be transparent about this in your dissertation — frame it as a proof-of-concept that could be validated with real data in future work.

### 3.2 — RDF Prediction Models

```python
# src/models/rdf_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

def train_rdf_models(csv_path='data/rdf_features/rdf_dataset.csv'):
    df = pd.read_csv(csv_path)
    
    # Encode categorical features
    le_material = LabelEncoder()
    df['material_encoded'] = le_material.fit_transform(df['material_type'])
    
    # Features and targets
    feature_cols = ['material_encoded', 'moisture_content', 'contamination_level',
                    'combustibility', 'calorific_value']
    X = df[feature_cols]
    
    # Target 1: Binary suitability
    y_binary = df['rdf_suitable']
    
    # Target 2: Multi-class grade
    le_grade = LabelEncoder()
    y_grade = le_grade.fit_transform(df['rdf_grade'])
    
    # Split
    X_train, X_test, y_train_b, y_test_b = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    _, _, y_train_g, y_test_g = train_test_split(
        X, y_grade, test_size=0.2, random_state=42, stratify=y_grade
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ─── Random Forest ───
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train_b)
    
    print("=== Random Forest (Binary) ===")
    print(f"Best params: {rf.best_params_}")
    print(classification_report(y_test_b, rf.predict(X_test_scaled)))
    
    # ─── XGBoost ───
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb_model = GridSearchCV(
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        xgb_params,
        cv=StratifiedKFold(5),
        scoring='f1_weighted',
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train_g)
    
    print("=== XGBoost (Multi-class Grade) ===")
    print(f"Best params: {xgb_model.best_params_}")
    print(classification_report(y_test_g, xgb_model.predict(X_test_scaled),
                                target_names=le_grade.classes_))
    
    # Save models
    joblib.dump(rf.best_estimator_, 'models/rf_rdf_binary.pkl')
    joblib.dump(xgb_model.best_estimator_, 'models/xgb_rdf_grade.pkl')
    joblib.dump(scaler, 'models/rdf_scaler.pkl')
    joblib.dump(le_material, 'models/le_material.pkl')
    joblib.dump(le_grade, 'models/le_grade.pkl')
    
    return rf, xgb_model
```

### 3.3 — Feature Importance Analysis

```python
# This is crucial for your dissertation — show which features matter most
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
```

---

## Phase 4: Multi-Modal Fusion (The Core Contribution)

This is the **most important part** of your project — it directly addresses your research questions.

### 4.1 — Fusion Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   Waste Image    │     │  Material Features│
│   (224×224×3)    │     │  (5 features)     │
└────────┬────────┘     └────────┬──────────┘
         │                       │
    ┌────▼────┐            ┌─────▼─────┐
    │MobileNet│            │  Dense    │
    │  V2     │            │  Network  │
    │ (frozen)│            │  (64→32)  │
    └────┬────┘            └─────┬─────┘
         │                       │
    ┌────▼────┐            ┌─────▼─────┐
    │  GAP    │            │  Feature  │
    │  +Dense │            │  Vector   │
    │  (256)  │            │  (32)     │
    └────┬────┘            └─────┬─────┘
         │                       │
         └───────────┬───────────┘
                     │ Concatenate
              ┌──────▼──────┐
              │   Fusion    │
              │   Dense     │
              │  (256→128)  │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Output    │
              │  (Softmax)  │
              └─────────────┘
```

### 4.2 — Implementation

```python
# src/models/multimodal.py
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2

def build_multimodal_model(num_classes=6, num_tabular_features=5):
    """
    Multi-modal fusion model combining image features (CNN) 
    with tabular RDF material features.
    """
    
    # ─── Image Branch ───
    image_input = layers.Input(shape=(224, 224, 3), name='image_input')
    
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_cnn.trainable = False  # Freeze initially
    
    x_img = base_cnn(image_input)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(256, activation='relu')(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.4)(x_img)
    
    # ─── Tabular Branch ───
    tabular_input = layers.Input(shape=(num_tabular_features,), name='tabular_input')
    
    x_tab = layers.Dense(64, activation='relu')(tabular_input)
    x_tab = layers.BatchNormalization()(x_tab)
    x_tab = layers.Dropout(0.3)(x_tab)
    x_tab = layers.Dense(32, activation='relu')(x_tab)
    x_tab = layers.BatchNormalization()(x_tab)
    
    # ─── Fusion ───
    fused = layers.Concatenate()([x_img, x_tab])
    
    x = layers.Dense(256, activation='relu')(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # ─── Dual Output Heads ───
    # Head 1: Waste material classification
    waste_output = layers.Dense(num_classes, activation='softmax', name='waste_class')(x)
    
    # Head 2: RDF suitability (binary)
    rdf_output = layers.Dense(1, activation='sigmoid', name='rdf_suitable')(x)
    
    model = Model(
        inputs=[image_input, tabular_input],
        outputs=[waste_output, rdf_output]
    )
    
    model.compile(
        optimizer='adam',
        loss={
            'waste_class': 'categorical_crossentropy',
            'rdf_suitable': 'binary_crossentropy'
        },
        loss_weights={'waste_class': 0.6, 'rdf_suitable': 0.4},
        metrics={
            'waste_class': 'accuracy',
            'rdf_suitable': 'accuracy'
        }
    )
    
    return model
```

### 4.3 — Custom Data Generator for Multi-Modal Training

```python
# src/data/multimodal_generator.py
import numpy as np
import tensorflow as tf

class MultiModalGenerator(tf.keras.utils.Sequence):
    """Generator that yields (image, tabular_features) pairs with labels."""
    
    def __init__(self, image_paths, tabular_features, waste_labels, rdf_labels,
                 batch_size=32, img_size=224, augment=False):
        self.image_paths = image_paths
        self.tabular_features = tabular_features
        self.waste_labels = waste_labels
        self.rdf_labels = rdf_labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.indices = np.arange(len(image_paths))
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        for i in batch_indices:
            img = tf.keras.preprocessing.image.load_img(
                self.image_paths[i], target_size=(self.img_size, self.img_size)
            )
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            batch_images.append(img)
        
        batch_images = np.array(batch_images)
        batch_tabular = self.tabular_features[batch_indices]
        batch_waste = self.waste_labels[batch_indices]
        batch_rdf = self.rdf_labels[batch_indices]
        
        return (
            {'image_input': batch_images, 'tabular_input': batch_tabular},
            {'waste_class': batch_waste, 'rdf_suitable': batch_rdf}
        )
```

---

## Phase 5: Comprehensive Evaluation

### 5.1 — Evaluation Metrics

```python
# src/evaluation/metrics.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(y_true, y_pred, y_prob=None, class_names=None, title='Model'):
    """Comprehensive evaluation with visualisations."""
    
    # Classification Report
    print(f"\n{'='*60}")
    print(f"  {title} — Classification Report")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{title} — Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}_cm.png', dpi=150)
    plt.show()
    
    # ROC Curve (binary)
    if y_prob is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title} — ROC Curve')
        plt.legend()
        plt.savefig(f'results/{title.lower().replace(" ", "_")}_roc.png', dpi=150)
        plt.show()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
```

### 5.2 — Comparative Analysis (Critical for Research Questions)

```python
# notebooks/05_comparison.ipynb

# Compare all models systematically
results = {}

# Image-only models
results['Custom CNN'] = evaluate_model(y_test, cnn_preds, class_names=class_names, title='Custom CNN')
results['MobileNetV2'] = evaluate_model(y_test, mobilenet_preds, class_names=class_names, title='MobileNetV2')
results['ResNet50'] = evaluate_model(y_test, resnet_preds, class_names=class_names, title='ResNet50')

# Tabular-only models
results['Random Forest'] = evaluate_model(y_test_rdf, rf_preds, title='Random Forest')
results['XGBoost'] = evaluate_model(y_test_rdf, xgb_preds, title='XGBoost')

# Multi-modal
results['Multi-Modal'] = evaluate_model(y_test_mm, mm_preds, title='Multi-Modal Fusion')

# Summary comparison table
import pandas as pd
comparison_df = pd.DataFrame(results).T
comparison_df.to_csv('results/model_comparison.csv')
print(comparison_df.to_markdown())
```

**Expected comparison table format:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Custom CNN | 0.82 | 0.83 | 0.82 | 0.82 |
| MobileNetV2 | 0.91 | 0.91 | 0.91 | 0.91 |
| ResNet50 | 0.89 | 0.90 | 0.89 | 0.89 |
| Random Forest (tabular) | 0.87 | 0.88 | 0.87 | 0.87 |
| XGBoost (tabular) | 0.90 | 0.90 | 0.90 | 0.90 |
| **Multi-Modal Fusion** | **0.94** | **0.94** | **0.94** | **0.94** |

> [!TIP]
> **Answering your research questions**: This comparison table directly answers all three research questions. The multi-modal model should demonstrate improved performance over image-only approaches, validating your hypothesis.

---

## Phase 6: Prototype Application

### 6.1 — Streamlit Web App

```python
# src/app/app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

# ─── Page Config ───
st.set_page_config(
    page_title="Waste RDF Classifier",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Multi-Modal Waste Characterisation for RDF Production")
st.markdown("Upload a waste image and provide material characteristics to assess RDF suitability.")

# ─── Load Models ───
@st.cache_resource
def load_models():
    image_model = tf.keras.models.load_model('models/best_mobilenetv2.h5')
    rdf_model = joblib.load('models/xgb_rdf_grade.pkl')
    scaler = joblib.load('models/rdf_scaler.pkl')
    multimodal = tf.keras.models.load_model('models/multimodal_model.h5')
    return image_model, rdf_model, scaler, multimodal

image_model, rdf_model, scaler, multimodal = load_models()

CLASS_NAMES = ['Cardboard', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']

# ─── Sidebar: Material Features ───
st.sidebar.header("📋 Material Characteristics")
moisture = st.sidebar.slider("Moisture Content (%)", 0.0, 80.0, 15.0, 0.5)
contamination = st.sidebar.slider("Contamination Level (0-10)", 0.0, 10.0, 2.0, 0.1)
combustibility = st.sidebar.slider("Combustibility (0-10)", 0.0, 10.0, 7.0, 0.1)
calorific = st.sidebar.slider("Calorific Value (MJ/kg)", 0.0, 46.0, 15.0, 0.5)

# ─── Main Area: Image Upload ───
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Waste Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file is not None:
        # Process image
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Image-only prediction
        img_pred = image_model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(img_pred)]
        confidence = np.max(img_pred) * 100
        
        st.subheader("🔍 Classification Results")
        st.metric("Predicted Material", predicted_class)
        st.metric("Confidence", f"{confidence:.1f}%")
        
        # RDF Assessment
        st.subheader("⚡ RDF Suitability Assessment")
        
        material_encoded = CLASS_NAMES.index(predicted_class)
        features = np.array([[material_encoded, moisture, contamination,
                              combustibility, calorific]])
        features_scaled = scaler.transform(features)
        
        rdf_grade = rdf_model.predict(features_scaled)[0]
        grade_names = ['High', 'Low', 'Medium', 'Unsuitable']
        
        grade_colors = {'High': '🟢', 'Medium': '🟡', 'Low': '🟠', 'Unsuitable': '🔴'}
        grade_name = grade_names[rdf_grade]
        
        st.metric("RDF Grade", f"{grade_colors.get(grade_name, '')} {grade_name}")
        
        # Show probabilities
        st.bar_chart(dict(zip(CLASS_NAMES, img_pred[0])))
```

### 6.2 — Run the Prototype

```bash
streamlit run src/app/app.py
```

---

## Phase 7: Dissertation Writing Structure

Map your code work directly to dissertation chapters:

| Chapter | Content | Key Code/Results |
|---|---|---|
| 1. Introduction | Problem, aims, objectives, scope | — |
| 2. Literature Review | Background on waste classification, RDF, CNNs, multi-modal ML | — |
| 3. Methodology | System design, algorithms, data pipeline, evaluation plan | Architecture diagrams |
| 4. Implementation | Code walkthrough, design decisions, challenges | Code snippets from `src/` |
| 5. Results & Evaluation | Model comparisons, metrics, visualisations | Tables, confusion matrices, ROC curves |
| 6. Discussion | Analysis of results vs. research questions, limitations | Comparison table |
| 7. Conclusion | Summary, contributions, future work | — |

---

## Key Tips for Success

### Research Quality
- [ ] **Statistical rigour**: Run experiments with multiple random seeds (3-5) and report mean ± std
- [ ] **Cross-validation**: Use k-fold (k=5) for tabular models, not just a single train/test split
- [ ] **Ablation study**: Show what each modality contributes by comparing image-only, tabular-only, and multi-modal
- [ ] **Baseline comparison**: Always include a simple baseline (e.g., majority class) for context

### Common Pitfalls to Avoid
- [ ] **Data leakage**: Ensure augmentation only on training data; test set stays clean
- [ ] **Overfitting**: Monitor train vs. validation loss curves; use early stopping
- [ ] **Class imbalance**: Use class weights, SMOTE, or focal loss if classes are imbalanced
- [ ] **Unfair comparisons**: Ensure all models are evaluated on the **exact same** test set

### Performance Optimisation
- [ ] Use **mixed precision training** (`tf.keras.mixed_precision`) for faster GPU training
- [ ] Use **TF Data pipeline** (`tf.data.Dataset`) instead of `ImageDataGenerator` for better performance
- [ ] Cache processed datasets to disk to avoid reprocessing

### Timeline Suggestion (12 weeks)

| Week | Task |
|---|---|
| 1-2 | Literature review, environment setup, data acquisition |
| 3-4 | EDA, data preprocessing, baseline CNN training |
| 5-6 | Transfer learning (MobileNetV2, ResNet50), hyperparameter tuning |
| 7 | RDF feature engineering, tabular model training |
| 8-9 | Multi-modal fusion model, training & evaluation |
| 10 | Comprehensive comparison, ablation study |
| 11 | Prototype development (Streamlit app) |
| 12 | Final evaluation, dissertation writing, documentation |

> [!CAUTION]
> **Start early on the multi-modal fusion** (Phase 4). It's the hardest and most novel part of your project. Building the data generator that pairs images with their corresponding tabular features correctly is where most bugs occur. Test this thoroughly with small batches before full training.
