"""
Tabular RDF suitability preprocessing utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "material_type",
    "moisture_content",
    "contamination_level",
    "combustibility",
    "calorific_value",
]
TARGET_COLUMN = "rdf_suitable"
GRADE_COLUMN = "rdf_grade"


@dataclass(frozen=True)
class RDFDataConfig:
    """Configuration for RDF tabular data handling."""

    csv_path: Path = Path("data/rdf_features/rdf_dataset.csv")
    output_dir: Path = Path("data/rdf_features")
    random_seed: int = 42
    n_samples: int = 3000
    test_size: float = 0.2


class RDFPreprocessingPipeline:
    """Generate, load, split, and preprocess RDF suitability data."""

    material_profiles = {
        "cardboard": {
            "moisture": (5, 25),
            "contamination": (0, 4),
            "combustibility": (7, 10),
            "calorific": (15, 18),
        },
        "paper": {
            "moisture": (5, 30),
            "contamination": (0, 5),
            "combustibility": (7, 10),
            "calorific": (13, 17),
        },
        "plastic": {
            "moisture": (0, 10),
            "contamination": (0, 6),
            "combustibility": (8, 10),
            "calorific": (30, 46),
        },
        "metal": {
            "moisture": (0, 5),
            "contamination": (0, 3),
            "combustibility": (0, 1),
            "calorific": (0, 0.5),
        },
        "glass": {
            "moisture": (0, 5),
            "contamination": (0, 3),
            "combustibility": (0, 0),
            "calorific": (0, 0),
        },
        "organic": {
            "moisture": (40, 80),
            "contamination": (3, 10),
            "combustibility": (2, 6),
            "calorific": (3, 8),
        },
    }

    def __init__(self, config: RDFDataConfig = RDFDataConfig()) -> None:
        self.config = config

    def generate_dataset(self, n_samples: int | None = None) -> pd.DataFrame:
        """Generate a synthetic but realistic RDF suitability dataset."""
        sample_count = n_samples or self.config.n_samples
        rng = np.random.default_rng(self.config.random_seed)
        records: List[dict] = []

        materials = list(self.material_profiles.keys())
        for _ in range(sample_count):
            material = rng.choice(materials)
            profile = self.material_profiles[material]

            moisture = rng.uniform(*profile["moisture"])
            contamination = rng.uniform(*profile["contamination"])
            combustibility = rng.uniform(*profile["combustibility"])
            calorific = rng.uniform(*profile["calorific"])

            rdf_score = (
                0.35 * (calorific / 46)
                + 0.25 * (1 - moisture / 80)
                + 0.20 * (combustibility / 10)
                + 0.20 * (1 - contamination / 10)
            )
            rdf_score += rng.normal(0, 0.05)
            rdf_score = float(np.clip(rdf_score, 0, 1))

            rdf_suitable = 1 if rdf_score >= 0.45 else 0
            if rdf_score >= 0.7:
                rdf_grade = "High"
            elif rdf_score >= 0.5:
                rdf_grade = "Medium"
            elif rdf_score >= 0.35:
                rdf_grade = "Low"
            else:
                rdf_grade = "Unsuitable"

            records.append(
                {
                    "material_type": material,
                    "moisture_content": round(float(moisture), 2),
                    "contamination_level": round(float(contamination), 2),
                    "combustibility": round(float(combustibility), 2),
                    "calorific_value": round(float(calorific), 2),
                    "rdf_score": round(rdf_score, 4),
                    "rdf_suitable": rdf_suitable,
                    "rdf_grade": rdf_grade,
                }
            )

        return pd.DataFrame(records)

    def save_dataset(self, df: pd.DataFrame) -> Path:
        """Persist the tabular RDF dataset to disk."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.config.csv_path, index=False)
        return self.config.csv_path

    def load_or_generate_dataset(self) -> pd.DataFrame:
        """Load the RDF dataset or generate it if it does not exist."""
        if self.config.csv_path.exists():
            return pd.read_csv(self.config.csv_path)

        df = self.generate_dataset()
        self.save_dataset(df)
        return df

    def split_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = TARGET_COLUMN,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train/test splits with stratification."""
        features = df[FEATURE_COLUMNS].copy()
        target = df[target_column].copy()
        return train_test_split(
            features,
            target,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=target,
        )

    def build_preprocessor(self) -> ColumnTransformer:
        """Build a reusable tabular preprocessing pipeline."""
        categorical_features = ["material_type"]
        numeric_features = [
            "moisture_content",
            "contamination_level",
            "combustibility",
            "calorific_value",
        ]

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("categorical", categorical_pipeline, categorical_features),
                ("numeric", numeric_pipeline, numeric_features),
            ]
        )
