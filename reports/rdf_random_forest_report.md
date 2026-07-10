# Random Forest RDF Suitability Report

**Project:** Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production  
**Generated:** 2026-07-02 19:54:20

---

## Dataset Summary

- Samples: 3000
- Class balance: {0: 1210, 1: 1790}
- Features: material_type, moisture_content, contamination_level, combustibility, calorific_value
- Target: `rdf_suitable`

## Preprocessing

The tabular pipeline applies:
- median imputation for numeric features
- most-frequent imputation for categorical features
- one-hot encoding for `material_type`
- standard scaling for numeric columns

## Model Configuration

- Estimator: Random Forest Classifier
- Best parameters: {
  "classifier__max_depth": 10,
  "classifier__min_samples_leaf": 4,
  "classifier__min_samples_split": 10,
  "classifier__n_estimators": 300
}
- Cross-validation score: 0.9088

## Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9133 |
| Precision | 0.9276 |
| Recall | 0.9133 |
| F1-score | 0.9141 |

## Classification Report

```text
              precision    recall  f1-score   support

Not Suitable     0.8253    0.9959    0.9026       242
    Suitable     0.9968    0.8575    0.9219       358

    accuracy                         0.9133       600
   macro avg     0.9110    0.9267    0.9123       600
weighted avg     0.9276    0.9133    0.9141       600

```

## Artifacts

- Model pipeline: `D:\University\Msc Project\models\rdf_random_forest_pipeline.joblib`
- Confusion matrix: `D:\University\Msc Project\results\rdf_confusion_matrix.png`
- Feature importance: `D:\University\Msc Project\results\rdf_feature_importance.png`
- Results directory: `D:\University\Msc Project\results`

## Observations

- Material type is expected to be one of the strongest predictors because it encodes RDF-relevant composition.
- Moisture and contamination should penalize suitability because they lower fuel quality.
- Combustibility and calorific value should contribute positively to the suitability score.

## Limitations

- The dataset is synthetic and based on domain-informed rules, not real plant measurements.
- The model is intentionally isolated from the image classifier for now.
- Future work should validate the feature engineering on industrial RDF records.
