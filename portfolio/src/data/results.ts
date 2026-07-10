/**
 * Evaluation metrics copied directly from:
 *   - reports/cnn_baseline_report.md
 *   - reports/cnn_mobilenetv2_resnet50_evaluation_report.md
 *   - reports/rdf_random_forest_report.md
 *   - results/baseline_cnn_classification_report.txt
 *   - results/mobilenetv2_classification_report.txt
 *   - results/resnet50_classification_report.txt
 *   - results/rdf_classification_report.txt
 */

export interface ModelResult {
  id: string;
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  notes?: string;
}

export const imageModelResults: ModelResult[] = [
  {
    id: "baseline",
    name: "Baseline CNN",
    accuracy: 0.5744,
    precision: 0.5768,
    recall: 0.5744,
    f1: 0.5723,
    notes: "27 epochs, dropout 0.5, train acc 0.8349 / val acc 0.6089 (final).",
  },
  {
    id: "mobilenetv2",
    name: "MobileNetV2",
    accuracy: 0.8198,
    precision: 0.8281,
    recall: 0.8198,
    f1: 0.8206,
    notes: "Frozen head + fine-tune top 30 backbone layers at lr 1e-5.",
  },
  {
    id: "resnet50",
    name: "ResNet50",
    accuracy: 0.8877,
    precision: 0.8906,
    recall: 0.8877,
    f1: 0.8878,
    notes: "Best performer. Two-stage training identical to MobileNetV2.",
  },
];

export const imageModelDeltas = {
  mobilenetv2VsBaseline: {
    accuracy: 0.2454,
    precision: 0.2513,
    recall: 0.2454,
    f1: 0.2483,
  },
  resnet50VsBaseline: {
    accuracy: 0.3133,
    precision: 0.3138,
    recall: 0.3133,
    f1: 0.3155,
  },
};

export const perClassResnet50: Record<
  string,
  { precision: number; recall: number; f1: number; support: number }
> = {
  cardboard: { precision: 0.9815, recall: 0.8689, f1: 0.9217, support: 61 },
  glass: { precision: 0.8846, recall: 0.9079, f1: 0.8961, support: 76 },
  metal: { precision: 0.8621, recall: 0.8065, f1: 0.8333, support: 62 },
  paper: { precision: 0.8776, recall: 0.9556, f1: 0.9149, support: 90 },
  plastic: { precision: 0.9014, recall: 0.8767, f1: 0.8889, support: 73 },
  trash: { precision: 0.75, recall: 0.8571, f1: 0.8, support: 21 },
};

export const rdfModelResult: ModelResult = {
  id: "rdf-rf",
  name: "RDF Random Forest",
  accuracy: 0.9133,
  precision: 0.9276,
  recall: 0.9133,
  f1: 0.9141,
  notes:
    "Best params: n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=4. 5-fold CV F1 (weighted) = 0.9088.",
};

export const rdfClassBreakdown = {
  notSuitable: { precision: 0.8253, recall: 0.9959, f1: 0.9026, support: 242 },
  suitable: { precision: 0.9968, recall: 0.8575, f1: 0.9219, support: 358 },
};
