/**
 * Raw markdown reports copied from the project's reports/ directory
 * at build time. They are imported as ?raw strings so Vite bundles them
 * into the client without a runtime fetch.
 */

import datasetReport from "@/content/reports/dataset_report.md?raw";
import preprocessingReport from "@/content/reports/preprocessing_report.md?raw";
import cnnBaselineReport from "@/content/reports/cnn_baseline_report.md?raw";
import cnnMobileResnetReport from "@/content/reports/cnn_mobilenetv2_resnet50_evaluation_report.md?raw";
import rdfRandomForestReport from "@/content/reports/rdf_random_forest_report.md?raw";

export interface ReportDoc {
  id: string;
  title: string;
  description: string;
  source: string;
  generated: string;
  category:
    | "Dataset"
    | "Preprocessing"
    | "Model"
    | "Comparison"
    | "Tabular"
    | "Reference";
  markdown: string;
}

export const reports: ReportDoc[] = [
  {
    id: "dataset",
    title: "Dataset Exploration and Validation Report",
    description:
      "Auto-detection of TrashNet, class distribution, image resolution statistics and data-quality audit.",
    source: "reports/dataset_report.md",
    generated: "2026-06-26",
    category: "Dataset",
    markdown: datasetReport,
  },
  {
    id: "preprocessing",
    title: "Data Preprocessing Report",
    description:
      "Stratified 70 / 15 / 15 split, 224 x 224 RGB output, [0, 1] float32 normalisation, 4.37:1 imbalance analysis.",
    source: "reports/preprocessing_report.md",
    generated: "2026-06-26",
    category: "Preprocessing",
    markdown: preprocessingReport,
  },
  {
    id: "cnn-baseline",
    title: "CNN Baseline Model Training Report",
    description:
      "From-scratch 3-block CNN (77 M params), 27 epochs, dropout 0.5. Test accuracy 0.5744.",
    source: "reports/cnn_baseline_report.md",
    generated: "2026-07-02",
    category: "Model",
    markdown: cnnBaselineReport,
  },
  {
    id: "cnn-mobilenetv2-resnet50",
    title: "Comparative Evaluation: Baseline CNN, MobileNetV2 and ResNet50",
    description:
      "Three-model test-set comparison. ResNet50 F1 0.8878 (+0.3155 vs baseline).",
    source: "reports/cnn_mobilenetv2_resnet50_evaluation_report.md",
    generated: "2026-07-02",
    category: "Comparison",
    markdown: cnnMobileResnetReport,
  },
  {
    id: "rdf-random-forest",
    title: "Random Forest RDF Suitability Report",
    description:
      "Tabular RF on 3,000 synthetic RDF rows. Best params, 5-fold CV F1 0.9088, test accuracy 0.9133.",
    source: "reports/rdf_random_forest_report.md",
    generated: "2026-07-02",
    category: "Tabular",
    markdown: rdfRandomForestReport,
  },
];
