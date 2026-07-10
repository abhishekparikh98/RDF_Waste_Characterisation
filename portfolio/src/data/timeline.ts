/**
 * Real Git timeline. Source: git log on the project's master branch.
 * Only 5 commits exist on the project; the dates below come from
 * `git log --format="%H %s %ad" --date=iso`.
 */

export interface Commit {
  hash: string;
  shortHash: string;
  date: string;
  subject: string;
  description: string;
  scope: string[];
}

export const commits: Commit[] = [
  {
    hash: "565928c04f5e3601854428d01c44aeafa1edbe16",
    shortHash: "565928c",
    date: "2026-06-26",
    subject: "Initialize MSc project structure",
    description:
      "Created the initial directory layout, README, LICENSE, pyproject.toml, .gitignore, and requirements.txt for the MSc project.",
    scope: ["Repository scaffolding", "Documentation"],
  },
  {
    hash: "91889f77c0882f99beffaec143ebed9b793066b7",
    shortHash: "91889f7",
    date: "2026-06-26",
    subject: "Implement dataset exploration and validation",
    description:
      "Added scripts/explore_dataset.py with DatasetDetector, ImageAnalyzer, DatasetExplorer, VisualizationGenerator and ReportGenerator classes.",
    scope: ["scripts/explore_dataset.py", "reports/dataset_report.md", "reports/figures/trashnet_class_distribution.png"],
  },
  {
    hash: "a42a9ff1761d60bb450f37aa712230cdaf667baf",
    shortHash: "a42a9ff",
    date: "2026-06-26",
    subject: "Implement data preprocessing pipeline",
    description:
      "Added src/preprocessing.py (ImageValidator, ImagePreprocessor, DatasetSplitter, DatasetSaver, PreprocessingPipeline) and scripts/preprocess_dataset.py for stratified 70/15/15 splitting, 224 x 224 RGB PNG output and [0, 1] normalisation.",
    scope: ["src/preprocessing.py", "scripts/preprocess_dataset.py", "data/processed/", "reports/preprocessing_report.md"],
  },
  {
    hash: "e631f516a4d859c5d7027150a204165cfe39e4d3",
    shortHash: "e631f51",
    date: "2026-07-01",
    subject: "Implement baseline CNN classifier",
    description:
      "Added src/models.py (BaselineCNN, build_baseline_cnn), src/training.py (TrainingManager, prepare_dataset), src/evaluation.py (MetricsCalculator, ConfusionMatrixVisualizer, TrainingHistoryVisualizer) and scripts/train_cnn.py with full markdown reporting.",
    scope: ["src/models.py", "src/training.py", "src/evaluation.py", "scripts/train_cnn.py", "models/cnn_baseline_best.h5"],
  },
  {
    hash: "500132a19e8f2a22d1a0d8191df1451e4ac6cfef",
    shortHash: "500132a",
    date: "2026-07-01",
    subject: "Add multimodal inference and Flask app",
    description:
      "Added src/multimodal_inference.py, src/rdf_preprocessing.py, scripts/run_multimodal_inference.py, scripts/train_rdf_rf.py, scripts/compare_cnn_mobilenetv2.py, app.py, templates/index.html, static/style.css, docs/FLASK_WEB_APP_ARCHITECTURE.md and docs/MULTIMODAL_INFERENCE_ARCHITECTURE.md.",
    scope: [
      "src/multimodal_inference.py",
      "src/rdf_preprocessing.py",
      "scripts/run_multimodal_inference.py",
      "scripts/train_rdf_rf.py",
      "scripts/compare_cnn_mobilenetv2.py",
      "app.py",
      "templates/index.html",
      "static/style.css",
      "docs/FLASK_WEB_APP_ARCHITECTURE.md",
      "docs/MULTIMODAL_INFERENCE_ARCHITECTURE.md",
    ],
  },
];

export const phaseTimeline = [
  {
    phase: "Phase 1",
    title: "Project Scaffolding and Dataset Exploration",
    period: "June 2026",
    commits: ["565928c", "91889f7"],
    summary:
      "Repository initialised. Auto-detection and statistics of the TrashNet dataset (2,527 images, 6 classes, 512 x 384 JPEG, 0 corrupted).",
  },
  {
    phase: "Phase 2",
    title: "Data Preprocessing",
    period: "June 2026",
    commits: ["a42a9ff"],
    summary:
      "Stratified 70 / 15 / 15 split (1,763 / 381 / 383), 224 x 224 RGB PNG output, [0, 1] float32 normalisation, 4.37:1 imbalance preserved.",
  },
  {
    phase: "Phase 3",
    title: "Baseline CNN Training",
    period: "July 2026",
    commits: ["e631f51"],
    summary:
      "From-scratch 3-block CNN (77 M params, 25.8 M trainable) trained for 27 epochs with early stopping. Test accuracy 0.5744, weighted F1 0.5723.",
  },
  {
    phase: "Phase 4",
    title: "Multimodal Inference and Deployment",
    period: "July 2026",
    commits: ["500132a"],
    summary:
      "MobileNetV2 and ResNet50 transfer learning (two-stage frozen + fine-tune), Random Forest for RDF suitability, command-line multimodal inference and Flask web demo.",
  },
];
