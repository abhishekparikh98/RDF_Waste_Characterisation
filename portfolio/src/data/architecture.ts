/**
 * Architecture facts copied from:
 *   - src/preprocessing.py
 *   - src/multimodal_inference.py
 *   - src/models.py
 *   - scripts/preprocess_dataset.py
 *   - scripts/run_multimodal_inference.py
 *   - app.py
 *   - docs/MULTIMODAL_INFERENCE_ARCHITECTURE.md
 *   - docs/FLASK_WEB_APP_ARCHITECTURE.md
 */

export const preprocessing = {
  title: "Image Preprocessing Pipeline",
  location: "src/preprocessing.py + scripts/preprocess_dataset.py",
  classes: [
    {
      name: "ImageValidator",
      purpose:
        "Validates image readability with PIL.Image.verify and reports corrupted files.",
    },
    {
      name: "ImagePreprocessor",
      purpose:
        "Loads each image as RGB, resizes to 224 x 224 with Lanczos interpolation, and normalises pixel values to [0, 1].",
    },
    {
      name: "DatasetSplitter",
      purpose:
        "Performs a stratified 70 / 15 / 15 train / validation / test split with random_state = 42.",
    },
    {
      name: "DatasetSaver",
      purpose:
        "Writes the preprocessed images as lossless PNGs under data/processed/{split}/{class}/.",
    },
    {
      name: "PreprocessingPipeline",
      purpose: "Orchestrates the four steps above and returns a results dictionary.",
    },
  ],
  steps: [
    "Validate image integrity",
    "Convert to RGB and discard alpha / grayscale modes",
    "Resize to 224 x 224 with Lanczos interpolation",
    "Normalise to float32 in [0, 1]",
    "Stratified split (70 / 15 / 15, seed 42)",
    "Persist as PNG (lossless)",
  ],
};

export const rdfPreprocessing = {
  title: "RDF Tabular Preprocessing",
  location: "src/rdf_preprocessing.py",
  features: [
    "material_type (categorical)",
    "moisture_content (numeric)",
    "contamination_level (numeric)",
    "combustibility (numeric)",
    "calorific_value (numeric)",
  ],
  pipeline: [
    "Most-frequent imputation for material_type",
    "Median imputation for numeric features",
    "One-hot encoding for material_type",
    "Standard scaling for numeric columns",
  ],
  classes: [
    {
      name: "RDFDataConfig",
      purpose: "Dataclass holding CSV path, output dir, seed, n_samples, test_size.",
    },
    {
      name: "RDFPreprocessingPipeline",
      purpose:
        "Generates the synthetic dataset, saves / loads CSV, performs stratified split, and builds the ColumnTransformer.",
    },
  ],
};

export const multimodalInference = {
  title: "Multimodal Inference Pipeline",
  location: "src/multimodal_inference.py + scripts/run_multimodal_inference.py",
  flow: [
    {
      step: "ImageClassifier",
      file: "src/multimodal_inference.py",
      purpose:
        "Loads a trained Keras model (default: cnn_baseline_best.h5, mode baseline) and predicts one of the six waste classes.",
    },
    {
      step: "MaterialFeatureMapper",
      file: "src/multimodal_inference.py",
      purpose:
        "Looks the predicted class up in MATERIAL_FEATURE_LIBRARY and produces a one-row DataFrame with material_type, moisture_content, contamination_level, combustibility, calorific_value.",
    },
    {
      step: "RDFSuitabilityPredictor",
      file: "src/multimodal_inference.py",
      purpose:
        "Loads rdf_random_forest_pipeline.joblib and predicts both the binary suitability label and the probability of being Suitable.",
    },
    {
      step: "MultimodalInferencePipeline",
      file: "src/multimodal_inference.py",
      purpose:
        "Chains the three components and returns an InferenceResult dataclass containing the image path, predicted class, confidence, material features, suitability label and probability.",
    },
  ],
  cli: {
    file: "scripts/run_multimodal_inference.py",
    usage: "python scripts/run_multimodal_inference.py --image PATH [--output FILE]",
    output: "JSON with image_path, predicted_class, class_confidence, material_features, rdf_suitability, rdf_probability, rdf_label.",
  },
};

export const flaskApp = {
  title: "Flask Demo Application",
  location: "app.py + templates/index.html + static/style.css",
  route: "/",
  method: "GET / POST",
  validation: [
    "Allowed extensions: .jpg, .jpeg, .png, .bmp, .gif, .tiff",
    "MIME prefix: image/",
    "Max upload: 16 MB",
  ],
  flow: [
    "User uploads an image through the form in templates/index.html",
    "Flask validates the extension and content type",
    "The file is saved to a NamedTemporaryFile",
    "MultimodalInferencePipeline.infer() is called",
    "The result (predicted class, confidence, RDF label, RDF probability) is rendered on the same page along with a base64 preview",
  ],
  pipelineCache:
    "get_pipeline() is wrapped in functools.lru_cache so the models are loaded once per process.",
};

export const models: ModelSpec[] = [
  {
    id: "baseline",
    name: "Baseline CNN",
    type: "From-scratch CNN",
    file: "src/models.py -> build_baseline_cnn",
    params: "77.36 M total, 25.79 M trainable",
    description:
      "Three Conv -> ReLU -> MaxPool -> Dropout blocks (32, 64, 128 filters), a 256-unit dense hidden layer and a 6-way softmax head.",
    image: "/figures/baseline_confusion_matrix.png",
    trainCurves: {
      accuracy: "/figures/baseline_accuracy.png",
      loss: "/figures/baseline_loss.png",
    },
  },
  {
    id: "mobilenetv2",
    name: "MobileNetV2 (Transfer Learning)",
    type: "Transfer Learning (ImageNet)",
    file: "src/models.py -> build_mobilenetv2",
    params:
      "Frozen ImageNet backbone -> GlobalAveragePooling -> Dense(256) -> Dropout(0.5) -> Softmax(6). Two-stage training: head + fine-tune top 30 backbone layers at lr 1e-5.",
    description:
      "ImageNet-pretrained MobileNetV2 backbone, frozen for head training and then fine-tuned across the top 30 backbone layers with a low learning rate (1e-5) for domain adaptation to TrashNet.",
    image: "/figures/mobilenetv2_confusion_matrix.png",
    trainCurves: {
      accuracy: "/figures/mobilenetv2_accuracy.png",
      loss: "/figures/mobilenetv2_loss.png",
    },
  },
  {
    id: "resnet50",
    name: "ResNet50 (Transfer Learning)",
    type: "Transfer Learning (ImageNet)",
    file: "src/models.py -> build_resnet50",
    params:
      "Frozen ImageNet backbone -> GlobalAveragePooling -> Dense(256) -> Dropout(0.5) -> Softmax(6). Two-stage training identical to MobileNetV2.",
    description:
      "ImageNet-pretrained ResNet50 backbone trained with the same two-stage frozen-then-fine-tune recipe as MobileNetV2. Achieved the best test-set F1-score of the three image classifiers.",
    image: "/figures/resnet50_confusion_matrix.png",
    trainCurves: {
      accuracy: "/figures/resnet50_accuracy.png",
      loss: "/figures/resnet50_loss.png",
    },
  },
  {
    id: "rdf-rf",
    name: "RDF Random Forest",
    type: "Tabular Classifier",
    file: "src/models.py -> build_rdf_random_forest + src/rdf_preprocessing.py",
    params:
      "Pipeline: ColumnTransformer (OneHot + StandardScaler) + RandomForestClassifier(300, max_depth=10, min_samples_split=10, min_samples_leaf=4). 5-fold StratifiedKFold GridSearchCV with f1_weighted scoring.",
    description:
      "Tabular Random Forest trained on the synthetic RDF feature dataset (3,000 rows). Receives the material features mapped from the predicted waste class and outputs a binary suitability label with a probability score.",
    image: "/figures/rdf_confusion_matrix.png",
    trainCurves: undefined,
  },
];

export interface ModelSpec {
  id: string;
  name: string;
  type: string;
  file: string;
  params: string;
  description: string;
  image: string;
  trainCurves?: { accuracy: string; loss: string };
}
