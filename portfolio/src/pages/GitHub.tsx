import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { project } from "@/data/project";

const files = [
  "app.py",
  "PROJECT_EXECUTION_GUIDE.txt",
  "README.md",
  "requirements.txt",
  "pyproject.toml",
  ".env.example",
  ".gitignore",
  "LICENSE",
  "src/__init__.py",
  "src/config.py",
  "src/evaluation.py",
  "src/models.py",
  "src/multimodal_inference.py",
  "src/preprocessing.py",
  "src/rdf_preprocessing.py",
  "src/training.py",
  "src/utils.py",
  "scripts/compare_cnn_mobilenetv2.py",
  "scripts/explore_dataset.py",
  "scripts/preprocess_dataset.py",
  "scripts/run_multimodal_inference.py",
  "scripts/train_cnn.py",
  "scripts/train_rdf_rf.py",
  "templates/index.html",
  "static/style.css",
  "docs/EXPLORATION_SCRIPT.md",
  "docs/FLASK_WEB_APP_ARCHITECTURE.md",
  "docs/MULTIMODAL_INFERENCE_ARCHITECTURE.md",
  "docs/PREPROCESSING_PIPELINE.md",
  "docs/PREPROCESSING_QUICK_REFERENCE.md",
  "docs/PREPROCESSING_SUMMARY.md",
  "docs/README.md",
];

const reproduction = [
  "git clone <repository-url>",
  "cd msc-project",
  "python -m venv .venv",
  ".venv\\Scripts\\activate (Windows) or source .venv/bin/activate (Unix)",
  "pip install -r requirements.txt",
  "python scripts/preprocess_dataset.py   # only if data/processed/ is empty",
  "python scripts/train_cnn.py",
  "python scripts/compare_cnn_mobilenetv2.py",
  "python scripts/train_rdf_rf.py",
  "python scripts/run_multimodal_inference.py --image path/to/image.jpg",
  "python app.py   # Flask demo on http://127.0.0.1:5000",
];

export default function GitHub() {
  return (
    <>
      <PageHeader
        eyebrow="GitHub repository"
        title={project.repository.url.replace(/^https?:\/\//, "")}
        description={`Branch: ${project.repository.branch} · Author: ${project.author.name}`}
        meta={
          <div className="flex flex-wrap gap-2">
            <Badge tone="blue">Python 3.9+</Badge>
            <Badge tone="blue">TensorFlow 2.21</Badge>
            <Badge tone="blue">scikit-learn</Badge>
            <Badge tone="blue">Flask 3.x</Badge>
            <Badge tone="emerald">MIT License</Badge>
          </div>
        }
      />

      <div className="mx-auto w-full max-w-content space-y-8 px-6 py-10">
        <Card title="Top-level files">
          <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {files.map((f) => (
              <li
                key={f}
                className="flex items-center justify-between rounded-md border border-ink-100 bg-ink-50/60 px-3 py-1.5 font-mono text-xs"
              >
                <span className="truncate text-ink-700">{f}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="How to reproduce">
          <ol className="list-decimal space-y-1 pl-5 font-mono text-sm text-ink-700">
            {reproduction.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <p className="mt-3 text-xs text-ink-500">
            These steps mirror{" "}
            <code>PROJECT_EXECUTION_GUIDE.txt</code> in the project root.
          </p>
        </Card>

        <Card title="Generated artefacts (committed to the repo)">
          <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {[
              "reports/cnn_baseline_report.md",
              "reports/cnn_mobilenetv2_resnet50_evaluation_report.md",
              "reports/dataset_report.md",
              "reports/preprocessing_report.md",
              "reports/rdf_random_forest_report.md",
              "reports/figures/*.png (3 files)",
              "results/*.png (15 files)",
              "results/*.txt (4 classification reports)",
            ].map((f) => (
              <li
                key={f}
                className="rounded-md border border-ink-100 bg-ink-50/60 px-3 py-1.5 font-mono text-xs text-ink-700"
              >
                {f}
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Trained model files">
          <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {[
              "models/cnn_baseline_best.h5 (309 MB)",
              "models/mobilenetv2_best.h5 (25 MB)",
              "models/resnet50_best.h5 (216 MB)",
              "models/rdf_random_forest_pipeline.joblib (2.8 MB)",
            ].map((f) => (
              <li
                key={f}
                className="rounded-md border border-ink-100 bg-ink-50/60 px-3 py-1.5 font-mono text-xs text-ink-700"
              >
                {f}
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </>
  );
}
