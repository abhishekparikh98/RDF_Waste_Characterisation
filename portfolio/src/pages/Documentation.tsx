import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

const documents = [
  {
    id: "preprocessing-pipeline",
    name: "PREPROCESSING_PIPELINE.md",
    description:
      "Detailed architecture and implementation of the image preprocessing pipeline (615 lines of code, 5 modular classes).",
    topic: "Preprocessing",
  },
  {
    id: "preprocessing-summary",
    name: "PREPROCESSING_SUMMARY.md",
    description:
      "Module-by-module explanation of the preprocessing pipeline: ImageValidator, ImagePreprocessor, DatasetSplitter, DatasetSaver, PreprocessingPipeline.",
    topic: "Preprocessing",
  },
  {
    id: "preprocessing-quick-reference",
    name: "PREPROCESSING_QUICK_REFERENCE.md",
    description:
      "Quick reference cheatsheet for running the preprocessing pipeline and using the preprocessed dataset.",
    topic: "Preprocessing",
  },
  {
    id: "exploration-script",
    name: "EXPLORATION_SCRIPT.md",
    description:
      "Documentation of scripts/explore_dataset.py with DatasetDetector, ImageAnalyzer, DatasetExplorer, VisualizationGenerator, and ReportGenerator.",
    topic: "Exploration",
  },
  {
    id: "flask-web-app",
    name: "FLASK_WEB_APP_ARCHITECTURE.md",
    description:
      "Flask demo design notes: request flow, components, model dependencies and design limitations.",
    topic: "Deployment",
  },
  {
    id: "multimodal-inference",
    name: "MULTIMODAL_INFERENCE_ARCHITECTURE.md",
    description:
      "Multimodal inference pipeline data flow: Image -> Waste classification -> Material features -> RDF suitability.",
    topic: "Inference",
  },
  {
    id: "docs-readme",
    name: "docs/README.md",
    description: "Index of the docs/ directory.",
    topic: "Index",
  },
  {
    id: "readme",
    name: "README.md",
    description:
      "Top-level project README with research objectives, structure, install, and Flask usage instructions.",
    topic: "Project",
  },
  {
    id: "project-execution-guide",
    name: "PROJECT_EXECUTION_GUIDE.txt",
    description:
      "Operator guide: input folders, step-by-step execution, output locations, common issues.",
    topic: "Project",
  },
];

export default function Documentation() {
  return (
    <>
      <PageHeader
        eyebrow="Documentation"
        title="All docs shipped with the project"
        description="Every document listed here exists under docs/, README.md or PROJECT_EXECUTION_GUIDE.txt in the actual project repository."
      />

      <div className="mx-auto w-full max-w-content px-6 py-10">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {documents.map((d) => (
            <Card key={d.id} title={d.name} subtitle={`docs/${d.name}`}>
              <div className="mb-2">
                <Badge tone="blue">{d.topic}</Badge>
              </div>
              <p className="text-ink-700">{d.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </>
  );
}
