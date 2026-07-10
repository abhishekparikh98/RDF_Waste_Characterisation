import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { project } from "@/data/project";

const stack = {
  "Core language": ["Python 3.9+ (training reports 3.11.9)"],
  "Deep learning": ["TensorFlow 2.21.0", "Keras", "MobileNetV2 (ImageNet)", "ResNet50 (ImageNet)"],
  "Classical ML": [
    "scikit-learn",
    "RandomForestClassifier",
    "GridSearchCV",
    "StratifiedKFold",
    "ColumnTransformer",
  ],
  "Web / App": ["Flask 3.x", "Jinja2 templates", "Werkzeug file validation"],
  "Computer vision": ["Pillow", "OpenCV", "scikit-image", "imageio"],
  "Data tooling": ["pandas", "NumPy", "SciPy", "joblib"],
  "Visualisation": ["Matplotlib", "Seaborn", "Plotly"],
  "Documentation / docs": ["Sphinx", "sphinx-rtd-theme"],
  "Experiment tracking (declared)": ["Weights & Biases", "MLflow"],
  "Portfolio stack": [
    "React 18",
    "Vite 5",
    "TypeScript 5",
    "Tailwind CSS 3",
    "Framer Motion 11",
    "React Router 6",
  ],
};

export default function About() {
  return (
    <>
      <PageHeader
        eyebrow="About the project"
        title={project.shortTitle}
        description="A multi-modal machine learning system that classifies waste images and predicts Refuse-Derived Fuel suitability. This page lists the project at a glance."
        meta={
          <div className="flex flex-wrap gap-2">
            <Badge tone="blue">{project.type}</Badge>
            <Badge tone="slate">{project.status}</Badge>
            <Badge tone="emerald">Author: {project.author.name}</Badge>
            <Badge tone="amber">Year: {project.year}</Badge>
          </div>
        }
      />

      <div className="mx-auto w-full max-w-content space-y-6 px-6 py-10">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <Card title="Title (verbatim)">
            <p className="text-ink-700">{project.title}</p>
          </Card>
          <Card title="Scope">
            <p className="text-ink-700">
              Six-class waste image classification (cardboard, glass, metal,
              paper, plastic, trash) plus a binary RDF-suitability decision
              based on mapped material features. Includes a Flask demo and a
              command-line inference tool.
            </p>
          </Card>
          <Card title="Status">
            <p className="text-ink-700">
              Models trained, evaluated and committed. Three image models and
              one Random Forest are persisted under <code>models/</code>.
              Reports and figures under <code>reports/</code> and{" "}
              <code>results/</code>.
            </p>
          </Card>
          <Card title="Repository">
            <p className="text-ink-700">
              Branch <code className="font-mono">{project.repository.branch}</code>.
              The repository is a single linear Git history with five commits.
            </p>
          </Card>
        </div>

        <Card title="Technology stack">
          <p className="text-ink-700">
            Every dependency is declared in{" "}
            <code>requirements.txt</code> and{" "}
            <code>pyproject.toml</code>. The list below is taken from those
            files plus the actual imports used in the codebase.
          </p>
          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            {Object.entries(stack).map(([group, items]) => (
              <div
                key={group}
                className="rounded-xl border border-ink-200 bg-ink-50/60 p-4"
              >
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                  {group}
                </p>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {items.map((item) => (
                    <Badge key={item} tone="slate">
                      {item}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </>
  );
}
