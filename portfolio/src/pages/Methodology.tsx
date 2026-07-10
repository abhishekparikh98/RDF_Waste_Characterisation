import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { models } from "@/data/architecture";

export default function Methodology() {
  return (
    <>
      <PageHeader
        eyebrow="Methodology"
        title="How the system is built and trained"
        description="Every model and step below is implemented in the actual project source code."
      />

      <div className="mx-auto w-full max-w-content space-y-6 px-6 py-10">
        <Card title="Stages">
          <ol className="list-decimal space-y-2 pl-5 text-ink-700">
            <li>
              <strong>Data exploration</strong> &mdash; detect TrashNet and
              TACO, count classes, validate images, log statistics.
            </li>
            <li>
              <strong>Preprocessing</strong> &mdash; validate, resize to 224 x
              224, normalise to [0, 1], split 70 / 15 / 15 with seed 42.
            </li>
            <li>
              <strong>Image model training</strong> &mdash; Baseline CNN, then
              two-stage transfer learning (frozen backbone + fine-tune top 30
              layers at lr 1e-5) for MobileNetV2 and ResNet50.
            </li>
            <li>
              <strong>Tabular model training</strong> &mdash; GridSearchCV over a
              Pipeline of (ColumnTransformer + RandomForestClassifier) using
              StratifiedKFold (5-fold) and f1_weighted scoring.
            </li>
            <li>
              <strong>Evaluation</strong> &mdash; weighted precision / recall /
              F1, confusion matrices and training curves for each model.
            </li>
            <li>
              <strong>Multimodal inference</strong> &mdash; chain
              image classifier, material feature mapping and RDF Random Forest.
            </li>
            <li>
              <strong>Deployment</strong> &mdash; Flask upload form (app.py)
              and CLI runner (scripts/run_multimodal_inference.py).
            </li>
          </ol>
        </Card>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {models.map((m) => (
            <Card
              key={m.id}
              title={m.name}
              subtitle={m.type}
            >
              <p className="text-ink-700">{m.description}</p>
              <p className="mt-2 font-mono text-xs text-ink-500">{m.file}</p>
              <p className="mt-2 text-sm text-ink-700">
                <span className="font-semibold">Configuration: </span>
                {m.params}
              </p>
            </Card>
          ))}
        </div>
      </div>
    </>
  );
}
