import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import {
  preprocessing,
  rdfPreprocessing,
  multimodalInference,
  flaskApp,
} from "@/data/architecture";

export default function Architecture() {
  return (
    <>
      <PageHeader
        eyebrow="Project architecture"
        title="Module map and data flow"
        description="Each layer below maps 1-to-1 to a file or module in the real project."
      />

      <div className="mx-auto w-full max-w-content space-y-8 px-6 py-10">
        <Card title="High-level data flow">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
            {[
              "Image (TrashNet)",
              "Image classifier (CNN / MobileNetV2 / ResNet50)",
              "Predicted waste class",
              "Material features (Random Forest input)",
              "RDF Suitable / Not Suitable + probability",
            ].map((step, i) => (
              <div
                key={i}
                className="rounded-xl border border-accent-200 bg-accent-50/50 p-3 text-center text-sm font-semibold text-accent-900"
              >
                <Badge tone="blue">Step {i + 1}</Badge>
                <p className="mt-2">{step}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Image preprocessing pipeline" subtitle={preprocessing.location}>
          <p className="text-ink-700">{preprocessing.title}. Steps:</p>
          <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-ink-700">
            {preprocessing.steps.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ol>
          <p className="mt-4 text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
            Classes
          </p>
          <ul className="mt-2 grid grid-cols-1 gap-3 md:grid-cols-2">
            {preprocessing.classes.map((c) => (
              <li
                key={c.name}
                className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-sm"
              >
                <p className="font-mono text-ink-900">{c.name}</p>
                <p className="mt-1 text-ink-600">{c.purpose}</p>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="RDF tabular preprocessing" subtitle={rdfPreprocessing.location}>
          <p className="text-ink-700">{rdfPreprocessing.title}.</p>
          <p className="mt-3 text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
            Pipeline
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-ink-700">
            {rdfPreprocessing.pipeline.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
          <p className="mt-4 text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
            Features
          </p>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {rdfPreprocessing.features.map((f) => (
              <Badge key={f} tone="slate">
                {f}
              </Badge>
            ))}
          </div>
        </Card>

        <Card
          title="Multimodal inference pipeline"
          subtitle={multimodalInference.location}
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {multimodalInference.flow.map((step) => (
              <div
                key={step.step}
                className="rounded-xl border border-ink-200 bg-white p-4"
              >
                <Badge tone="blue">{step.step}</Badge>
                <p className="mt-2 text-sm text-ink-700">{step.purpose}</p>
                <p className="mt-2 font-mono text-xs text-ink-500">
                  {step.file}
                </p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm text-ink-700">
            CLI entrypoint: <code>{multimodalInference.cli.file}</code> &mdash;{" "}
            <code>{multimodalInference.cli.usage}</code>
          </p>
        </Card>

        <Card title="Flask demo application" subtitle={flaskApp.location}>
          <p className="text-ink-700">{flaskApp.title}.</p>
          <p className="mt-2 text-sm text-ink-700">
            <Badge tone="blue">Route</Badge>{" "}
            <code>
              {flaskApp.method} {flaskApp.route}
            </code>
          </p>
          <p className="mt-2 text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
            Validation
          </p>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-ink-700">
            {flaskApp.validation.map((v) => (
              <li key={v}>{v}</li>
            ))}
          </ul>
          <p className="mt-4 text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
            Request flow
          </p>
          <ol className="mt-1 list-decimal space-y-1 pl-5 text-sm text-ink-700">
            {flaskApp.flow.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ol>
          <p className="mt-3 text-sm text-ink-700">
            <Badge tone="emerald">Caching</Badge> {flaskApp.pipelineCache}
          </p>
        </Card>
      </div>
    </>
  );
}
