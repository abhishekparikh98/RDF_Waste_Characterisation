import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Metric } from "@/components/ui/Metric";
import { Badge } from "@/components/ui/Badge";
import { Figure } from "@/components/ui/Figure";
import { Section } from "@/components/ui/Section";
import { project, projectSummary } from "@/data/project";
import {
  imageModelResults,
  rdfModelResult,
  imageModelDeltas,
} from "@/data/results";
import { trashnet } from "@/data/datasets";

const best = imageModelResults.find((m) => m.id === "resnet50")!;

export default function Home() {
  return (
    <>
      <section className="relative overflow-hidden border-b border-ink-200 bg-white">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(60%_50%_at_50%_0%,rgba(29,78,216,0.08),transparent_70%)]" />
        <div className="mx-auto w-full max-w-content px-6 py-16 sm:py-24">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-3xl"
          >
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent-700">
              {project.type} &middot; {project.year}
            </p>
            <h1 className="mt-3 font-sans-ui text-4xl font-bold tracking-tight text-ink-900 sm:text-5xl">
              {project.title}
            </h1>
            <p className="mt-5 text-lg leading-8 text-ink-600">
              {projectSummary.goal}
            </p>
            <p className="mt-3 text-base leading-7 text-ink-600">
              The full pipeline is:{" "}
              <code className="rounded bg-ink-100 px-1.5 py-0.5 font-mono text-sm">
                {projectSummary.pipeline}
              </code>
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                to="/model-comparison"
                className="inline-flex items-center rounded-lg bg-accent-700 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-800"
              >
                View Model Comparison
              </Link>
              <Link
                to="/architecture"
                className="inline-flex items-center rounded-lg border border-ink-300 bg-white px-4 py-2 text-sm font-semibold text-ink-800 transition hover:border-accent-300 hover:text-accent-800"
              >
                Explore Architecture
              </Link>
              <Link
                to="/timeline"
                className="inline-flex items-center rounded-lg border border-ink-300 bg-white px-4 py-2 text-sm font-semibold text-ink-800 transition hover:border-accent-300 hover:text-accent-800"
              >
                Development Timeline
              </Link>
            </div>
          </motion.div>

          <div className="mt-12 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric
              label="Best Image Model"
              value={best.name}
              hint={`Test accuracy ${(best.accuracy * 100).toFixed(2)}%, F1 ${(
                best.f1 * 100
              ).toFixed(2)}%`}
              tone="highlight"
            />
            <Metric
              label="RDF Suitability"
              value={`${(rdfModelResult.accuracy * 100).toFixed(2)}%`}
              hint={`F1 ${(rdfModelResult.f1 * 100).toFixed(2)}% on 600 test rows`}
            />
            <Metric
              label="Δ ResNet50 vs Baseline"
              value={`+${(imageModelDeltas.resnet50VsBaseline.accuracy * 100).toFixed(2)} pp`}
              hint="Weighted-accuracy gain over from-scratch CNN"
            />
            <Metric
              label="TrashNet Images"
              value={trashnet.totalRawImages.toLocaleString()}
              hint={`6 classes, 224 x 224 RGB after preprocessing`}
            />
          </div>
        </div>
      </section>

      <Section
        eyebrow="Project at a glance"
        title="A real, runnable multi-modal pipeline"
        description="Every figure, report and metric on this site is sourced from the actual MSc project. The Flask demo and the multimodal CLI both load the saved Keras and scikit-learn artefacts."
      >
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <Figure
              src="/figures/comparison.png"
              alt="Comparison of Baseline CNN, MobileNetV2 and ResNet50 across accuracy, precision, recall and F1-score"
              caption="Test-set comparison of Baseline CNN, MobileNetV2 and ResNet50 (source: results/cnn_mobilenetv2_resnet50_comparison.png)"
            />
          </div>
          <div className="space-y-4">
            <div className="rounded-2xl border border-ink-200 bg-white p-5 shadow-card">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                Pipeline
              </p>
              <p className="mt-2 font-sans-ui text-base font-semibold text-ink-900">
                Image &rarr; Class &rarr; Material features &rarr; RDF
              </p>
              <p className="mt-2 text-sm text-ink-600">
                The image classifier predicts one of six waste categories. A
                deterministic feature map then produces a single-row tabular
                vector that the Random Forest scores for RDF suitability.
              </p>
            </div>
            <div className="rounded-2xl border border-ink-200 bg-white p-5 shadow-card">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                Stack
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {[
                  "Python 3.11",
                  "TensorFlow 2.21",
                  "Keras",
                  "scikit-learn",
                  "Flask",
                  "Pandas",
                  "NumPy",
                  "Pillow",
                  "React",
                  "Vite",
                ].map((t) => (
                  <Badge key={t} tone="blue">
                    {t}
                  </Badge>
                ))}
              </div>
            </div>
            <div className="rounded-2xl border border-emerald-200 bg-emerald-50/60 p-5 shadow-card">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">
                Status
              </p>
              <p className="mt-2 font-sans-ui text-base font-semibold text-ink-900">
                {project.status}
              </p>
              <p className="mt-1 text-sm text-ink-600">
                All training runs, evaluation artefacts and inference components
                are in the repository.
              </p>
            </div>
          </div>
        </div>
      </Section>
    </>
  );
}
