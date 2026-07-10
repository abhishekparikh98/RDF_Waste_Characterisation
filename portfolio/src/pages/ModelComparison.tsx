import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Figure } from "@/components/ui/Figure";
import {
  imageModelResults,
  rdfModelResult,
  imageModelDeltas,
  perClassResnet50,
} from "@/data/results";
import { trashnet } from "@/data/datasets";

export default function ModelComparison() {
  return (
    <>
      <PageHeader
        eyebrow="Model comparison"
        title="Baseline CNN vs MobileNetV2 vs ResNet50"
        description="Test-set metrics for all three image models on the same stratified TrashNet split."
        meta={
          <div className="flex flex-wrap gap-2">
            {imageModelResults.map((m) => (
              <Badge
                key={m.id}
                tone={m.id === "resnet50" ? "emerald" : "slate"}
              >
                {m.name}: F1 {(m.f1 * 100).toFixed(2)}%
              </Badge>
            ))}
          </div>
        }
      />

      <div className="mx-auto w-full max-w-content space-y-8 px-6 py-10">
        <Figure
          src="/figures/comparison.png"
          alt="Accuracy, precision, recall and F1 bar chart for the three image models"
          caption="results/cnn_mobilenetv2_resnet50_comparison.png"
        />

        <Card title="Test-set metrics (weighted)">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-ink-50 text-left">
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2 text-right">Accuracy</th>
                  <th className="px-3 py-2 text-right">Precision</th>
                  <th className="px-3 py-2 text-right">Recall</th>
                  <th className="px-3 py-2 text-right">F1-score</th>
                </tr>
              </thead>
              <tbody>
                {imageModelResults.map((m) => (
                  <tr
                    key={m.id}
                    className={`border-t border-ink-100 ${
                      m.id === "resnet50" ? "bg-emerald-50/40" : ""
                    }`}
                  >
                    <td className="px-3 py-2 font-medium">{m.name}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {m.accuracy.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {m.precision.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {m.recall.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {m.f1.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <Card title="Δ vs Baseline (accuracy)">
            <p className="text-ink-700">
              <strong>MobileNetV2:</strong>{" "}
              <code>
                +
                {(imageModelDeltas.mobilenetv2VsBaseline.accuracy * 100).toFixed(
                  2,
                )}{" "}
                pp
              </code>
            </p>
            <p className="text-ink-700">
              <strong>ResNet50:</strong>{" "}
              <code>
                +
                {(imageModelDeltas.resnet50VsBaseline.accuracy * 100).toFixed(
                  2,
                )}{" "}
                pp
              </code>
            </p>
          </Card>
          <Card title="Δ vs Baseline (F1)">
            <p className="text-ink-700">
              <strong>MobileNetV2:</strong>{" "}
              <code>
                +
                {(imageModelDeltas.mobilenetv2VsBaseline.f1 * 100).toFixed(2)}{" "}
                pp
              </code>
            </p>
            <p className="text-ink-700">
              <strong>ResNet50:</strong>{" "}
              <code>
                +
                {(imageModelDeltas.resnet50VsBaseline.f1 * 100).toFixed(2)} pp
              </code>
            </p>
          </Card>
        </div>

        <Card title="Per-class metrics for ResNet50 (best model)">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-ink-50 text-left">
                  <th className="px-3 py-2">Class</th>
                  <th className="px-3 py-2 text-right">Precision</th>
                  <th className="px-3 py-2 text-right">Recall</th>
                  <th className="px-3 py-2 text-right">F1</th>
                  <th className="px-3 py-2 text-right">Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(perClassResnet50).map(([cls, m]) => (
                  <tr key={cls} className="border-t border-ink-100">
                    <td className="px-3 py-2 font-medium">{cls}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {m.precision.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {m.recall.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {m.f1.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right">{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-ink-500">
            Support values come from the test split (
            {trashnet.testTotal} images). The minority <code>trash</code> class
            is the weakest with 21 samples.
          </p>
        </Card>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <Figure
            src="/figures/baseline_confusion_matrix.png"
            alt="Baseline CNN confusion matrix"
            caption="Baseline CNN confusion matrix"
          />
          <Figure
            src="/figures/mobilenetv2_confusion_matrix.png"
            alt="MobileNetV2 confusion matrix"
            caption="MobileNetV2 confusion matrix"
          />
          <Figure
            src="/figures/resnet50_confusion_matrix.png"
            alt="ResNet50 confusion matrix"
            caption="ResNet50 confusion matrix"
          />
        </div>

        <Card title="RDF Random Forest" subtitle="src/models.py + src/rdf_preprocessing.py">
          <p className="text-ink-700">
            The Random Forest operates on the mapped material features
            produced from the predicted waste class. Test metrics on 600 held-out
            rows of the synthetic RDF dataset:
          </p>
          <div className="mt-3 grid grid-cols-2 gap-4 md:grid-cols-4">
            <div className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-center">
              <p className="text-xs text-ink-500">Accuracy</p>
              <p className="mt-1 font-sans-ui text-xl font-bold">
                {(rdfModelResult.accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-center">
              <p className="text-xs text-ink-500">Precision</p>
              <p className="mt-1 font-sans-ui text-xl font-bold">
                {(rdfModelResult.precision * 100).toFixed(2)}%
              </p>
            </div>
            <div className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-center">
              <p className="text-xs text-ink-500">Recall</p>
              <p className="mt-1 font-sans-ui text-xl font-bold">
                {(rdfModelResult.recall * 100).toFixed(2)}%
              </p>
            </div>
            <div className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-center">
              <p className="text-xs text-ink-500">F1</p>
              <p className="mt-1 font-sans-ui text-xl font-bold">
                {(rdfModelResult.f1 * 100).toFixed(2)}%
              </p>
            </div>
          </div>
          <p className="mt-3 text-sm text-ink-700">
            <em>{rdfModelResult.notes}</em>
          </p>
        </Card>
      </div>
    </>
  );
}
