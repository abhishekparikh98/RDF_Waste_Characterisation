import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Figure } from "@/components/ui/Figure";
import { Section } from "@/components/ui/Section";
import { models } from "@/data/architecture";
import { rdfClassBreakdown } from "@/data/results";

export default function Results() {
  return (
    <>
      <PageHeader
        eyebrow="Results"
        title="All figures produced by the project"
        description="Training curves, confusion matrices and feature importance from results/ and reports/figures/."
      />

      <div className="mx-auto w-full max-w-content space-y-10 px-6 py-10">
        <Section title="Image classifiers" eyebrow="Image models">
          <div className="space-y-8">
            {models
              .filter((m) => m.id !== "rdf-rf")
              .map((m) => (
                <div key={m.id} className="space-y-3">
                  <h3 className="font-sans-ui text-xl font-bold text-ink-900">
                    {m.name}
                  </h3>
                  <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
                    <Figure
                      src={m.image}
                      alt={`${m.name} confusion matrix`}
                      caption={`${m.name} confusion matrix`}
                    />
                    {m.trainCurves ? (
                      <>
                        <Figure
                          src={m.trainCurves.accuracy}
                          alt={`${m.name} accuracy curves`}
                          caption={`${m.name} training and validation accuracy`}
                        />
                        <Figure
                          src={m.trainCurves.loss}
                          alt={`${m.name} loss curves`}
                          caption={`${m.name} training and validation loss`}
                        />
                      </>
                    ) : null}
                  </div>
                </div>
              ))}
          </div>
        </Section>

        <Section title="RDF Random Forest" eyebrow="Tabular model">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Figure
              src="/figures/rdf_confusion_matrix.png"
              alt="RDF Random Forest confusion matrix"
              caption="RDF Random Forest confusion matrix"
            />
            <Figure
              src="/figures/rdf_feature_importance.png"
              alt="RDF Random Forest feature importance"
              caption="RDF Random Forest feature importance"
            />
          </div>

          <Card
            title="RDF class breakdown"
            subtitle="From results/rdf_classification_report.txt"
            className="mt-6"
          >
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
                  <tr className="border-t border-ink-100">
                    <td className="px-3 py-2 font-medium">Not Suitable</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {rdfClassBreakdown.notSuitable.precision.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {rdfClassBreakdown.notSuitable.recall.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {rdfClassBreakdown.notSuitable.f1.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {rdfClassBreakdown.notSuitable.support}
                    </td>
                  </tr>
                  <tr className="border-t border-ink-100">
                    <td className="px-3 py-2 font-medium">Suitable</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {rdfClassBreakdown.suitable.precision.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {rdfClassBreakdown.suitable.recall.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {rdfClassBreakdown.suitable.f1.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {rdfClassBreakdown.suitable.support}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        </Section>
      </div>
    </>
  );
}
