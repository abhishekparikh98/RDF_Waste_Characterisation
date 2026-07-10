import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Figure } from "@/components/ui/Figure";
import { trashnet, taco, rdfDataset } from "@/data/datasets";

export default function Datasets() {
  const totalRaw = trashnet.classes.reduce((s, c) => s + c.rawCount, 0);
  const totalTrain = trashnet.classes.reduce((s, c) => s + c.trainCount, 0);
  const totalVal = trashnet.classes.reduce((s, c) => s + c.valCount, 0);
  const totalTest = trashnet.classes.reduce((s, c) => s + c.testCount, 0);

  return (
    <>
      <PageHeader
        eyebrow="Datasets"
        title="TrashNet, TACO, and the RDF tabular dataset"
        description="Concrete numbers and splits, taken from the project reports and preprocessing logs."
      />

      <div className="mx-auto w-full max-w-content space-y-8 px-6 py-10">
        <section>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Figure
              src="/figures/trashnet_class_distribution.png"
              alt="TrashNet class distribution"
              caption="TrashNet class distribution (reports/figures/trashnet_class_distribution.png)"
            />
            <Figure
              src="/figures/split_distribution.png"
              alt="Train / validation / test class distribution after preprocessing"
              caption="Stratified split distribution (reports/figures/split_distribution.png)"
            />
          </div>
          <div className="mt-6">
            <Figure
              src="/figures/class_distribution_comparison.png"
              alt="Class distribution comparison across train, validation and test"
              caption="Class distribution comparison (reports/figures/class_distribution_comparison.png)"
            />
          </div>
        </section>

        <Card
          title="TrashNet (raw + processed)"
          subtitle={`Path: ${trashnet.path}`}
        >
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                Raw dataset
              </p>
              <ul className="mt-2 space-y-1 text-sm text-ink-700">
                <li>
                  Total images:{" "}
                  <strong>{totalRaw.toLocaleString()}</strong>
                </li>
                <li>
                  Format: <strong>{trashnet.format}</strong>
                </li>
                <li>
                  Resolution: <strong>{trashnet.resolution}</strong>
                </li>
                <li>
                  Corrupted:{" "}
                  <strong className="text-emerald-700">{trashnet.corrupted}</strong>
                </li>
              </ul>
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                Processed split
              </p>
              <ul className="mt-2 space-y-1 text-sm text-ink-700">
                <li>
                  Train: <strong>{totalTrain.toLocaleString()}</strong> (
                  {(trashnet.splitRatios.train * 100).toFixed(0)}%)
                </li>
                <li>
                  Validation: <strong>{totalVal.toLocaleString()}</strong> (
                  {(trashnet.splitRatios.validation * 100).toFixed(0)}%)
                </li>
                <li>
                  Test: <strong>{totalTest.toLocaleString()}</strong> (
                  {(trashnet.splitRatios.test * 100).toFixed(0)}%)
                </li>
                <li>
                  Random seed: <code>{trashnet.randomSeed}</code>
                </li>
                <li>
                  Imbalance ratio:{" "}
                  <strong>{trashnet.imbalanceRatio}</strong>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-5 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-ink-50 text-left">
                  <th className="px-3 py-2">Class</th>
                  <th className="px-3 py-2">Raw</th>
                  <th className="px-3 py-2">Train</th>
                  <th className="px-3 py-2">Validation</th>
                  <th className="px-3 py-2">Test</th>
                </tr>
              </thead>
              <tbody>
                {trashnet.classes.map((c) => (
                  <tr key={c.name} className="border-t border-ink-100">
                    <td className="px-3 py-2 font-medium">{c.name}</td>
                    <td className="px-3 py-2">{c.rawCount}</td>
                    <td className="px-3 py-2">{c.trainCount}</td>
                    <td className="px-3 py-2">{c.valCount}</td>
                    <td className="px-3 py-2">{c.testCount}</td>
                  </tr>
                ))}
                <tr className="border-t-2 border-ink-200 bg-ink-50 font-semibold">
                  <td className="px-3 py-2">Total</td>
                  <td className="px-3 py-2">{totalRaw}</td>
                  <td className="px-3 py-2">{totalTrain}</td>
                  <td className="px-3 py-2">{totalVal}</td>
                  <td className="px-3 py-2">{totalTest}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-sm">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                Resolution
              </p>
              <p className="mt-1 font-mono">{trashnet.imageProperties.resolution}</p>
            </div>
            <div className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-sm">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                Channels / dtype / range
              </p>
              <p className="mt-1">
                {trashnet.imageProperties.channels} &middot;{" "}
                {trashnet.imageProperties.dtype} &middot;{" "}
                {trashnet.imageProperties.range}
              </p>
            </div>
            <div className="rounded-xl border border-ink-200 bg-ink-50/60 p-3 text-sm">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                Interpolation / storage
              </p>
              <p className="mt-1">
                {trashnet.imageProperties.interpolation} &middot;{" "}
                {trashnet.imageProperties.storage}
              </p>
            </div>
          </div>
        </Card>

        <Card title="TACO (Trash Annotations in Context)" subtitle={taco.path}>
          <Badge tone="amber">{taco.status}</Badge>
          <p className="mt-3 text-ink-700">
            The TACO library is bundled in the project but the actual TACO
            images are not used in the current training pipeline. The
            components present in the repository are:
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-ink-700">
            {taco.components.map((c) => (
              <li key={c}>
                <code>{c}</code>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Synthetic RDF tabular dataset" subtitle={rdfDataset.path}>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <ul className="space-y-1 text-sm text-ink-700">
              <li>
                Rows: <strong>{rdfDataset.rows.toLocaleString()}</strong>
              </li>
              <li>
                Class balance:{" "}
                <code>
                  Not Suitable: {rdfDataset.classBalance.notSuitable} /
                  Suitable: {rdfDataset.classBalance.suitable}
                </code>
              </li>
              <li>
                Target: <code>{rdfDataset.target}</code>
              </li>
              <li>
                Grade: <code>{rdfDataset.grade}</code>
              </li>
            </ul>
            <ul className="space-y-1 text-sm text-ink-700">
              <li>
                Materials:{" "}
                {rdfDataset.materials.map((m) => (
                  <Badge key={m} tone="slate" className="mr-1">
                    {m}
                  </Badge>
                ))}
              </li>
              <li>
                Source: <em>{rdfDataset.source}</em>
              </li>
              <li>
                <Badge tone="amber">Limitation</Badge> {rdfDataset.limitation}
              </li>
            </ul>
          </div>

          <p className="mt-4 text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
            Feature columns
          </p>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {rdfDataset.featureColumns.map((f) => (
              <Badge key={f} tone="blue">
                {f}
              </Badge>
            ))}
          </div>
        </Card>
      </div>
    </>
  );
}
