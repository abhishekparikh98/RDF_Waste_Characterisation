import { useState } from "react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Markdown } from "@/components/ui/Markdown";
import { reports } from "@/data/reports";

export default function Reports() {
  const [active, setActive] = useState(reports[0].id);
  const current = reports.find((r) => r.id === active) ?? reports[0];

  return (
    <>
      <PageHeader
        eyebrow="Reports"
        title="Actual markdown reports from the project"
        description="The reports below are imported verbatim from reports/*.md in the project and rendered as Markdown."
      />

      <div className="mx-auto w-full max-w-content px-6 py-10">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
          <aside className="lg:col-span-1">
            <ul className="space-y-2">
              {reports.map((r) => (
                <li key={r.id}>
                  <button
                    type="button"
                    onClick={() => setActive(r.id)}
                    className={`w-full rounded-xl border p-3 text-left transition ${
                      active === r.id
                        ? "border-accent-300 bg-accent-50"
                        : "border-ink-200 bg-white hover:border-accent-200"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <p className="font-sans-ui text-sm font-semibold text-ink-900">
                        {r.title}
                      </p>
                    </div>
                    <p className="mt-1 text-xs text-ink-500">
                      {r.source}
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1">
                      <Badge tone="slate">{r.category}</Badge>
                      <Badge tone="emerald">{r.generated}</Badge>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </aside>

          <div className="lg:col-span-3">
            <Card
              title={current.title}
              subtitle={`${current.source} · generated ${current.generated}`}
            >
              <Markdown source={current.markdown} />
            </Card>
          </div>
        </div>
      </div>
    </>
  );
}
