import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { commits, phaseTimeline } from "@/data/timeline";

export default function Timeline() {
  return (
    <>
      <PageHeader
        eyebrow="Development timeline"
        title="Real Git history and project phases"
        description="Five commits on the master branch, grouped into four phases. Data is taken from `git log`."
      />

      <div className="mx-auto w-full max-w-content space-y-10 px-6 py-10">
        <section>
          <h2 className="font-sans-ui text-2xl font-bold text-ink-900">
            Phases
          </h2>
          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
            {phaseTimeline.map((p) => (
              <Card
                key={p.phase}
                title={`${p.phase} — ${p.title}`}
                subtitle={`${p.period} · commits: ${p.commits.join(", ")}`}
              >
                <p className="text-ink-700">{p.summary}</p>
              </Card>
            ))}
          </div>
        </section>

        <section>
          <h2 className="font-sans-ui text-2xl font-bold text-ink-900">
            Commit history
          </h2>
          <ol className="relative mt-6 space-y-6 border-l border-ink-200 pl-6">
            {commits.map((c) => (
              <li key={c.hash} className="relative">
                <span className="absolute -left-[33px] top-1 grid h-5 w-5 place-items-center rounded-full border-4 border-white bg-accent-700" />
                <Card
                  title={c.subject}
                  subtitle={`${c.date} · ${c.shortHash}`}
                >
                  <p className="text-ink-700">{c.description}</p>
                  <p className="mt-3 text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
                    Touched
                  </p>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {c.scope.map((s) => (
                      <Badge key={s} tone="slate">
                        {s}
                      </Badge>
                    ))}
                  </div>
                  <p className="mt-3 break-all font-mono text-[11px] text-ink-500">
                    {c.hash}
                  </p>
                </Card>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </>
  );
}
