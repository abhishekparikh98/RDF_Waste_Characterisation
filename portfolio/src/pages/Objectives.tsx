import { PageHeader } from "@/components/layout/PageHeader";
import { researchObjectives } from "@/data/project";

export default function Objectives() {
  return (
    <>
      <PageHeader
        eyebrow="Research objectives"
        title="What this dissertation set out to do"
        description="Verbatim objectives as listed in the project README."
      />

      <div className="mx-auto w-full max-w-content px-6 py-10">
        <ol className="space-y-4">
          {researchObjectives.map((objective, index) => (
            <li
              key={index}
              className="flex gap-4 rounded-2xl border border-ink-200 bg-white p-6 shadow-card"
            >
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-accent-700 font-sans-ui text-base font-bold text-white">
                {index + 1}
              </span>
              <p className="text-base leading-7 text-ink-700">{objective}</p>
            </li>
          ))}
        </ol>
      </div>
    </>
  );
}
