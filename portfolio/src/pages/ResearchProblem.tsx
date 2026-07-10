import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { researchProblem } from "@/data/project";

export default function ResearchProblem() {
  return (
    <>
      <PageHeader
        eyebrow="Research problem"
        title="Why RDF, and why multi-modal?"
        description="The motivation, gap and approach for combining computer vision and tabular modelling on the TrashNet dataset."
      />

      <div className="mx-auto w-full max-w-content space-y-6 px-6 py-10">
        <Card title="Context">{researchProblem.context}</Card>
        <Card title="Identified gap">{researchProblem.gap}</Card>
        <Card title="Approach">{researchProblem.approach}</Card>
        <Card title="Research questions">
          <ol className="list-decimal space-y-2 pl-5">
            {researchProblem.questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ol>
        </Card>
      </div>
    </>
  );
}
