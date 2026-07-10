import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { literatureReview } from "@/data/project";

export default function LiteratureReview() {
  return (
    <>
      <PageHeader
        eyebrow="Literature review"
        title="Foundational themes"
        description="A condensed view of the themes covered by the reference material stored under UNI/ in the project repository."
      />

      <div className="mx-auto grid w-full max-w-content grid-cols-1 gap-6 px-6 py-10 md:grid-cols-2">
        {literatureReview.map((entry) => (
          <Card key={entry.title} title={entry.title}>
            <p className="text-ink-700">{entry.summary}</p>
          </Card>
        ))}
      </div>

      <div className="mx-auto w-full max-w-content px-6 pb-10">
        <Card title="Source material in the project">
          <p className="text-ink-700">
            The following files are bundled in the project under{" "}
            <code>UNI/</code>:
          </p>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-ink-700">
            <li>
              <code>RDF.pdf</code>, <code>processes-13-02691-v2.pdf</code>,{" "}
              <code>s11356-022-23272-6.pdf</code> &mdash; background on RDF
              production and waste characterisation.
            </li>
            <li>
              <code>Waste Reseach paper reference.pdf</code>,{" "}
              <code>3417473.3417474.pdf</code> &mdash; reference papers on
              waste classification.
            </li>
            <li>
              <code>Literature Review.pptx</code> &mdash; project literature
              review deck.
            </li>
            <li>
              <code>MSC Project Proposal*.docx</code>,{" "}
              <code>MSc Project Report Template.docx</code>,{" "}
              <code>MSc Project Mid-Point Review Template.pptx</code>,{" "}
              <code>MSc Project Viva Presentation Template.pptx</code>{" "}
              &mdash; university templates and project proposal.
            </li>
            <li>
              <code>AI Talk - AI Usage in Research.pdf</code>,{" "}
              <code>Beyond Copy-Paste.pdf</code>,{" "}
              <code>MScProject-Workshop1.1.pdf</code>,{" "}
              <code>MSc Project Management Tools.pdf</code> &mdash; university
              workshops and research-tooling guidance.
            </li>
          </ul>
        </Card>
      </div>
    </>
  );
}
