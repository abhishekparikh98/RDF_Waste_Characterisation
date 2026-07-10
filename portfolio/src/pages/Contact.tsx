import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { project } from "@/data/project";

export default function Contact() {
  return (
    <>
      <PageHeader
        eyebrow="Contact"
        title="Author and supervisor"
        description="All contact details are taken directly from the project repository."
      />

      <div className="mx-auto w-full max-w-content grid grid-cols-1 gap-6 px-6 py-10 md:grid-cols-2">
        <Card title="Author">
          <p className="text-lg font-semibold text-ink-900">
            {project.author.name}
          </p>
          <p className="text-sm text-ink-600">{project.author.role}</p>
          <p className="mt-3 text-sm text-ink-700">
            <Badge tone="blue">Email</Badge>{" "}
            <a
              href={`mailto:${project.author.email}`}
              className="text-accent-700 hover:underline"
            >
              {project.author.email}
            </a>
          </p>
          <p className="mt-2 text-sm text-ink-700">
            <Badge tone="emerald">Repository</Badge>{" "}
            <a
              href={project.repository.url}
              className="text-accent-700 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              {project.repository.url}
            </a>
          </p>
        </Card>

        <Card title="Project status">
          <p className="text-ink-700">
            <Badge tone="emerald">{project.status}</Badge>
          </p>
          <p className="mt-3 text-sm text-ink-700">
            The dissertation submission is in <code>UNI/</code>:
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-ink-700">
            <li>
              <code>MSc Project Report Template.docx</code>
            </li>
            <li>
              <code>MSc Project Mid-Point Review Template.pptx</code>
            </li>
            <li>
              <code>MSc Project Viva Presentation Template.pptx</code>
            </li>
            <li>
              <code>MSC Project Proposal*.docx</code>
            </li>
          </ul>
        </Card>
      </div>
    </>
  );
}
