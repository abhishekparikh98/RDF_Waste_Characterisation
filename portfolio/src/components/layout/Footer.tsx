import { project } from "@/data/project";

export function Footer() {
  return (
    <footer className="mt-16 border-t border-ink-200 bg-white">
      <div className="mx-auto grid max-w-content gap-8 px-6 py-10 sm:grid-cols-3">
        <div>
          <p className="font-sans-ui text-sm font-semibold text-ink-900">
            {project.shortTitle}
          </p>
          <p className="mt-2 text-xs text-ink-500">
            {project.type} &middot; {project.year}
          </p>
          <p className="mt-2 text-xs text-ink-500">
            Author: {project.author.name}
          </p>
        </div>
        <div>
          <p className="font-sans-ui text-sm font-semibold text-ink-900">
            Repository
          </p>
          <p className="mt-2 text-xs text-ink-500">
            Branch: <code className="font-mono">{project.repository.branch}</code>
          </p>
          <p className="mt-2 text-xs text-ink-500">
            <a
              className="text-accent-700 hover:underline"
              href={project.repository.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              View source on GitHub
            </a>
          </p>
        </div>
        <div>
          <p className="font-sans-ui text-sm font-semibold text-ink-900">
            About this portfolio
          </p>
          <p className="mt-2 text-xs text-ink-500">
            Built with React, Vite, TypeScript, Tailwind CSS, Framer Motion and
            React Router. All content is sourced from the actual project files.
          </p>
        </div>
      </div>
      <div className="border-t border-ink-100 bg-ink-50 py-3 text-center text-[11px] text-ink-500">
        {project.year} &middot; MSc Computing Dissertation
      </div>
    </footer>
  );
}
