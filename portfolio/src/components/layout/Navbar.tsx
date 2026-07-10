import { NavLink } from "react-router-dom";
import { project } from "@/data/project";

export function Navbar() {
  return (
    <header className="sticky top-0 z-30 border-b border-ink-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-content items-center justify-between px-6 py-3">
        <NavLink
          to="/"
          className="flex items-center gap-3 font-sans-ui"
        >
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-accent-700 text-white shadow-sm">
            <svg
              viewBox="0 0 24 24"
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <path d="M4 18 L9 8 L13 14 L17 10 L20 18" />
              <circle cx="4" cy="18" r="1.2" fill="currentColor" />
              <circle cx="9" cy="8" r="1.2" fill="currentColor" />
              <circle cx="13" cy="14" r="1.2" fill="currentColor" />
              <circle cx="17" cy="10" r="1.2" fill="currentColor" />
              <circle cx="20" cy="18" r="1.2" fill="currentColor" />
            </svg>
          </span>
          <div className="leading-tight">
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-accent-700">
              {project.type}
            </p>
            <p className="text-sm font-semibold text-ink-900">
              Waste Characterisation for RDF
            </p>
          </div>
        </NavLink>

        <nav className="hidden items-center gap-1 md:flex">
          {[
            { to: "/about", label: "About" },
            { to: "/datasets", label: "Datasets" },
            { to: "/architecture", label: "Architecture" },
            { to: "/model-comparison", label: "Results" },
            { to: "/timeline", label: "Timeline" },
            { to: "/reports", label: "Reports" },
            { to: "/contact", label: "Contact" },
          ].map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-accent-50 text-accent-800"
                    : "text-ink-600 hover:bg-ink-100 hover:text-ink-900"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
