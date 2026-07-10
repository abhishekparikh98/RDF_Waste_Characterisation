import { NavLink } from "react-router-dom";
import { navItems } from "@/lib/navigation";
import type { ReactNode } from "react";

interface SidebarProps {
  children?: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const groups: Record<string, typeof navItems> = {};
  for (const item of navItems) {
    if (!groups[item.group]) groups[item.group] = [];
    groups[item.group].push(item);
  }

  return (
    <aside className="sticky top-20 hidden h-[calc(100vh-6rem)] w-72 shrink-0 overflow-y-auto border-r border-ink-200 bg-white/40 px-4 py-6 lg:block">
      <nav className="space-y-6">
        {Object.entries(groups).map(([group, items]) => (
          <div key={group}>
            <p className="px-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-500">
              {group}
            </p>
            <ul className="mt-2 space-y-1">
              {items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.to === "/"}
                    className={({ isActive }) =>
                      `block rounded-lg px-3 py-2 text-sm transition ${
                        isActive
                          ? "bg-accent-50 font-semibold text-accent-800"
                          : "text-ink-700 hover:bg-ink-100"
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      {children}
    </aside>
  );
}
