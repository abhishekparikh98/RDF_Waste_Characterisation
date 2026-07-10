import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="mx-auto flex w-full max-w-content">
      <Sidebar />
      <main className="min-w-0 flex-1 px-2 sm:px-6">{children}</main>
    </div>
  );
}
