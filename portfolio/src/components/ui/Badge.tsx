import type { ReactNode } from "react";

type Tone = "blue" | "emerald" | "amber" | "slate";

interface BadgeProps {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}

const tones: Record<Tone, string> = {
  blue: "bg-accent-50 text-accent-800 ring-accent-200",
  emerald: "bg-emerald-50 text-emerald-700 ring-emerald-100",
  amber: "bg-amber-50 text-amber-700 ring-amber-100",
  slate: "bg-ink-100 text-ink-700 ring-ink-200",
};

export function Badge({ children, tone = "slate", className }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${
        tones[tone]
      } ${className ?? ""}`}
    >
      {children}
    </span>
  );
}
