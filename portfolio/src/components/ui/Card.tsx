import type { ReactNode } from "react";

interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
}

export function Card({ title, subtitle, children, footer, className }: CardProps) {
  return (
    <div
      className={`rounded-2xl border border-ink-200 bg-white p-6 shadow-card transition hover:shadow-cardHover ${
        className ?? ""
      }`}
    >
      {title ? (
        <h3 className="font-sans-ui text-lg font-semibold text-ink-900">
          {title}
        </h3>
      ) : null}
      {subtitle ? (
        <p className="mt-1 text-sm text-ink-500">{subtitle}</p>
      ) : null}
      {(title || subtitle) && <div className="my-4 h-px bg-ink-100" />}
      <div className="text-sm leading-6 text-ink-700">{children}</div>
      {footer ? (
        <div className="mt-4 border-t border-ink-100 pt-4 text-xs text-ink-500">
          {footer}
        </div>
      ) : null}
    </div>
  );
}
