import { motion } from "framer-motion";

interface MetricProps {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "highlight";
}

export function Metric({ label, value, hint, tone = "default" }: MetricProps) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ type: "spring", stiffness: 250, damping: 22 }}
      className={`rounded-2xl border p-5 shadow-card ${
        tone === "highlight"
          ? "border-accent-300 bg-accent-50/60"
          : "border-ink-200 bg-white"
      }`}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-500">
        {label}
      </p>
      <p
        className={`mt-2 font-sans-ui text-3xl font-bold tracking-tight ${
          tone === "highlight" ? "text-accent-800" : "text-ink-900"
        }`}
      >
        {value}
      </p>
      {hint ? <p className="mt-1 text-xs text-ink-500">{hint}</p> : null}
    </motion.div>
  );
}
