import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface PageHeaderProps {
  eyebrow: string;
  title: string;
  description?: string;
  meta?: ReactNode;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  meta,
}: PageHeaderProps) {
  return (
    <header className="border-b border-ink-200 bg-gradient-to-b from-white to-ink-50">
      <div className="mx-auto w-full max-w-content px-6 py-12 sm:py-16">
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-xs font-semibold uppercase tracking-[0.18em] text-accent-700"
        >
          {eyebrow}
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="mt-2 font-sans-ui text-4xl font-bold tracking-tight text-ink-900 sm:text-5xl"
        >
          {title}
        </motion.h1>
        {description ? (
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="mt-4 max-w-3xl text-base leading-7 text-ink-600"
          >
            {description}
          </motion.p>
        ) : null}
        {meta ? <div className="mt-5">{meta}</div> : null}
      </div>
    </header>
  );
}
