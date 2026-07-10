import { motion, type Variants } from "framer-motion";
import type { ReactNode } from "react";

interface SectionProps {
  id?: string;
  eyebrow?: string;
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}

const variants: Variants = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

export function Section({
  id,
  eyebrow,
  title,
  description,
  children,
  className,
}: SectionProps) {
  return (
    <section
      id={id}
      className={`scroll-mt-nav py-12 sm:py-16 ${className ?? ""}`}
    >
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.15 }}
        variants={variants}
        transition={{ duration: 0.45, ease: "easeOut" }}
        className="mx-auto w-full max-w-content px-6"
      >
        {eyebrow ? (
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent-700">
            {eyebrow}
          </p>
        ) : null}
        <h2 className="mt-2 font-sans-ui text-3xl font-bold tracking-tight text-ink-900 sm:text-4xl">
          {title}
        </h2>
        {description ? (
          <p className="mt-3 max-w-3xl text-base leading-7 text-ink-600">
            {description}
          </p>
        ) : null}
        <div className="mt-8">{children}</div>
      </motion.div>
    </section>
  );
}
