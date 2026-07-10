import type { ReactNode } from "react";

interface FigureProps {
  src: string;
  alt: string;
  caption?: ReactNode;
  width?: number;
  height?: number;
}

export function Figure({ src, alt, caption, width, height }: FigureProps) {
  return (
    <figure className="rounded-2xl border border-ink-200 bg-white p-4 shadow-card">
      <img
        src={src}
        alt={alt}
        width={width}
        height={height}
        loading="lazy"
        className="block w-full rounded-xl border border-ink-100"
      />
      {caption ? (
        <figcaption className="mt-3 text-center text-xs text-ink-500">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
