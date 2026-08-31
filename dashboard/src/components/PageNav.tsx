import Link from "next/link";

type PageId = "map" | "training" | "eval";

const LINKS: { href: string; label: string; id: PageId }[] = [
  { href: "/", label: "Live map", id: "map" },
  { href: "/training", label: "Training", id: "training" },
  { href: "/eval", label: "Evaluation", id: "eval" },
];

export default function PageNav({ active }: { active: PageId }) {
  return (
    <div className="flex items-center justify-center gap-1.5 p-3">
      {LINKS.map(({ href, label, id }) => {
        const isActive = active === id;
        return (
          <Link
            key={href}
            href={href}
            className={`rounded-[2.5px] px-2.5 py-1 text-xs font-medium transition-colors ${
              isActive
                ? "bg-cyan-500/20 text-white ring-1 ring-cyan-400/50"
                : "text-zinc-400 hover:bg-white/10 hover:text-zinc-200"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </div>
  );
}
