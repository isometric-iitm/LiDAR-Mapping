import PageNav from "@/components/PageNav";
import EvalPanel from "@/components/EvalPanel";

export const metadata = { title: "Evaluation - PC2D" };

export default function EvalPage() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <PageNav active="eval" />
      <EvalPanel />
    </div>
  );
}
