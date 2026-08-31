import PageNav from "@/components/PageNav";
import TrainingCurves from "@/components/TrainingCurves";

export const metadata = { title: "Training curves - PC2D" };

export default function TrainingPage() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <PageNav active="training" />
      <TrainingCurves />
    </div>
  );
}
