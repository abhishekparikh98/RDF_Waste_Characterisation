import { Routes, Route } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import Home from "@/pages/Home";
import About from "@/pages/About";
import ResearchProblem from "@/pages/ResearchProblem";
import Objectives from "@/pages/Objectives";
import LiteratureReview from "@/pages/LiteratureReview";
import Datasets from "@/pages/Datasets";
import Architecture from "@/pages/Architecture";
import Methodology from "@/pages/Methodology";
import ModelComparison from "@/pages/ModelComparison";
import Results from "@/pages/Results";
import Timeline from "@/pages/Timeline";
import Reports from "@/pages/Reports";
import Documentation from "@/pages/Documentation";
import GitHub from "@/pages/GitHub";
import Contact from "@/pages/Contact";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/research-problem" element={<ResearchProblem />} />
        <Route path="/objectives" element={<Objectives />} />
        <Route path="/literature-review" element={<LiteratureReview />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/architecture" element={<Architecture />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/model-comparison" element={<ModelComparison />} />
        <Route path="/results" element={<Results />} />
        <Route path="/timeline" element={<Timeline />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/documentation" element={<Documentation />} />
        <Route path="/github" element={<GitHub />} />
        <Route path="/contact" element={<Contact />} />
        <Route
          path="*"
          element={
            <div className="px-6 py-16">
              <h1 className="font-sans-ui text-3xl font-bold">Not found</h1>
              <p className="mt-2 text-ink-600">
                The page you are looking for does not exist.
              </p>
            </div>
          }
        />
      </Routes>
    </AppShell>
  );
}
