import {
  ShieldCheck,
  Cpu,
  Layers,
  Search,
  BookOpen,
  Code2,
  Users,
  CheckCircle2,
  Sparkles,
} from 'lucide-react'
import Card from '../components/Card'

export default function About() {
  const pipelineSteps = [
    {
      step: '01',
      title: 'Multi-Format Ingestion',
      tech: 'FastAPI + PyMuPDF + python-docx + Pillow',
      desc: 'Validates MIME signatures, prevents path traversal, and sanitizes filenames across PDF, DOCX, TXT, and scanned image formats.',
    },
    {
      step: '02',
      title: 'Computer Vision & OCR',
      tech: 'OpenCV + Tesseract OCR',
      desc: 'Applies image preprocessing (grayscale, bilateral filtering, Otsu binarization, deskewing) and extracts text from image-only scans.',
    },
    {
      step: '03',
      title: 'Document Type Detection',
      tech: 'Weighted Regex Pattern Scoring',
      desc: 'Automatically classifies documents into 11 semantic categories (Invoice, Certificate, Identity, etc.) to establish expected field baselines.',
    },
    {
      step: '04',
      title: 'Feature Engineering (32-D)',
      tech: 'NumPy + TF-IDF Vectorization',
      desc: 'Extracts 32 quantitative features covering text entropy, layout variance, font deviations, numerical counts, and n-gram term frequencies.',
    },
    {
      step: '05',
      title: 'Forensic Marker Engine',
      tech: 'Heuristic Rules (markers.json)',
      desc: 'Runs 7 independent detectors evaluating mathematical discrepancies (tax+subtotal!=total), inverted dates, duplicate IDs, and tampered metadata.',
    },
    {
      step: '06',
      title: 'Multi-Model Machine Learning',
      tech: 'Logistic Regression, Random Forest, XGBoost',
      desc: 'Three complementary classifiers output calibrated fraud probabilities trained on held-out stratified test splits.',
    },
    {
      step: '07',
      title: 'Ensemble Risk Scoring (0–100)',
      tech: 'Weighted Composite Calibration',
      desc: 'Fuses ML probabilities, marker penalties, and structural anomalies into a calibrated 0-100 risk index (Likely Genuine, Suspicious, Likely Fake).',
    },
    {
      step: '08',
      title: 'Explainable AI (XAI)',
      tech: 'Local Feature Attribution',
      desc: 'Generates plain-language forensic explanations and visual horizontal contribution bars explaining exactly why a verdict was reached.',
    },
  ]

  const vivaFaq = [
    {
      q: 'Why was the Security & Surveillance domain chosen?',
      a: 'Forged invoices, altered academic credentials, and doctored identity cards pose severe risks to KYC and fraud prevention. Automating forensic scrutiny ensures scalable security.',
    },
    {
      q: 'Why not use a generative LLM as a single-prompt classifier?',
      a: 'Generative models hallucinate arithmetic and fail strict forensic verification. Document security demands deterministic arithmetic checks (tax verification), explicit calendar validations, and traceable attribution.',
    },
    {
      q: 'Why compare three separate ML models?',
      a: 'Academic rigor requires benchmarking distinct algorithmic paradigms: a linear baseline (Logistic Regression), a bagging ensemble (Random Forest), and a gradient boosting framework (XGBoost).',
    },
    {
      q: 'How does the system ensure ethical AI compliance?',
      a: 'DocuGuard explicitly delivers probabilistic risk assessments ("Likely Genuine", "Suspicious", "Likely Fake") with audit trails, never claiming infallible absolute legal authenticity.',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="rounded-lg border border-edge bg-carbon p-6">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-md border border-accent/40 bg-accent/10 text-accent">
            <ShieldCheck size={24} />
          </span>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-zinc-50">
              DocuGuard – System Architecture & Viva Guide
            </h1>
            <p className="text-xs text-zinc-400">
              College AIML Mini-Project · Domain: Security & Surveillance
            </p>
          </div>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-zinc-300">
          DocuGuard is an intelligent forensic platform designed to bridge the gap between opaque machine learning classifiers and human auditability. The platform executes an 8-stage forensic pipeline combining computer vision, NLP, heuristic anomaly rules, and ensemble machine learning.
        </p>
      </div>

      {/* Forensic Pipeline Diagram */}
      <div className="space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-accent flex items-center gap-2">
          <Layers size={16} /> 8-Stage Forensic Pipeline Architecture
        </h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          {pipelineSteps.map((step) => (
            <Card key={step.step} className="border-edge bg-carbon/90 hover:border-accent/40 transition">
              <div className="flex items-center justify-between border-b border-edge/60 pb-2">
                <span className="font-mono text-xs font-bold text-accent">{step.step}</span>
                <span className="rounded bg-surface px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                  {step.tech.split('+')[0]}
                </span>
              </div>
              <h3 className="mt-2 text-xs font-semibold text-zinc-100">{step.title}</h3>
              <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">{step.desc}</p>
            </Card>
          ))}
        </div>
      </div>

      {/* Viva Presentation Q&A Guide */}
      <div className="space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-2">
          <BookOpen size={16} className="text-accent" /> Viva Defense & Academic Q&A
        </h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {vivaFaq.map((item, index) => (
            <Card key={index} className="border-edge bg-carbon">
              <h3 className="text-xs font-bold text-zinc-100 flex items-start gap-2">
                <span className="text-accent">Q:</span>
                <span>{item.q}</span>
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-zinc-400 pl-4 border-l border-accent/30">
                {item.a}
              </p>
            </Card>
          ))}
        </div>
      </div>

      {/* Team Contributions */}
      <Card className="border-edge bg-carbon">
        <h2 className="text-sm font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-2 border-b border-edge/60 pb-3">
          <Users size={16} className="text-accent" /> Project Team Roles & Contributions
        </h2>
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-md border border-edge/60 bg-surface/50 p-3">
            <span className="text-xs font-bold text-zinc-200">ML & NLP Engineer</span>
            <p className="mt-1 text-[11px] text-zinc-400">
              Feature engineering (32-D vector), Logistic Regression, Random Forest, XGBoost model training, and ROC-AUC evaluation.
            </p>
          </div>
          <div className="rounded-md border border-edge/60 bg-surface/50 p-3">
            <span className="text-xs font-bold text-zinc-200">Backend & Forensics</span>
            <p className="mt-1 text-[11px] text-zinc-400">
              FastAPI REST routing, document parsing (PyMuPDF, docx), 7-category marker detection rules, and ReportLab PDF generator.
            </p>
          </div>
          <div className="rounded-md border border-edge/60 bg-surface/50 p-3">
            <span className="text-xs font-bold text-zinc-200">Full-Stack & UI/UX</span>
            <p className="mt-1 text-[11px] text-zinc-400">
              React 19, Tailwind CSS cybersecurity dark design system, Recharts visualization, and Document Viewer components.
            </p>
          </div>
          <div className="rounded-md border border-edge/60 bg-surface/50 p-3">
            <span className="text-xs font-bold text-zinc-200">QA & Security</span>
            <p className="mt-1 text-[11px] text-zinc-400">
              Pytest automated test suite (39 tests), MIME & size validation security, SQLite schemas, and academic documentation.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
