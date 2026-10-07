import { useCallback, useEffect, useMemo, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  FileText,
  Download,
  Printer,
  FileWarning,
  Boxes,
  BookOpen,
  Rows3,
  Hash,
  Table2,
  ScanSearch,
} from 'lucide-react'

import api from '../services/api'
import RiskGauge from '../components/RiskGauge'
import MarkerTable from '../components/MarkerTable'
import ModelBar from '../components/ModelBar'
import DecisionBadge from '../components/DecisionBadge'
import ErrorState from '../components/ErrorState'
import Spinner from '../components/Spinner'
import Card from '../components/Card'
import {
  formatBytes,
  formatDate,
  formatPercent,
  docTypeLabel,
  riskColor,
} from '../utils/format'

export default function AnalysisResult() {
  const { analysisId } = useParams()
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [reportBusy, setReportBusy] = useState(false)
  const [reportError, setReportError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await api.getResults(analysisId)
      setAnalysis(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [analysisId])

  useEffect(() => {
    load()
  }, [load])

  const contributions = useMemo(() => {
    if (!analysis) return []
    const parts = [
      { key: 'ml', label: 'Machine Learning', value: analysis.ml_risk_score },
      { key: 'marker', label: 'Detected Markers', value: analysis.marker_risk_score },
      { key: 'structural', label: 'Structural', value: analysis.structural_risk_score },
      { key: 'consistency', label: 'Consistency', value: analysis.consistency_risk_score },
    ]
    const max = Math.max(1, ...parts.map((p) => Number(p.value) || 0))
    return parts.map((part) => ({
      ...part,
      value: Number(part.value) || 0,
      width: Math.min(100, ((Number(part.value) || 0) / max) * 100),
    }))
  }, [analysis])

  const downloadReport = async () => {
    setReportBusy(true)
    setReportError(null)
    try {
      const report = await api.generateReport(analysisId)
      if (report?.report_url) {
        const url = report.report_url.startsWith('http')
          ? report.report_url
          : `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}${report.report_url}`
        window.open(url, '_blank', 'noopener,noreferrer')
      } else {
        setReportError('No report URL returned by the server.')
      }
    } catch (err) {
      setReportError(err.message)
    } finally {
      setReportBusy(false)
    }
  }

  if (loading) {
    return (
      <div className="flex h-72 items-center justify-center text-zinc-500">
        <div className="flex flex-col items-center gap-3">
          <Spinner size={28} />
          <span className="text-sm">Loading analysis result…</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-zinc-50">
            Analysis Result
          </h1>
        </div>
        <ErrorState message={error} onRetry={load} />
      </div>
    )
  }

  if (!analysis) return null

  const doc = analysis.document || {}
  const predictions = analysis.model_predictions || []
  const markers = analysis.markers || []

  const docStats = [
    { icon: Table2, label: 'Pages', value: doc.pages ?? '—' },
    { icon: BookOpen, label: 'Words', value: doc.words ?? '—' },
    { icon: Hash, label: 'Characters', value: doc.characters ?? '—' },
    { icon: Boxes, label: 'File Size', value: formatBytes(doc.file_size) },
    { icon: FileText, label: 'File Type', value: String(doc.file_type || '—').toUpperCase() },
    { icon: Rows3, label: 'Document Type', value: docTypeLabel(doc.document_type) },
  ]

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <Link to="/history" className="hover:text-accent">History</Link>
            <span>/</span>
            <span>Analysis #{analysis.id}</span>
            <span className="rounded border border-edge bg-carbon px-1.5 py-0.5 font-mono text-[10px] uppercase">
              {doc.file_type}
            </span>
          </div>
          <h1 className="mt-2 truncate text-xl font-semibold tracking-tight text-zinc-50">
            {doc.original_filename || 'Document'}
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            {docTypeLabel(doc.document_type)} · analyzed {formatDate(analysis.created_at)}
          </p>
        </div>
        <button
          onClick={downloadReport}
          disabled={reportBusy}
          className="btn btn-accent"
        >
          <Download size={16} />
          {reportBusy ? 'Generating report…' : 'Download Report'}
        </button>
      </div>

      {reportError && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-400">
          Report generation failed: {reportError}
        </div>
      )}

      {/* --- Top: verdict + gauge --- */}
      <div className="panel p-6">
        <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-center">
          <RiskGauge value={analysis.risk_score} label="Overall Risk" />

          <div className="flex min-w-0 flex-1 flex-col items-center gap-6 lg:items-start">
            <div className="text-center lg:text-left">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                Decision
              </p>
              <div className="mt-2 flex justify-center lg:justify-start">
                <DecisionBadge decision={analysis.decision} />
              </div>
              <p className="mt-3 max-w-xl text-sm leading-relaxed text-zinc-400">
                Confidence:{' '}
                <span className="mono font-medium text-zinc-100">
                  {formatPercent(analysis.confidence)}
                </span>{' '}
                · based on {markers.length} detected markers and{' '}
                {predictions.length} model predictions.
              </p>
            </div>

            <div className="grid w-full max-w-xl grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: 'ML Risk', value: analysis.ml_risk_score },
                { label: 'Marker Risk', value: analysis.marker_risk_score },
                { label: 'Structural', value: analysis.structural_risk_score },
                { label: 'Consistency', value: analysis.consistency_risk_score },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-md border border-edge bg-carbon px-3 py-2.5"
                >
                  <p className="text-[11px] text-zinc-500">{item.label}</p>
                  <p className="mono mt-1 text-lg font-semibold" style={{ color: riskColor(item.value) }}>
                    {(Number(item.value) || 0).toFixed(1)}
                    <span className="text-xs text-zinc-600"> /100</span>
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* --- Model predictions --- */}
      <Card
        title="Model Predictions"
        subtitle="Probability estimates from each trained classifier"
        bodyClassName="p-5"
      >
        {predictions.length ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {predictions.map((prediction) => (
              <ModelBar
                key={`${prediction.model_name}-${prediction.id}`}
                modelName={prediction.model_name}
                genuineProbability={prediction.genuine_probability}
                fakeProbability={prediction.fake_probability}
              />
            ))}
          </div>
        ) : (
          <div className="flex items-center gap-3 rounded-md border border-edge bg-carbon px-4 py-6 text-sm text-zinc-500">
            <FileWarning size={18} className="shrink-0" />
            No model predictions were recorded for this analysis (ML may be disabled).
          </div>
        )}
      </Card>

      {/* --- Detected markers --- */}
      <Card
        title="Detected Markers"
        subtitle="Anomalies and forensic signals found in the document"
        bodyClassName="p-0"
      >
        <MarkerTable markers={markers} />
      </Card>

      {/* --- Document statistics + explainability --- */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card
          title="Document Statistics"
          subtitle="Extracted document properties"
          bodyClassName="p-5"
        >
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {docStats.map((stat) => {
              const Icon = stat.icon
              return (
                <div
                  key={stat.label}
                  className="rounded-md border border-edge bg-carbon px-3.5 py-3"
                >
                  <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
                    <Icon size={13} className="text-zinc-600" />
                    {stat.label}
                  </div>
                  <p className="mono mt-1.5 truncate text-base font-semibold text-zinc-100">
                    {stat.value}
                  </p>
                </div>
              )
            })}
          </div>
        </Card>

        <Card
          title="Risk Explainability"
          subtitle="Contribution of each component to the overall risk score"
          bodyClassName="p-5"
        >
          <div className="space-y-4">
            {contributions.map((part) => (
              <div key={part.key}>
                <div className="mb-1.5 flex items-center justify-between text-xs">
                  <span className="text-zinc-400">{part.label}</span>
                  <span className="mono font-medium text-zinc-100">{part.value.toFixed(1)}</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-carbon">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${part.width}%`,
                      backgroundColor: riskColor(part.value),
                    }}
                  />
                </div>
              </div>
            ))}
            <div className="border-t border-edge pt-3">
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <ScanSearch size={14} className="text-accent" />
                <p>
                  The composite risk score is a weighted combination of marker,
                  ML, structural and consistency signals. The higher a component
                  scores, the more strongly it contributed to{' '}
                  <span className="font-medium text-zinc-300">
                    {analysis.risk_score.toFixed(1)}/100
                  </span>.
                </p>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <div className="flex items-center justify-end gap-3 pt-1">
        <Link to="/analyze" className="btn btn-ghost">
          Analyze Another Document
        </Link>
        <button onClick={window.print} className="btn btn-ghost">
          <Printer size={16} /> Print View
        </button>
      </div>
    </div>
  )
}