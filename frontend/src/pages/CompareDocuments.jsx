import { useState, useEffect } from 'react'
import {
  GitCompare,
  UploadCloud,
  AlertTriangle,
  CheckCircle2,
  FileText,
  ShieldAlert,
  ArrowRight,
  RefreshCw,
  Sparkles,
} from 'lucide-react'
import api from '../services/api'
import Card from '../components/Card'
import Spinner from '../components/Spinner'
import ErrorState from '../components/ErrorState'
import DecisionBadge from '../components/DecisionBadge'
import { formatBytes, formatDate } from '../utils/format'

export default function CompareDocuments() {
  const [historyDocs, setHistoryDocs] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(true)

  const [docAId, setDocAId] = useState('')
  const [docBId, setDocBId] = useState('')
  const [fileA, setFileA] = useState(null)
  const [fileB, setFileB] = useState(null)

  const [comparing, setComparing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let active = true
    api
      .getHistory({ limit: 50 })
      .then((data) => {
        if (active) {
          setHistoryDocs(data.items || [])
          if (data.items?.length >= 2) {
            setDocAId(String(data.items[0].document_id))
            setDocBId(String(data.items[1].document_id))
          } else if (data.items?.length === 1) {
            setDocAId(String(data.items[0].document_id))
          }
        }
      })
      .catch(() => {})
      .finally(() => {
        if (active) setLoadingHistory(false)
      })
    return () => {
      active = false
    }
  }, [])

  const handleUploadAndSelect = async (file, targetSlot) => {
    try {
      const res = await api.uploadDocument(file)
      if (targetSlot === 'A') {
        setDocAId(String(res.document_id))
        setFileA(file)
      } else {
        setDocBId(String(res.document_id))
        setFileB(file)
      }
    } catch (err) {
      setError(`Failed to upload ${file.name}: ${err.message}`)
    }
  }

  const runComparison = async () => {
    if (!docAId || !docBId) {
      setError('Please select or upload both Document A and Document B.')
      return
    }
    if (docAId === docBId) {
      setError('Document A and Document B must be different documents to perform comparative analysis.')
      return
    }

    setComparing(true)
    setError(null)
    setResult(null)

    try {
      const comparison = await api.compareDocuments(Number(docAId), Number(docBId))
      setResult(comparison)
    } catch (err) {
      setError(err.message || 'Comparison failed')
    } finally {
      setComparing(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-accent/30 bg-accent/10 text-accent">
            <GitCompare size={18} />
          </span>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-zinc-100">
              Cross-Document Forensic Comparison
            </h1>
            <p className="text-xs text-zinc-500">
              Compare two documents side-by-side to detect date mismatches, numerical inconsistencies, differing metadata, and structural divergence.
            </p>
          </div>
        </div>
      </div>

      {error && <ErrorState title="Comparison Error" message={error} onRetry={() => setError(null)} />}

      {/* Document Selectors / Uploaders */}
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        {/* Document A */}
        <Card className="flex flex-col gap-4 border-edge bg-carbon">
          <div className="flex items-center justify-between border-b border-edge/60 pb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-accent">
              Document A (Reference)
            </span>
            {fileA && <span className="text-[11px] text-zinc-400">{fileA.name}</span>}
          </div>

          <div className="space-y-3">
            <label className="text-xs font-medium text-zinc-300">
              Select previously uploaded document:
            </label>
            <select
              value={docAId}
              onChange={(e) => setDocAId(e.target.value)}
              className="w-full rounded-md border border-edge bg-surface px-3 py-2 text-xs text-zinc-200 focus:border-accent focus:outline-none"
            >
              <option value="">-- Choose document --</option>
              {historyDocs.map((item) => (
                <option key={item.document_id} value={item.document_id}>
                  #{item.document_id} · {item.original_filename} ({item.decision})
                </option>
              ))}
            </select>

            <div className="relative flex items-center justify-center py-2 text-center">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-edge/40" />
              </div>
              <span className="relative bg-carbon px-2 text-[10px] uppercase text-zinc-500">
                Or upload new file
              </span>
            </div>

            <label className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-edge/80 bg-surface/40 p-4 transition hover:border-accent hover:bg-surface/70">
              <UploadCloud size={20} className="text-zinc-500" />
              <span className="mt-1.5 text-xs text-zinc-300">Browse Document A</span>
              <span className="text-[10px] text-zinc-500">PDF, DOCX, Images, TXT</span>
              <input
                type="file"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleUploadAndSelect(e.target.files[0], 'A')}
              />
            </label>
          </div>
        </Card>

        {/* Document B */}
        <Card className="flex flex-col gap-4 border-edge bg-carbon">
          <div className="flex items-center justify-between border-b border-edge/60 pb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-blue-400">
              Document B (Candidate)
            </span>
            {fileB && <span className="text-[11px] text-zinc-400">{fileB.name}</span>}
          </div>

          <div className="space-y-3">
            <label className="text-xs font-medium text-zinc-300">
              Select previously uploaded document:
            </label>
            <select
              value={docBId}
              onChange={(e) => setDocBId(e.target.value)}
              className="w-full rounded-md border border-edge bg-surface px-3 py-2 text-xs text-zinc-200 focus:border-blue-400 focus:outline-none"
            >
              <option value="">-- Choose document --</option>
              {historyDocs.map((item) => (
                <option key={item.document_id} value={item.document_id}>
                  #{item.document_id} · {item.original_filename} ({item.decision})
                </option>
              ))}
            </select>

            <div className="relative flex items-center justify-center py-2 text-center">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-edge/40" />
              </div>
              <span className="relative bg-carbon px-2 text-[10px] uppercase text-zinc-500">
                Or upload new file
              </span>
            </div>

            <label className="flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-edge/80 bg-surface/40 p-4 transition hover:border-blue-400 hover:bg-surface/70">
              <UploadCloud size={20} className="text-zinc-500" />
              <span className="mt-1.5 text-xs text-zinc-300">Browse Document B</span>
              <span className="text-[10px] text-zinc-500">PDF, DOCX, Images, TXT</span>
              <input
                type="file"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleUploadAndSelect(e.target.files[0], 'B')}
              />
            </label>
          </div>
        </Card>
      </div>

      {/* Execute Comparison Button */}
      <div className="flex justify-center">
        <button
          onClick={runComparison}
          disabled={!docAId || !docBId || comparing}
          className="flex items-center gap-2 rounded-md bg-accent px-6 py-2.5 text-xs font-semibold text-night shadow-[0_0_15px_rgba(34,197,94,0.3)] transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {comparing ? (
            <>
              <Spinner size={14} className="text-night" />
              <span>Analyzing cross-document inconsistencies…</span>
            </>
          ) : (
            <>
              <GitCompare size={15} />
              <span>Compare Documents Now</span>
            </>
          )}
        </button>
      </div>

      {/* Comparison Results */}
      {result && (
        <div className="space-y-6">
          {/* Verdict Banner */}
          <div
            className={`rounded-lg border p-4 shadow-lg ${
              result.differences?.length > 0
                ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
                : 'border-accent/40 bg-accent/10 text-accent'
            }`}
          >
            <div className="flex items-start gap-3">
              {result.differences?.length > 0 ? (
                <AlertTriangle size={22} className="shrink-0 text-amber-400" />
              ) : (
                <CheckCircle2 size={22} className="shrink-0 text-accent" />
              )}
              <div>
                <h3 className="text-sm font-bold">
                  {result.comparison?.verdict || 'Comparison Complete'}
                </h3>
                <p className="mt-1 text-xs leading-relaxed text-zinc-300">
                  Calculated composite text & structural similarity:{' '}
                  <span className="font-mono font-bold text-zinc-100">
                    {result.text_similarity}%
                  </span>
                  . Risk delta between documents:{' '}
                  <span className="font-mono font-bold text-zinc-100">
                    {result.comparison?.risk_difference} pts
                  </span>
                  .
                </p>
              </div>
            </div>
          </div>

          {/* Suspicious Inconsistencies List */}
          {result.differences?.length > 0 && (
            <Card className="border-edge bg-carbon">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-red-400 flex items-center gap-1.5">
                <ShieldAlert size={14} /> Detected Discrepancies & Anomalies ({result.differences.length})
              </h3>
              <ul className="mt-3 space-y-2">
                {result.differences.map((diff, index) => (
                  <li
                    key={index}
                    className="flex items-start gap-2 rounded border border-red-500/30 bg-red-500/5 p-2.5 text-xs text-red-200"
                  >
                    <span className="text-red-400 font-bold">•</span>
                    <span>{diff}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* Side-by-Side Attribute Comparison Table */}
          <Card className="border-edge bg-carbon">
            <h3 className="border-b border-edge/60 pb-3 text-xs font-semibold uppercase tracking-wider text-zinc-300">
              Side-by-Side Attribute Matrix
            </h3>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-edge text-[11px] uppercase tracking-wider text-zinc-500">
                    <th className="py-2.5 pr-4 font-semibold">Attribute</th>
                    <th className="py-2.5 px-4 font-semibold text-accent">Document A ({result.document_a.original_filename})</th>
                    <th className="py-2.5 px-4 font-semibold text-blue-400">Document B ({result.document_b.original_filename})</th>
                    <th className="py-2.5 pl-4 font-semibold text-right">Alignment</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-edge/40">
                  {result.field_comparison?.map((field, index) => (
                    <tr key={index} className="hover:bg-surface/40">
                      <td className="py-2.5 pr-4 font-medium text-zinc-300">{field.attribute}</td>
                      <td className="py-2.5 px-4 font-mono text-zinc-200">{field.value_a}</td>
                      <td className="py-2.5 px-4 font-mono text-zinc-200">{field.value_b}</td>
                      <td className="py-2.5 pl-4 text-right">
                        {field.matches ? (
                          <span className="rounded bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent">
                            Match
                          </span>
                        ) : (
                          <span className="rounded bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-400">
                            Mismatch
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Common Entities Discovered */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <Card className="border-edge bg-carbon">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
                Shared Dates ({result.common_dates?.length || 0})
              </span>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {result.common_dates?.length > 0 ? (
                  result.common_dates.map((d, i) => (
                    <span key={i} className="rounded bg-surface px-2 py-0.5 font-mono text-[11px] text-zinc-300">
                      {d}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-zinc-600">None detected</span>
                )}
              </div>
            </Card>

            <Card className="border-edge bg-carbon">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
                Shared Reference IDs ({result.common_ids?.length || 0})
              </span>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {result.common_ids?.length > 0 ? (
                  result.common_ids.map((id, i) => (
                    <span key={i} className="rounded bg-surface px-2 py-0.5 font-mono text-[11px] text-accent">
                      {id}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-zinc-600">None detected</span>
                )}
              </div>
            </Card>

            <Card className="border-edge bg-carbon">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
                Shared Numbers ({result.common_numbers?.length || 0})
              </span>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {result.common_numbers?.length > 0 ? (
                  result.common_numbers.slice(0, 8).map((n, i) => (
                    <span key={i} className="rounded bg-surface px-2 py-0.5 font-mono text-[11px] text-zinc-300">
                      {n}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-zinc-600">None detected</span>
                )}
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
