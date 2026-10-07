import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Layers,
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertTriangle,
  ArrowUpDown,
  Filter,
  ExternalLink,
  ShieldCheck,
  Loader2,
} from 'lucide-react'
import api from '../services/api'
import Card from '../components/Card'
import Spinner from '../components/Spinner'
import ErrorState from '../components/ErrorState'
import DecisionBadge from '../components/DecisionBadge'
import { formatBytes, formatDate, riskColor } from '../utils/format'

const ALLOWED_EXTENSIONS = ['pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'txt']

export default function BatchScan() {
  const [files, setFiles] = useState([])
  const [status, setStatus] = useState('idle') // idle | uploading | analyzing | completed | error
  const [progressText, setProgressText] = useState('')
  const [batchResults, setBatchResults] = useState([])
  const [error, setError] = useState(null)

  // Sorting
  const [sortField, setSortField] = useState('risk_score')
  const [sortAsc, setSortAsc] = useState(false)
  const [filterDecision, setFilterDecision] = useState('all')

  const handleFileSelect = (selected) => {
    if (!selected || !selected.length) return
    const valid = []
    const rejected = []

    Array.from(selected).forEach((f) => {
      const ext = f.name.split('.').pop()?.toLowerCase()
      if (ALLOWED_EXTENSIONS.includes(ext)) {
        valid.push(f)
      } else {
        rejected.push(f.name)
      }
    })

    if (rejected.length) {
      setError(`Ignored unsupported files: ${rejected.join(', ')}`)
    } else {
      setError(null)
    }

    setFiles((prev) => [...prev, ...valid])
  }

  const runBatchPipeline = async () => {
    if (!files.length) return
    setStatus('uploading')
    setError(null)
    setProgressText(`Uploading ${files.length} document(s)…`)

    try {
      // 1. Upload files
      const uploadRes = await api.uploadBatch(files)
      setProgressText('Registering files and executing forensic analyses…')
      setStatus('analyzing')

      // Get document IDs of successfully uploaded files:
      // Since uploadBatch registers documents in the database, we can batch analyze them:
      // We retrieve recent documents or fetch recent uploaded ones
      const history = await api.getHistory({ limit: files.length + 5 })
      const uploadedNames = new Set(files.map((f) => f.name.toLowerCase()))
      const matchedDocIds = (history.items || [])
        .filter((item) => uploadedNames.has(item.original_filename.toLowerCase()))
        .map((item) => item.document_id)

      if (!matchedDocIds.length) {
        throw new Error('Could not identify uploaded document IDs for batch execution.')
      }

      const uniqueIds = Array.from(new Set(matchedDocIds)).slice(0, files.length)
      const res = await api.analyzeBatch(uniqueIds)

      setBatchResults(res.results || [])
      setStatus('completed')
    } catch (err) {
      setError(err.message || 'Batch scanning failed')
      setStatus('error')
    }
  }

  const resetBatch = () => {
    setFiles([])
    setBatchResults([])
    setStatus('idle')
    setError(null)
  }

  const sortedResults = useMemo(() => {
    let list = [...batchResults]
    if (filterDecision !== 'all') {
      list = list.filter((item) => item.decision?.toLowerCase() === filterDecision.toLowerCase())
    }

    list.sort((a, b) => {
      let valA = a[sortField]
      let valB = b[sortField]

      if (sortField === 'risk_score') {
        valA = Number(valA) || 0
        valB = Number(valB) || 0
      } else if (sortField === 'original_filename') {
        valA = String(valA || '').toLowerCase()
        valB = String(valB || '').toLowerCase()
      } else if (sortField === 'created_at') {
        valA = new Date(valA || 0).getTime()
        valB = new Date(valB || 0).getTime()
      }

      if (valA < valB) return sortAsc ? -1 : 1
      if (valA > valB) return sortAsc ? 1 : -1
      return 0
    })

    return list
  }, [batchResults, sortField, sortAsc, filterDecision])

  const toggleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-accent/30 bg-accent/10 text-accent">
            <Layers size={18} />
          </span>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-zinc-100">
              Batch Document Ingestion & Verification
            </h1>
            <p className="text-xs text-zinc-500">
              Upload multiple files simultaneously. The engine analyzes every file in parallel and produces a consolidated risk matrix.
            </p>
          </div>
        </div>
      </div>

      {error && <ErrorState title="Batch Notice" message={error} onRetry={() => setError(null)} />}

      {/* Upload Box (Only shown if not completed) */}
      {status !== 'completed' && (
        <Card className="flex flex-col gap-4 border-edge bg-carbon">
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-edge/80 bg-surface/40 p-8 text-center transition hover:border-accent hover:bg-surface/70">
            <UploadCloud size={32} className="text-accent" />
            <h3 className="mt-2 text-sm font-semibold text-zinc-100">
              Drop batch documents here or click to browse
            </h3>
            <p className="mt-1 text-xs text-zinc-500">
              Accepts PDF, DOCX, DOC, TXT, PNG, JPG, JPEG, TIFF (Multiple files supported)
            </p>
            <input
              type="file"
              multiple
              className="hidden"
              onChange={(e) => handleFileSelect(e.target.files)}
            />
          </label>

          {files.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Selected files for batch scan ({files.length}):</span>
                <button onClick={resetBatch} className="text-red-400 hover:underline">
                  Clear queue
                </button>
              </div>

              <div className="max-h-48 overflow-y-auto divide-y divide-edge/40 rounded-md border border-edge bg-surface/60">
                {files.map((file, i) => (
                  <div key={i} className="flex items-center justify-between p-2.5 text-xs">
                    <div className="flex items-center gap-2 text-zinc-200">
                      <FileText size={15} className="text-zinc-500" />
                      <span>{file.name}</span>
                    </div>
                    <span className="font-mono text-zinc-500">{formatBytes(file.size)}</span>
                  </div>
                ))}
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={runBatchPipeline}
                  disabled={status === 'uploading' || status === 'analyzing'}
                  className="flex items-center gap-2 rounded-md bg-accent px-5 py-2 text-xs font-semibold text-night shadow-[0_0_12px_rgba(34,197,94,0.3)] transition hover:bg-accent/90 disabled:opacity-50"
                >
                  {status === 'uploading' || status === 'analyzing' ? (
                    <>
                      <Loader2 size={14} className="animate-spin text-night" />
                      <span>{progressText}</span>
                    </>
                  ) : (
                    <>
                      <Layers size={14} />
                      <span>Execute Batch Verification ({files.length})</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Results Matrix */}
      {status === 'completed' && (
        <Card className="space-y-4 border-edge bg-carbon">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge/60 pb-3">
            <div>
              <h3 className="text-sm font-bold text-zinc-100">
                Batch Verification Results ({sortedResults.length} processed)
              </h3>
              <p className="text-xs text-zinc-500">
                Sort by column header to prioritize high-risk documents for forensic audit.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <select
                value={filterDecision}
                onChange={(e) => setFilterDecision(e.target.value)}
                className="rounded-md border border-edge bg-surface px-2.5 py-1 text-xs text-zinc-200 focus:outline-none"
              >
                <option value="all">All Decisions</option>
                <option value="original">Likely Genuine</option>
                <option value="suspicious">Suspicious</option>
                <option value="forged">Likely Fake</option>
              </select>

              <button
                onClick={resetBatch}
                className="rounded-md border border-edge bg-surface px-3 py-1 text-xs text-zinc-300 hover:bg-raised"
              >
                New Batch
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-edge text-[11px] uppercase tracking-wider text-zinc-500">
                  <th
                    className="cursor-pointer py-2.5 pr-4 hover:text-zinc-200"
                    onClick={() => toggleSort('original_filename')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Document File</span>
                      <ArrowUpDown size={12} />
                    </div>
                  </th>
                  <th
                    className="cursor-pointer py-2.5 px-4 hover:text-zinc-200"
                    onClick={() => toggleSort('risk_score')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Risk Score</span>
                      <ArrowUpDown size={12} />
                    </div>
                  </th>
                  <th className="py-2.5 px-4">Verdict</th>
                  <th className="py-2.5 px-4">Status</th>
                  <th
                    className="cursor-pointer py-2.5 px-4 hover:text-zinc-200"
                    onClick={() => toggleSort('created_at')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Timestamp</span>
                      <ArrowUpDown size={12} />
                    </div>
                  </th>
                  <th className="py-2.5 pl-4 text-right">Inspection</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-edge/40">
                {sortedResults.map((item, index) => {
                  const risk = Number(item.risk_score) || 0
                  const color = riskColor(risk)
                  return (
                    <tr key={index} className="hover:bg-surface/40">
                      <td className="py-2.5 pr-4 font-medium text-zinc-200">
                        {item.original_filename}
                      </td>
                      <td className="py-2.5 px-4 font-mono font-bold" style={{ color }}>
                        {item.status === 'success' ? `${risk.toFixed(1)}/100` : '—'}
                      </td>
                      <td className="py-2.5 px-4">
                        {item.status === 'success' ? (
                          <DecisionBadge decision={item.decision} />
                        ) : (
                          <span className="text-zinc-500">Failed</span>
                        )}
                      </td>
                      <td className="py-2.5 px-4">
                        {item.status === 'success' ? (
                          <span className="rounded bg-accent/10 px-2 py-0.5 text-[10px] font-semibold text-accent">
                            Analyzed
                          </span>
                        ) : (
                          <span className="rounded bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold text-red-400">
                            Error: {item.error}
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-4 font-mono text-[11px] text-zinc-500">
                        {item.created_at ? formatDate(item.created_at) : 'Just now'}
                      </td>
                      <td className="py-2.5 pl-4 text-right">
                        {item.analysis_id ? (
                          <Link
                            to={`/analysis/${item.analysis_id}`}
                            className="inline-flex items-center gap-1 text-[11px] font-medium text-accent hover:underline"
                          >
                            <span>Report</span>
                            <ExternalLink size={12} />
                          </Link>
                        ) : (
                          <span className="text-zinc-600">—</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
