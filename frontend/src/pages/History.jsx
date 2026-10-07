import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Search,
  FileText,
  Eye,
  Trash2,
  ChevronLeft,
  ChevronRight,
  MapPin,
} from 'lucide-react'

import api from '../services/api'
import Card from '../components/Card'
import DecisionBadge from '../components/DecisionBadge'
import Spinner from '../components/Spinner'
import ErrorState from '../components/ErrorState'
import { formatDate, formatPercent, riskColor } from '../utils/format'

const PAGE_SIZE = 20

const FILTERS = [
  { key: null, label: 'All' },
  { key: 'original', label: 'Genuine' },
  { key: 'suspicious', label: 'Suspicious' },
  { key: 'forged', label: 'Fake' },
]

export default function History() {
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [skip, setSkip] = useState(0)
  const [filter, setFilter] = useState(null)
  const [search, setSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [deleting, setDeleting] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getHistory({ skip, limit: PAGE_SIZE, decision: filter, search })
      setItems(data.items || [])
      setTotal(data.total || 0)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [skip, filter, search])

  useEffect(() => {
    load()
  }, [load])

  const submitSearch = () => {
    setSkip(0)
    setSearch(searchInput.trim())
  }

  const selectFilter = (key) => {
    setFilter(key)
    setSkip(0)
  }

  const remove = async (analysisId) => {
    setDeleting(analysisId)
    setError(null)
    try {
      await api.deleteHistory(analysisId)
      await load()
    } catch (err) {
      setError(`Could not delete analysis #${analysisId}: ${err.message}`)
    } finally {
      setDeleting(null)
    }
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const currentPage = Math.floor(skip / PAGE_SIZE) + 1

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-zinc-50">Scan History</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Complete record of all document analyses, sorted by date.
        </p>
      </div>

      {/* --- Filters --- */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative w-full max-w-sm">
          <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-zinc-600" />
          <input
            className="input pl-10"
            placeholder="Search by file name or type…"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submitSearch()}
          />
        </div>
        <div className="flex items-center gap-1.5 rounded-lg border border-edge bg-surface p-1">
          {FILTERS.map((item) => (
            <button
              key={item.label}
              onClick={() => selectFilter(item.key)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                filter === item.key
                  ? 'bg-accent text-black'
                  : 'text-zinc-400 hover:text-zinc-100'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
        {(search || filter) && (
          <button
            onClick={() => {
              setSearch('')
              setSearchInput('')
              setFilter(null)
            }}
            className="btn btn-ghost px-3 py-1.5 text-xs"
          >
            Clear filters
          </button>
        )}
      </div>

      {error && !loading && (
        <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* --- Table --- */}
      <Card
        title="Analyses"
        subtitle={`${total} record(s) · page ${currentPage} of ${totalPages}`}
        bodyClassName="p-0"
      >
        {loading ? (
          <div className="flex h-64 items-center justify-center text-zinc-500">
            <div className="flex flex-col items-center gap-3">
              <Spinner size={28} />
              <span className="text-sm">Loading scan history…</span>
            </div>
          </div>
        ) : error && !items.length ? (
          <div className="p-5">
            <ErrorState message={error} onRetry={load} />
          </div>
        ) : items.length === 0 ? (
          <div className="flex flex-col items-center px-5 py-14 text-center">
            <FileText size={22} className="text-zinc-600" />
            <p className="mt-2 text-sm text-zinc-500">No analyses match the current filters.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[860px] text-sm">
              <thead>
                <tr className="border-b border-edge text-left text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  <th className="px-5 py-3">Date</th>
                  <th className="px-5 py-3">File Name</th>
                  <th className="px-5 py-3">Document Type</th>
                  <th className="px-5 py-3">Risk Score</th>
                  <th className="px-5 py-3">Confidence</th>
                  <th className="px-5 py-3">Decision</th>
                  <th className="px-5 py-3">Detected Markers</th>
                  <th className="px-5 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#171717]">
                {items.map((item) => (
                  <tr key={item.analysis_id} className="hover:bg-raised/50">
                    <td className="whitespace-nowrap px-5 py-3.5 text-xs text-zinc-500">
                      {formatDate(item.created_at)}
                    </td>
                    <td className="px-5 py-3.5">
                      <Link
                        to={`/analysis/${item.analysis_id}`}
                        className="flex max-w-[240px] items-center gap-2 truncate font-medium text-zinc-100 hover:text-accent"
                      >
                        <FileText size={14} className="shrink-0 text-zinc-500" />
                        <span className="truncate">{item.original_filename}</span>
                      </Link>
                      <span className="mono mt-0.5 block text-[10px] uppercase text-zinc-600">
                        #{item.analysis_id}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="rounded border border-edge bg-carbon px-2 py-0.5 font-mono text-[11px] uppercase text-zinc-400">
                        {item.file_type}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="mono flex items-center gap-2 font-semibold" style={{ color: riskColor(item.risk_score) }}>
                        {item.risk_score?.toFixed(1)}
                      </span>
                    </td>
                    <td className="mono px-5 py-3.5 text-xs text-zinc-400">
                      {formatPercent(item.confidence)}
                    </td>
                    <td className="px-5 py-3.5">
                      <DecisionBadge decision={item.decision} size="sm" />
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="inline-flex items-center gap-1.5 text-xs text-zinc-400">
                        <MapPin size={12} className="text-zinc-600" />
                        {item.marker_count ?? 0} marker(s)
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center justify-end gap-2">
                        <Link
                          to={`/analysis/${item.analysis_id}`}
                          className="btn btn-ghost px-2.5 py-1.5 text-xs"
                          title="View analysis"
                        >
                          <Eye size={14} />
                          View
                        </Link>
                        <button
                          onClick={() => remove(item.analysis_id)}
                          disabled={deleting === item.analysis_id}
                          className="btn btn-danger px-2.5 py-1.5 text-xs"
                          title="Delete scan"
                        >
                          {deleting === item.analysis_id ? (
                            <Spinner size={14} />
                          ) : (
                            <Trash2 size={14} />
                          )}
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* --- Pagination --- */}
        {!loading && total > PAGE_SIZE && (
          <div className="flex items-center justify-between border-t border-edge px-5 py-3.5">
            <p className="text-xs text-zinc-500">
              Showing {skip + 1}–{Math.min(skip + PAGE_SIZE, total)} of {total}
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSkip((s) => Math.max(0, s - PAGE_SIZE))}
                disabled={skip === 0}
                className="btn btn-ghost px-3 py-1.5 text-xs"
              >
                <ChevronLeft size={14} /> Prev
              </button>
              <span className="mono text-xs text-zinc-500">
                {currentPage} / {totalPages}
              </span>
              <button
                onClick={() => setSkip((s) => Math.min(total - PAGE_SIZE, s + PAGE_SIZE))}
                disabled={skip + PAGE_SIZE >= total}
                className="btn btn-ghost px-3 py-1.5 text-xs"
              >
                Next <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}