import { useState, useEffect } from 'react'
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  FileText,
  AlertCircle,
  Eye,
  Maximize2,
  ChevronLeft,
  ChevronRight,
  Sparkles,
} from 'lucide-react'
import api from '../services/api'
import Card from './Card'

export default function DocumentViewer({ documentId, filename, fileType, markers = [] }) {
  const [zoom, setZoom] = useState(100)
  const [content, setContent] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selectedMarker, setSelectedMarker] = useState(null)
  const [currentPage, setCurrentPage] = useState(1)

  useEffect(() => {
    if (!documentId) return
    let active = true
    setLoading(true)
    api
      .getDocumentContent(documentId)
      .then((data) => {
        if (active) setContent(data)
      })
      .catch(() => {
        if (active) setContent(null)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [documentId])

  const fileUrl = documentId ? api.getDocumentFileUrl(documentId) : null
  const isImage = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'].includes(
    fileType?.toLowerCase()?.replace('.', ''),
  )
  const isPdf = fileType?.toLowerCase()?.replace('.', '') === 'pdf'

  const totalPages = content?.stats?.pages || 1

  return (
    <Card className="flex flex-col gap-4 border-edge bg-carbon/90">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-edge/60 pb-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-md border border-accent/30 bg-accent/10 text-accent">
            <Eye size={18} />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-zinc-100">
              Forensic Document Preview & Marker Inspector
            </h3>
            <p className="text-xs text-zinc-500">
              {filename} · {fileType?.toUpperCase()} · {totalPages} page{totalPages > 1 ? 's' : ''}
            </p>
          </div>
        </div>

        {/* Zoom & Page controls */}
        <div className="flex items-center gap-2">
          {totalPages > 1 && (
            <div className="flex items-center gap-1 rounded-md border border-edge bg-surface px-2 py-1 text-xs">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage <= 1}
                className="text-zinc-400 hover:text-zinc-100 disabled:opacity-40"
              >
                <ChevronLeft size={16} />
              </button>
              <span className="px-1 text-zinc-300">
                {currentPage} / {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage >= totalPages}
                className="text-zinc-400 hover:text-zinc-100 disabled:opacity-40"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          )}

          <div className="flex items-center gap-1 rounded-md border border-edge bg-surface px-1.5 py-1 text-xs text-zinc-300">
            <button
              onClick={() => setZoom((z) => Math.max(50, z - 15))}
              className="rounded p-1 hover:bg-raised text-zinc-400 hover:text-zinc-100"
              title="Zoom out"
            >
              <ZoomOut size={15} />
            </button>
            <span className="w-10 text-center font-mono text-[11px]">{zoom}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(200, z + 15))}
              className="rounded p-1 hover:bg-raised text-zinc-400 hover:text-zinc-100"
              title="Zoom in"
            >
              <ZoomIn size={15} />
            </button>
            <button
              onClick={() => setZoom(100)}
              className="rounded p-1 hover:bg-raised text-zinc-500 hover:text-zinc-300"
              title="Reset Zoom"
            >
              <RotateCcw size={13} />
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Document Canvas / Preview */}
        <div className="flex min-h-[420px] max-h-[560px] flex-col overflow-auto rounded-md border border-edge bg-night/80 p-4 lg:col-span-8">
          {loading ? (
            <div className="flex h-full items-center justify-center text-xs text-zinc-500">
              Loading preview canvas…
            </div>
          ) : isImage && fileUrl ? (
            <div className="flex flex-1 items-center justify-center overflow-auto">
              <img
                src={fileUrl}
                alt={filename}
                style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}
                className="max-w-full rounded border border-edge/60 object-contain shadow-md transition-transform duration-150"
              />
            </div>
          ) : isPdf && fileUrl ? (
            <div className="flex flex-1 flex-col overflow-auto">
              <iframe
                src={`${fileUrl}#toolbar=0&navpanes=0`}
                title={filename}
                className="h-[460px] w-full rounded border border-edge/60 bg-zinc-900"
              />
            </div>
          ) : content?.text ? (
            <div className="flex flex-1 flex-col overflow-auto font-mono text-xs leading-relaxed text-zinc-300">
              <div className="mb-2 flex items-center justify-between border-b border-edge/40 pb-2 text-[11px] text-zinc-500">
                <span>Extracted Document Stream</span>
                <span>{content.text.length} characters</span>
              </div>
              <pre
                style={{ fontSize: `${(zoom / 100) * 12}px` }}
                className="whitespace-pre-wrap rounded bg-surface/50 p-3 text-zinc-300"
              >
                {content.text}
              </pre>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-2 text-zinc-500">
              <FileText size={32} strokeWidth={1.5} />
              <p className="text-xs">Preview unavailable for this format</p>
            </div>
          )}
        </div>

        {/* Highlighted Forensic Markers Sidebar */}
        <div className="flex flex-col gap-3 lg:col-span-4">
          <div className="flex items-center justify-between border-b border-edge/40 pb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-accent flex items-center gap-1.5">
              <Sparkles size={14} /> Highlighted Findings ({markers.length})
            </span>
            <span className="text-[11px] text-zinc-500">Click to inspect</span>
          </div>

          <div className="flex max-h-[500px] flex-col gap-2 overflow-y-auto pr-1">
            {markers.length === 0 ? (
              <div className="rounded-md border border-edge/60 bg-surface/40 p-4 text-center text-xs text-zinc-500">
                No anomalous markers flagged on this document.
              </div>
            ) : (
              markers.map((marker, index) => {
                const isSelected = selectedMarker === index
                const severity = marker.severity?.toLowerCase() || 'medium'
                const badgeColor =
                  severity === 'critical' || severity === 'high'
                    ? 'border-red-500/40 bg-red-500/10 text-red-400'
                    : severity === 'medium'
                      ? 'border-amber-500/40 bg-amber-500/10 text-amber-400'
                      : 'border-zinc-500/40 bg-zinc-500/10 text-zinc-400'

                return (
                  <div
                    key={marker.id || index}
                    onClick={() => setSelectedMarker(isSelected ? null : index)}
                    className={`cursor-pointer rounded-md border p-3 transition-all ${
                      isSelected
                        ? 'border-accent bg-accent/5 ring-1 ring-accent/30'
                        : 'border-edge bg-surface/60 hover:border-edge/80 hover:bg-surface'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-xs font-medium text-zinc-200">
                        {marker.name}
                      </span>
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${badgeColor}`}
                      >
                        {marker.severity}
                      </span>
                    </div>

                    <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">
                      {marker.description}
                    </p>

                    {marker.evidence && (
                      <div className="mt-2 rounded border border-edge/40 bg-night/80 p-2 font-mono text-[10px] text-zinc-300">
                        <span className="text-accent font-semibold">Evidence: </span>
                        {marker.evidence}
                      </div>
                    )}

                    {marker.page_number && (
                      <div className="mt-2 flex items-center gap-1 text-[10px] text-zinc-500">
                        <span>Page location:</span>
                        <span className="font-semibold text-zinc-400">Page {marker.page_number}</span>
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>
    </Card>
  )
}
