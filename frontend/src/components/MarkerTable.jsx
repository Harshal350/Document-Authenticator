import { MapPin } from 'lucide-react'
import { severityMeta } from '../utils/format'

/**
 * Reusable detected-markers table.
 * Expects markers shaped like the backend MarkerResult:
 * { category, name, severity, description, page_number, evidence, score }
 */
export default function MarkerTable({ markers = [] }) {
  if (!markers.length) {
    return (
      <div className="rounded-md border border-edge bg-carbon px-5 py-8 text-center text-sm text-zinc-500">
        No markers were detected during this analysis.
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="border-b border-edge text-left text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
            <th className="px-4 py-3">Marker</th>
            <th className="px-4 py-3">Category</th>
            <th className="px-4 py-3">Severity</th>
            <th className="px-4 py-3 w-44">Evidence</th>
            <th className="px-4 py-3 w-40">Score</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#171717]">
          {markers.map((marker) => {
            const severity = severityMeta(marker.severity)
            return (
              <tr key={marker.id ?? marker.name} className="hover:bg-raised/50">
                <td className="px-4 py-3">
                  <p className="font-medium text-zinc-100">{marker.name}</p>
                  {marker.description && (
                    <p className="mt-0.5 max-w-xs truncate text-xs text-zinc-500">
                      {marker.description}
                    </p>
                  )}
                  {marker.page_number != null && (
                    <span className="mt-1 inline-flex items-center gap-1 text-[11px] text-zinc-500">
                      <MapPin size={11} />
                      Page {marker.page_number}
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-zinc-400">
                  <span className="capitalize">{marker.category}</span>
                </td>
                <td className="px-4 py-3">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${severity.badge}`}
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${severity.dot}`} />
                    {severity.label}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <p className="line-clamp-2 text-xs leading-relaxed text-zinc-400">
                    {marker.evidence || '—'}
                  </p>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 overflow-hidden rounded-full bg-carbon">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(100, Number(marker.score) || 0)}%`,
                          backgroundColor: severity.bar,
                        }}
                      />
                    </div>
                    <span className="mono text-xs text-zinc-300">
                      {(Number(marker.score) || 0).toFixed(1)}
                    </span>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}