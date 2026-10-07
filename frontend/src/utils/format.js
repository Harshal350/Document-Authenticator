const SEVERITY_META = {
  low: {
    label: 'Low',
    badge: 'bg-zinc-800 text-zinc-300 border-zinc-700',
    dot: 'bg-zinc-400',
    bar: '#a1a1aa',
  },
  medium: {
    label: 'Medium',
    badge: 'bg-yellow-950/40 text-yellow-400 border-yellow-900/60',
    dot: 'bg-yellow-500',
    bar: '#f59e0b',
  },
  high: {
    label: 'High',
    badge: 'bg-orange-950/40 text-orange-400 border-orange-900/60',
    dot: 'bg-orange-500',
    bar: '#f97316',
  },
  critical: {
    label: 'Critical',
    badge: 'bg-red-950/40 text-red-400 border-red-900/60',
    dot: 'bg-red-500',
    bar: '#ef4444',
  },
}

export function severityMeta(value) {
  return SEVERITY_META[String(value || '').toLowerCase()] || SEVERITY_META.medium
}

const DECISION_META = {
  original: {
    label: 'Likely Genuine',
    badge: 'bg-accent/10 text-accent border-accent/30',
    dot: 'bg-accent',
    color: '#22c55e',
  },
  suspicious: {
    label: 'Suspicious',
    badge: 'bg-yellow-950/40 text-yellow-400 border-yellow-900/60',
    dot: 'bg-yellow-500',
    color: '#f59e0b',
  },
  forged: {
    label: 'Likely Fake',
    badge: 'bg-red-950/40 text-red-400 border-red-900/60',
    dot: 'bg-red-500',
    color: '#ef4444',
  },
}

export function decisionMeta(value) {
  return DECISION_META[String(value || '').toLowerCase()] || {
    label: 'Unknown',
    badge: 'bg-zinc-800 text-zinc-300 border-zinc-700',
    dot: 'bg-zinc-500',
    color: '#a1a1aa',
  }
}

const TYPE_LABELS = {
  original: 'Original',
  forged: 'Forged',
  suspicious: 'Suspicious',
  certificate: 'Certificate',
  invoice: 'Invoice',
  id: 'ID Document',
  contract: 'Contract',
  other: 'Other',
}

export function docTypeLabel(value) {
  const key = String(value || '').toLowerCase()
  return TYPE_LABELS[key] || value || '—'
}

export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) return '—'
  const value = Number(bytes)
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(2)} MB`
}

export function formatDate(iso) {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDateShort(iso) {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
  })
}

export function formatPercent(value, digits = 1) {
  const number = Number(value)
  if (Number.isNaN(number)) return '—'
  return `${(number * 100).toFixed(digits)}%`
}

export function clamp(value, min, max) {
  const number = Number(value)
  if (Number.isNaN(number)) return min
  return Math.max(min, Math.min(max, number))
}

export function riskColor(value) {
  const score = clamp(value, 0, 100)
  if (score < 40) return '#22c55e'
  if (score < 70) return '#f59e0b'
  return '#ef4444'
}

export function statusText(status) {
  return String(status || '').replace(/_/g, ' ')
}