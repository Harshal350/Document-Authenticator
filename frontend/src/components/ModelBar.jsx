import { Bot } from 'lucide-react'

/**
 * Single model prediction: a two-segment probability bar.
 * genuine_probability (green) versus fake_probability (red) — sums to ~1.
 */
export default function ModelBar({ modelName, genuineProbability, fakeProbability }) {
  const genuine = Math.max(0, Math.min(1, Number(genuineProbability) || 0))
  const fake = Math.max(0, Math.min(1, Number(fakeProbability) || 0))
  const fakeWidth = (fake / (genuine + fake || 1)) * 100
  const genuineWidth = 100 - fakeWidth

  const subtitle = String(modelName || '')
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((word) => word[0]?.toUpperCase())
    .join('')

  return (
    <div className="rounded-md border border-edge bg-carbon px-4 py-3.5">
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded border border-edge bg-raised text-accent">
            <Bot size={14} strokeWidth={2} />
          </span>
          <span className="truncate text-sm font-medium text-zinc-100">{modelName}</span>
        </div>
        <span className="mono text-[11px] uppercase tracking-wider text-zinc-500">
          {subtitle || 'ML'}
        </span>
      </div>

      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-raised">
        <div
          className="h-full transition-all duration-700"
          style={{ width: `${genuineWidth}%`, backgroundColor: '#22c55e' }}
          title={`Genuine ${(genuine * 100).toFixed(1)}%`}
        />
        <div
          className="h-full transition-all duration-700"
          style={{ width: `${fakeWidth}%`, backgroundColor: '#3f1d1d' }}
          title={`Fake ${(fake * 100).toFixed(1)}%`}
        />
      </div>

      <div className="mt-2 flex items-center justify-between text-[11px]">
        <span className="flex items-center gap-1.5 text-accent">
          <span className="h-1.5 w-1.5 rounded-sm bg-accent" />
          Genuine <b className="mono">{genuine.toFixed(3)}</b>
        </span>
        <span className="flex items-center gap-1.5 text-red-400">
          <span className="h-1.5 w-1.5 rounded-sm bg-red-500/80" />
          Fake <b className="mono">{fake.toFixed(3)}</b>
        </span>
      </div>
    </div>
  )
}