import { clamp, riskColor } from '../utils/format'

/**
 * Circular risk-score gauge.
 * Green below 40, amber 40–70, red above 70 (matching backend thresholds).
 */
export default function RiskGauge({
  value = 0,
  size = 190,
  strokeWidth = 12,
  label = 'Risk Score',
}) {
  const score = clamp(value, 0, 100)
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - score / 100)
  const color = riskColor(score)

  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#1b1b1b"
            strokeWidth={strokeWidth}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="mono font-semibold text-zinc-50" style={{ fontSize: size / 4.2 }}>
            {score.toFixed(0)}
          </span>
          <span className="text-[11px] uppercase tracking-wider text-zinc-500">
            / 100
          </span>
        </div>
      </div>
      <div className="mt-4 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
          {label}
        </span>
      </div>
    </div>
  )
}