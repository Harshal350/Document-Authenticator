import { useEffect, useMemo, useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  FileText,
  ShieldCheck,
  AlertTriangle,
  ShieldAlert,
  Scale,
  PieChart as PieIcon,
  ArrowRight,
} from 'lucide-react'
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'

import api from '../services/api'
import StatCard from '../components/StatCard'
import Card from '../components/Card'
import ChartTooltip from '../components/ChartTooltip'
import DecisionBadge from '../components/DecisionBadge'
import ErrorState from '../components/ErrorState'
import Spinner from '../components/Spinner'
import { formatDate, riskColor } from '../utils/format'

const PIE_COLORS = {
  original: '#22c55e',
  suspicious: '#f59e0b',
  forged: '#ef4444',
}

const SEVERITY_COLORS = {
  low: '#a1a1aa',
  medium: '#f59e0b',
  high: '#f97316',
  critical: '#ef4444',
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getDashboardStats()
      setStats(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const chartData = useMemo(() => {
    if (!stats) return { pie: [], risk: [], severity: [] }

    const pie = Object.entries(stats.decision_distribution || {}).map(
      ([key, value]) => ({
        name: key,
        label:
          key === 'original'
            ? 'Likely Genuine'
            : key === 'suspicious'
              ? 'Suspicious'
              : 'Likely Fake',
        value: Number(value) || 0,
      }),
    )

    const risk = (stats.risk_distribution || []).map((bucket) => ({
      name: bucket.label,
      count: bucket.count,
    }))

    const severity = Object.entries(stats.severity_distribution || {})
      .map(([key, value]) => ({
        name: key.charAt(0).toUpperCase() + key.slice(1),
        key,
        count: Number(value) || 0,
      }))
      .filter((entry) => entry.count > 0)

    return { pie, risk, severity }
  }, [stats])

  if (loading) {
    return (
      <div className="flex h-72 items-center justify-center text-zinc-500">
        <div className="flex flex-col items-center gap-3">
          <Spinner size={28} />
          <span className="text-sm">Loading dashboard statistics…</span>
        </div>
      </div>
    )
  }

  if (error) {
    return <ErrorState message={error} onRetry={load} />
  }

  const decision = stats.decision_distribution || {}
  const severity = stats.severity_distribution || {}

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-zinc-50">
          Dashboard
        </h1>
        <p className="mt-1 text-sm text-zinc-500">
          Aggregate forensic pipeline statistics across all analyzed documents.
        </p>
      </div>

      {/* --- Stat cards --- */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        <StatCard
          icon={FileText}
          label="Total Documents"
          value={stats.total_documents || 0}
          sub={`${stats.files_this_week || 0} uploaded this week`}
          iconClass="text-zinc-400"
        />
        <StatCard
          icon={ShieldCheck}
          label="Likely Genuine"
          value={(decision.original ?? 0).toLocaleString()}
          sub={`of ${stats.total_analyses || 0} analyses`}
          iconClass="text-accent"
        />
        <StatCard
          icon={AlertTriangle}
          label="Suspicious"
          value={(decision.suspicious ?? 0).toLocaleString()}
          sub={`of ${stats.total_analyses || 0} analyses`}
          iconClass="text-yellow-500"
        />
        <StatCard
          icon={ShieldAlert}
          label="Likely Fake"
          value={(decision.forged ?? 0).toLocaleString()}
          sub={`of ${stats.total_analyses || 0} analyses`}
          iconClass="text-red-500"
        />
        <StatCard
          icon={Scale}
          label="Average Risk Score"
          value={stats.average_risk_score != null ? stats.average_risk_score.toFixed(1) : '0.0'}
          sub="across all analyses / 100"
          iconClass="text-accent"
        />
      </div>

      {/* --- Charts --- */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card
          title="Document Classification Distribution"
          subtitle="Decision label per completed analysis"
          bodyClassName="p-5"
        >
          {chartData.pie.length ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData.pie}
                    dataKey="value"
                    nameKey="label"
                    cx="50%"
                    cy="50%"
                    innerRadius={52}
                    outerRadius={82}
                    paddingAngle={3}
                    stroke="#111111"
                    strokeWidth={2}
                  >
                    {chartData.pie.map((entry) => (
                      <Cell key={entry.name} fill={PIE_COLORS[entry.name] || '#a1a1aa'} />
                    ))}
                  </Pie>
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    verticalAlign="bottom"
                    iconType="circle"
                    iconSize={8}
                    wrapperStyle={{ fontSize: 12, color: '#a1a1aa' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyChart message="No analyses performed yet." />
          )}
          <div className="mt-3 grid grid-cols-3 gap-2 border-t border-edge pt-3 text-center">
            {['original', 'suspicious', 'forged'].map((key) => (
              <div key={key}>
                <p className="mono text-lg font-semibold" style={{ color: PIE_COLORS[key] }}>
                  {decision[key] ?? 0}
                </p>
                <p className="text-[11px] text-zinc-500">
                  {key === 'original' ? 'Genuine' : key === 'suspicious' ? 'Suspicious' : 'Fake'}
                </p>
              </div>
            ))}
          </div>
        </Card>

        <Card
          title="Risk Score Distribution"
          subtitle="Analyses bucketed into risk bands"
          bodyClassName="p-5"
        >
          {chartData.risk.length ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData.risk} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1b1b1b" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                    axisLine={{ stroke: '#232323' }}
                    tickLine={false}
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fill: '#71717a', fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: '#161616' }} />
                  <Bar dataKey="count" name="Analyses" radius={[3, 3, 0, 0]}>
                    {chartData.risk.map((entry, index) => (
                      <Cell key={index} fill={index === 2 ? '#ef4444' : index === 1 ? '#f59e0b' : '#22c55e'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyChart message="No risk data available." />
          )}
        </Card>

        <Card
          title="Marker Frequency"
          subtitle="Detected markers by severity"
          bodyClassName="p-5"
        >
          {chartData.severity.length ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData.severity} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1b1b1b" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                    axisLine={{ stroke: '#232323' }}
                    tickLine={false}
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fill: '#71717a', fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: '#161616' }} />
                  <Bar dataKey="count" name="Markers" radius={[3, 3, 0, 0]}>
                    {chartData.severity.map((entry) => (
                      <Cell key={entry.key} fill={SEVERITY_COLORS[entry.key] || '#a1a1aa'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyChart message="No markers detected yet." />
          )}
          <div className="mt-3 flex items-center justify-between border-t border-edge pt-3 text-xs text-zinc-500">
            <span>
              {stats.total_markers || 0} total markers across all analyses
            </span>
            <span
              className="mono font-semibold text-red-400"
              title="Critical severity markers"
            >
              {severity.critical || 0} critical
            </span>
          </div>
        </Card>
      </div>

      {/* --- Recent scans --- */}
      <Card
        title="Recent Scans"
        subtitle="Latest document analyses"
        action={
          <Link
            to="/history"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-accent hover:text-accent-deep"
          >
            View all <ArrowRight size={14} />
          </Link>
        }
        bodyClassName="p-0"
      >
        {stats.recent_analyses && stats.recent_analyses.length ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px] text-sm">
              <thead>
                <tr className="border-b border-edge text-left text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  <th className="px-5 py-3">Document</th>
                  <th className="px-5 py-3">Type</th>
                  <th className="px-5 py-3">Risk</th>
                  <th className="px-5 py-3">Decision</th>
                  <th className="px-5 py-3 text-right">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#171717]">
                {stats.recent_analyses.map((item) => (
                  <tr key={item.analysis_id} className="hover:bg-raised/50">
                    <td className="px-5 py-3">
                      <Link
                        to={`/analysis/${item.analysis_id}`}
                        className="flex max-w-xs items-center gap-2 truncate font-medium text-zinc-100 hover:text-accent"
                      >
                        <FileText size={14} className="shrink-0 text-zinc-500" />
                        <span className="truncate">{item.original_filename}</span>
                      </Link>
                    </td>
                    <td className="px-5 py-3 text-zinc-400">
                      <span className="rounded border border-edge bg-carbon px-2 py-0.5 font-mono text-[11px] uppercase">
                        {item.file_type}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <span
                        className="mono inline-flex items-center gap-2 font-medium"
                        style={{ color: riskColor(item.risk_score) }}
                      >
                        {item.risk_score?.toFixed(1)}
                        <span
                          className="h-1.5 w-12 overflow-hidden rounded-full bg-zinc-800"
                          title="Risk bar"
                        >
                          <span
                            className="block h-full rounded-full"
                            style={{
                              width: `${item.risk_score || 0}%`,
                              backgroundColor: riskColor(item.risk_score),
                            }}
                          />
                        </span>
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <DecisionBadge decision={item.decision} size="sm" />
                    </td>
                    <td className="px-5 py-3 text-right text-xs text-zinc-500">
                      {formatDate(item.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="px-5 py-10 text-center">
            <PieIcon size={22} className="mx-auto text-zinc-600" />
            <p className="mt-2 text-sm text-zinc-500">No scans yet. Analyze a document to get started.</p>
          </div>
        )}
      </Card>
    </div>
  )
}

function EmptyChart({ message }) {
  return (
    <div className="flex h-64 items-center justify-center">
      <p className="text-xs text-zinc-600">{message}</p>
    </div>
  )
}