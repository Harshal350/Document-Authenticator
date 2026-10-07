import { useEffect, useState } from 'react'
import { Routes, Route, Navigate, NavLink } from 'react-router-dom'
import { ShieldCheck } from 'lucide-react'

import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import AnalyzeDocument from './pages/AnalyzeDocument'
import AnalysisResult from './pages/AnalysisResult'
import ModelPerformance from './pages/ModelPerformance'
import History from './pages/History'
import api from './services/api'

const MOBILE_NAV = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/analyze', label: 'Analyze' },
  { to: '/models', label: 'Models' },
  { to: '/history', label: 'History' },
]

export default function App() {
  const [health, setHealth] = useState({ loading: true, status: null, detail: null })

  useEffect(() => {
    let active = true
    const check = async () => {
      try {
        const result = await api.getHealth()
        if (active) setHealth({ loading: false, status: result.status, detail: result.version })
      } catch {
        if (active) setHealth({ loading: false, status: 'offline', detail: null })
      }
    }
    check()
    return () => {
      active = false
    }
  }, [])

  return (
    <div className="flex h-screen bg-night text-zinc-200">
      <Sidebar />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* --- Top header --- */}
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-edge bg-carbon px-5">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md border border-accent/40 bg-accent/10 text-accent">
              <ShieldCheck size={20} strokeWidth={1.8} />
            </span>
            <div>
              <p className="text-sm font-bold leading-tight tracking-tight text-zinc-50">
                DocuGuard
              </p>
              <p className="text-[11px] leading-tight text-zinc-500">
                AI-Powered Document Verification
              </p>
            </div>
          </div>

          {/* API status */}
          <div className="flex items-center gap-2.5">
            <span
              className={
                health.loading
                  ? 'h-2 w-2 animate-pulse rounded-full bg-zinc-600'
                  : health.status === 'healthy'
                    ? 'h-2 w-2 rounded-full bg-accent shadow-[0_0_8px_rgba(34,197,94,0.5)]'
                    : 'h-2 w-2 rounded-full bg-red-500'
              }
            />
            <span className="hidden text-xs text-zinc-500 sm:inline">
              {health.loading
                ? 'Checking API…'
                : health.status === 'healthy'
                  ? `API online · v${health.detail}`
                  : 'API offline · localhost:8000'}
            </span>
          </div>
        </header>

        {/* --- Mobile nav --- */}
        <nav className="flex items-center gap-1 overflow-x-auto border-b border-edge bg-carbon px-3 py-2 md:hidden">
          {MOBILE_NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-accent/10 text-accent border border-accent/30'
                    : 'text-zinc-400 border border-transparent hover:text-zinc-100'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* --- Main content --- */}
        <main className="min-h-0 flex-1 overflow-y-auto bg-night">
          <div className="mx-auto max-w-7xl px-5 py-7 lg:px-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analyze" element={<AnalyzeDocument />} />
              <Route path="/analysis/:analysisId" element={<AnalysisResult />} />
              <Route path="/models" element={<ModelPerformance />} />
              <Route path="/history" element={<History />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  )
}