import { NavLink } from 'react-router-dom'
import { LayoutDashboard, FileSearch, Activity, Clock } from 'lucide-react'

export default function Sidebar() {
  const linkClasses = ({ isActive }) =>
    `group flex items-center gap-3 rounded-md px-3.5 py-2.5 text-sm font-medium transition-colors ${
      isActive
        ? 'bg-accent/10 text-accent border-l-2 border-accent'
        : 'text-zinc-400 hover:text-zinc-100 hover:bg-raised'
    }`

  const navItems = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
    { to: '/analyze', label: 'Analyze Document', icon: FileSearch },
    { to: '/models', label: 'Model Performance', icon: Activity },
    { to: '/history', label: 'History', icon: Clock },
  ]

  return (
    <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-edge bg-carbon">
      <div className="flex flex-col gap-1 px-4 py-6">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={linkClasses}
            >
              {({ isActive }) => (
                <>
                  <Icon
                    size={18}
                    strokeWidth={1.8}
                    className={
                      isActive
                        ? 'text-accent'
                        : 'text-zinc-500 group-hover:text-zinc-300'
                    }
                  />
                  <span>{item.label}</span>
                </>
              )}
            </NavLink>
          )
        })}
      </div>

      <div className="mt-auto px-4 py-5">
        <div className="rounded-md border border-edge bg-surface p-3.5">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
            <span className="text-xs font-semibold text-zinc-200">DocuGuard Engine</span>
          </div>
          <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-500">
            Forensic analysis pipeline v1 · held-out test evaluation
          </p>
        </div>
      </div>
    </aside>
  )
}