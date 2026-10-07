export default function StatCard({ icon: Icon, label, value, sub, iconClass }) {
  return (
    <div className="panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
            {label}
          </p>
          <p className="mono mt-2 truncate text-3xl font-semibold text-zinc-50">
            {value}
          </p>
          {sub && <p className="mt-1.5 text-xs text-zinc-500">{sub}</p>}
        </div>
        {Icon && (
          <span
            className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-edge ${iconClass || 'text-accent'}`}
          >
            <Icon size={18} strokeWidth={1.8} />
          </span>
        )}
      </div>
    </div>
  )
}