export default function Card({ title, subtitle, action, children, className = '', bodyClassName = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || action) && (
        <div className="panel-header">
          <div>
            {title && (
              <h2 className="text-sm font-semibold tracking-wide text-zinc-100">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>
            )}
          </div>
          {action}
        </div>
      )}
      <div className={bodyClassName || 'p-5'}>{children}</div>
    </section>
  )
}