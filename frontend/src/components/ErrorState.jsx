import { AlertTriangle, RefreshCw } from 'lucide-react'

export default function ErrorState({ message, onRetry }) {
  return (
    <div className="panel flex flex-col items-center justify-center px-6 py-16 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-full border border-red-900/60 bg-red-950/30 text-red-400">
        <AlertTriangle size={22} strokeWidth={1.8} />
      </span>
      <h3 className="mt-4 text-sm font-semibold text-zinc-100">
        Could not reach the DocuGuard API
      </h3>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-zinc-500">{message}</p>
      {onRetry && (
        <button className="btn btn-ghost mt-5" onClick={onRetry}>
          <RefreshCw size={15} />
          Retry
        </button>
      )}
    </div>
  )
}