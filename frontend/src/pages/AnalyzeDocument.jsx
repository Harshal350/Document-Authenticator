import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  Loader2,
  XCircle,
  ShieldCheck,
  ArrowRight,
} from 'lucide-react'

import api from '../services/api'
import ErrorState from '../components/ErrorState'
import { formatBytes } from '../utils/format'

const ALLOWED_EXTENSIONS = ['pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']

const PIPELINE_STEPS = [
  { key: 'uploaded', label: 'Uploading' },
  { key: 'extracting', label: 'Extracting' },
  { key: 'analyzing', label: 'Analyzing' },
  { key: 'markers', label: 'Detecting Markers' },
  { key: 'ml', label: 'Running ML' },
  { key: 'report', label: 'Generating Report' },
]

export default function AnalyzeDocument() {
  const navigate = useNavigate()
  const inputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [phase, setPhase] = useState('idle') // idle | uploading | analyzing | done | error
  const [activeStep, setActiveStep] = useState(0)
  const [uploadResult, setUploadResult] = useState(null)

  const reset = () => {
    setFile(null)
    setError(null)
    setUploadResult(null)
    setActiveStep(0)
    setPhase('idle')
  }

  const validateAndSelect = (candidate) => {
    if (!candidate) return
    const ext = candidate.name.split('.').pop()?.toLowerCase()
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(
        `Unsupported file type ".${ext || '?'}". Allowed: ${ALLOWED_EXTENSIONS.map((e) => `.${e}`).join(', ')}`,
      )
      return
    }
    setError(null)
    setFile(candidate)
  }

  const onDrop = useCallback(
    (event) => {
      event.preventDefault()
      setDragOver(false)
      validateAndSelect(event.dataTransfer.files?.[0])
    },
    [],
  )

  const onDragOver = (event) => {
    event.preventDefault()
    setDragOver(true)
  }

  const onDragLeave = (event) => {
    event.preventDefault()
    setDragOver(false)
  }

  const pickFile = (event) => {
    validateAndSelect(event.target.files?.[0])
    if (inputRef.current) inputRef.current.value = ''
  }

  const runAnalysis = async () => {
    if (!file || phase === 'uploading' || phase === 'analyzing') return

    setError(null)
    setPhase('uploading')
    setActiveStep(0)

    let documentId = null
    try {
      const upload = await api.uploadDocument(file)
      documentId = upload.document_id
      setUploadResult(upload)

      setPhase('analyzing')
      setActiveStep(1)
      const stepTimer = window.setInterval(() => {
        setActiveStep((step) => {
          if (step < PIPELINE_STEPS.length - 1) return step + 1
          return step
        })
      }, 1500)

      const analysis = await api.analyzeDocument(documentId)
      window.clearInterval(stepTimer)
      setActiveStep(PIPELINE_STEPS.length - 1)
      setPhase('done')
      setUploadResult({ ...upload, analysis_id: analysis.id })
      navigate(`/analysis/${analysis.id}`, { state: { fresh: true } })
    } catch (err) {
      setPhase('error')
      setError(err.message)
    }
  }

  useEffect(() => {
    return () => {
      /* noop cleanup on unmount */
    }
  }, [])

  if (phase === 'error' && error) {
    return (
      <div className="space-y-6">
        <Header />
        <ErrorState message={error} onRetry={runAnalysis} />
      </div>
    )
  }

  const busy = phase === 'uploading' || phase === 'analyzing'

  return (
    <div className="space-y-6">
      <Header />

      {/* --- Upload area --- */}
      {!file && (
        <div
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`cursor-pointer rounded-lg border-2 border-dashed p-12 text-center transition-colors focus:outline-none focus:ring-1 focus:ring-accent/50 ${
            dragOver
              ? 'border-accent bg-accent/10'
              : 'border-zinc-700 bg-surface hover:border-accent/60 hover:bg-carbon'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            onChange={pickFile}
            accept={ALLOWED_EXTENSIONS.map((e) => `.${e}`).join(',')}
          />
          <span
            className={`mx-auto flex h-16 w-16 items-center justify-center rounded-full border ${
              dragOver ? 'border-accent text-accent' : 'border-edge bg-raised text-zinc-400'
            }`}
          >
            <UploadCloud size={30} strokeWidth={1.6} />
          </span>
          <p className="mt-5 text-sm font-semibold text-zinc-100">
            {dragOver ? 'Release to upload' : 'Drag & drop a document here'}
          </p>
          <p className="mt-1.5 text-xs text-zinc-500">
            or <span className="font-medium text-accent">browse files</span> from your device
          </p>
          {error && (
            <p className="mx-auto mt-4 max-w-lg rounded-md border border-red-900/60 bg-red-950/30 px-4 py-2 text-xs text-red-400">
              {error}
            </p>
          )}
        </div>
      )}

      {/* --- Supported formats --- */}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <span className="text-[11px] uppercase tracking-wider text-zinc-600">
          Supported formats
        </span>
        {ALLOWED_EXTENSIONS.map((ext) => (
          <span
            key={ext}
            className="rounded border border-edge bg-carbon px-2 py-0.5 font-mono text-[11px] uppercase text-zinc-400"
          >
            {ext}
          </span>
        ))}
      </div>

      {/* --- File info + actions --- */}
      {file && (
        <div className="panel p-5">
          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
            <div className="flex min-w-0 items-center gap-4">
              <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-md border border-accent/30 bg-accent/10 text-accent">
                <FileText size={22} strokeWidth={1.8} />
              </span>
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-zinc-100">{file.name}</p>
                <p className="mono mt-0.5 text-xs text-zinc-500">
                  {formatBytes(file.size)} · .{file.name.split('.').pop()?.toLowerCase()}
                </p>
              </div>
            </div>
            {phase === 'done' && (
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-accent">
                <CheckCircle2 size={15} /> Analysis complete
              </span>
            )}
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              onClick={runAnalysis}
              disabled={busy}
              className="btn btn-accent"
            >
              {busy ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  {phase === 'uploading' ? 'Uploading…' : 'Analyzing…'}
                </>
              ) : (
                <>
                  <ShieldCheck size={16} />
                  Analyze Document
                </>
              )}
            </button>
            <button onClick={reset} disabled={busy} className="btn btn-ghost">
              Clear
            </button>
          </div>
        </div>
      )}

      {error && phase === 'idle' && (
        <div className="flex items-center gap-2 rounded-lg border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-400">
          <XCircle size={16} className="shrink-0" />
          <span className="min-w-0">{error}</span>
        </div>
      )}

      {/* --- Pipeline steps --- */}
      {busy && (
        <div className="panel p-6">
          <h3 className="text-sm font-semibold text-zinc-100">Detection Pipeline</h3>
          <p className="mt-1 text-xs text-zinc-500">
            Executing forensic extraction, marker detection and ML risk scoring.
          </p>
          <div className="mt-6 flex flex-col gap-1">
            {PIPELINE_STEPS.map((step, index) => (
              <div
                key={step.key}
                className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                  index < activeStep
                    ? 'text-zinc-400'
                    : index === activeStep
                      ? 'bg-accent/10 text-zinc-100'
                      : 'text-zinc-600'
                }`}
              >
                {index < activeStep ? (
                  <CheckCircle2 size={16} className="text-accent" />
                ) : index === activeStep ? (
                  <Loader2 size={16} className="animate-spin text-accent" />
                ) : (
                  <span className="flex h-4 w-4 items-center justify-center">
                    <span className="block h-1.5 w-1.5 rounded-full bg-zinc-700" />
                  </span>
                )}
                <span className="font-medium">
                  {index + 1}. {step.label}
                </span>
                {index === activeStep && (
                  <span className="ml-auto text-[11px] uppercase tracking-wider text-accent">
                    running
                  </span>
                )}
              </div>
            ))}
          </div>
          <div className="mt-5 h-1.5 w-full overflow-hidden rounded-full bg-carbon">
            <div
              className="h-full rounded-full bg-accent transition-all duration-700"
              style={{ width: `${((activeStep + 1) / PIPELINE_STEPS.length) * 100}%` }}
            />
          </div>
        </div>
      )}

      {phase === 'done' && uploadResult && (
        <div className="panel flex items-center justify-between gap-4 border-accent/30 bg-accent/5 p-5">
          <div className="flex items-center gap-3">
            <ShieldCheck size={22} className="text-accent" />
            <div>
              <p className="text-sm font-semibold text-zinc-100">Analysis completed successfully</p>
              <p className="text-xs text-zinc-500">
                Document #{uploadResult.document_id} — {uploadResult.filename}
              </p>
            </div>
          </div>
          <button className="btn btn-accent" onClick={() => navigate(`/analysis/${uploadResult.analysis_id}`)}>
            View Results <ArrowRight size={15} />
          </button>
        </div>
      )}
    </div>
  )
}

function Header() {
  return (
    <div>
      <h1 className="text-xl font-semibold tracking-tight text-zinc-50">Analyze Document</h1>
      <p className="mt-1 text-sm text-zinc-500">
        Upload a document and run the full forensic verification pipeline.
      </p>
    </div>
  )
}