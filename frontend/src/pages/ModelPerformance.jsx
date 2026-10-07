import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  RefreshCw,
  GraduationCap,
  Loader2,
  Info,
  CheckCircle2,
  Database,
  UploadCloud,
  FileSpreadsheet,
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from 'recharts'

import api from '../services/api'
import Card from '../components/Card'
import ChartTooltip from '../components/ChartTooltip'
import ErrorState from '../components/ErrorState'
import Spinner from '../components/Spinner'

const MODEL_ACCENT = ['#22c55e', '#4ade80', '#e2e8f0']

function buildConfusionMatrix(model) {
  // Estimated from reported metrics on a balanced held-out test set.
  const tp = Math.max(0, Math.min(1, Number(model.recall) || 0))
  const precision = Math.max(0.001, Math.min(1, Number(model.precision) || 0))
  const fp = (tp * (1 - precision)) / precision
  const tn = Math.max(0, Math.min(1, 2 * (Number(model.accuracy) || 0) - tp))
  const fn = Math.max(0, 1 - tp)
  return [
    { label: 'TP', value: tp, note: 'True Positives' },
    { label: 'FP', value: fn, note: 'False Positives' },
    { label: 'FN', value: fp, note: 'False Negatives' },
    { label: 'TN', value: tn, note: 'True Negatives' },
  ]
}

export default function ModelPerformance() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [training, setTraining] = useState(false)
  const [trainingStage, setTrainingStage] = useState(0)
  const [trainResult, setTrainResult] = useState(null)
  const [trainError, setTrainError] = useState(null)

  const [datasetInfo, setDatasetInfo] = useState(null)
  const [datasetPath, setDatasetPath] = useState(null)
  const [uploadingDataset, setUploadingDataset] = useState(false)
  const [uploadMsg, setUploadMsg] = useState(null)
  const [uploadErr, setUploadErr] = useState(null)

  const loadDataset = useCallback(async (path) => {
    try {
      const info = await api.getDatasetInfo(path)
      setDatasetInfo(info)
    } catch {
      // Optional fallback if dataset is unavailable
    }
  }, [])

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getModelPerformance()
      setModels(data.models || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const train = async () => {
    setTraining(true)
    setTrainResult(null)
    setTrainError(null)
    setTrainingStage(1)

    const timer = window.setInterval(() => {
      setTrainingStage((stage) => Math.min(5, stage + 1))
    }, 2500)

    try {
      const result = await api.trainModels(datasetPath)
      setTrainResult(result)
      setTrainingStage(5)
      await load()
      await loadDataset(datasetPath)
    } catch (err) {
      setTrainError(err.message)
    } finally {
      window.clearInterval(timer)
      setTraining(false)
    }
  }

  const handleDatasetUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadingDataset(true)
    setUploadMsg(null)
    setUploadErr(null)
    try {
      const res = await api.uploadDataset(file)
      setDatasetPath(res.dataset_path)
      setUploadMsg(`Successfully uploaded "${res.dataset_name}" (${res.total_rows} rows). Ready for training!`)
      await loadDataset(res.dataset_path)
    } catch (err) {
      setUploadErr(err.message || 'Dataset upload failed')
    } finally {
      setUploadingDataset(false)
      e.target.value = ''
    }
  }

  useEffect(() => {
    load()
    loadDataset()
  }, [load, loadDataset])

  const chartData = useMemo(
    () =>
      models.map((model) => ({
        name: model.model_name,
        Accuracy: Number(model.accuracy) || 0,
        Precision: Number(model.precision) || 0,
        Recall: Number(model.recall) || 0,
        F1: Number(model.f1) || 0,
      })),
    [models],
  )

  const TRAINING_STAGES = [
    'Loading labelled dataset',
    'Extracting features',
    'Training classifiers',
    'Evaluating on test set',
    'Persisting metrics',
  ]

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-zinc-50">
            Model Performance
          </h1>
          <p className="mt-1 flex items-center gap-1.5 text-sm text-zinc-500">
            <GraduationCap size={15} className="text-accent" />
            Academic comparison of the trained detection classifiers.
          </p>
        </div>
        <button onClick={train} disabled={training || loading} className="btn btn-accent">
          {training ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Training in progress…
            </>
          ) : (
            <>
              <RefreshCw size={16} />
              Train Models
            </>
          )}
        </button>
      </div>

      <div className="flex items-start gap-2 rounded-lg border border-edge bg-surface px-4 py-3 text-xs text-zinc-400">
        <Info size={14} className="mt-0.5 shrink-0 text-accent" />
        <p>
          Model performance is measured on the held-out test dataset. Metrics are
          recorded after each training run and represent balanced-class
          evaluation on documents never seen during training.
        </p>
      </div>

      {/* --- Training progress --- */}
      {(training || trainResult || trainError) && (
        <div className="panel p-5">
          {training ? (
            <>
              <div className="flex items-center gap-3">
                <Loader2 size={18} className="animate-spin text-accent" />
                <p className="text-sm font-semibold text-zinc-100">
                  Retraining detection models…
                </p>
              </div>
              <div className="mt-4 space-y-2">
                {TRAINING_STAGES.map((stage, index) => (
                  <div
                    key={stage}
                    className={`flex items-center gap-2.5 text-xs ${
                      index < trainingStage ? 'text-zinc-400' : 'text-zinc-600'
                    }`}
                  >
                    {index < trainingStage ? (
                      <CheckCircle2 size={14} className="text-accent" />
                    ) : (
                      <span className="flex h-3.5 w-3.5 items-center justify-center">
                        <span className="block h-1.5 w-1.5 rounded-full bg-zinc-700" />
                      </span>
                    )}
                    {stage}
                    {index === trainingStage - 1 && (
                      <span className="text-[10px] uppercase tracking-wider text-accent">
                        running
                      </span>
                    )}
                  </div>
                ))}
              </div>
              <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-carbon">
                <div
                  className="h-full rounded-full bg-accent transition-all duration-700"
                  style={{ width: `${(trainingStage / TRAINING_STAGES.length) * 100}%` }}
                />
              </div>
            </>
          ) : trainError ? (
            <div className="rounded-md border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-400">
              Training failed: {trainError}
            </div>
          ) : (
            trainResult && (
              <div className="flex items-center gap-3">
                <CheckCircle2 size={18} className="text-accent" />
                <p className="text-sm text-zinc-300">
                  {trainResult.message} —{' '}
                  <span className="mono text-zinc-100">
                    {trainResult.metrics_saved || 0}
                  </span>{' '}
                  metric set(s) saved
                  {trainResult.charts_generated ? ' · comparison charts generated' : ''}.
                </p>
              </div>
            )
          )}
        </div>
      )}

      {loading ? (
        <div className="flex h-48 items-center justify-center text-zinc-500">
          <div className="flex flex-col items-center gap-3">
            <Spinner size={28} />
            <span className="text-sm">Loading model metrics…</span>
          </div>
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={load} />
      ) : (
        <>
          {/* --- Dataset Management & Retraining (§18) --- */}
          <Card
            title="Training Dataset & Corpus Management"
            subtitle="Inspect training corpus balance, sample splits, and upload custom datasets"
            bodyClassName="p-5"
          >
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              <div className="space-y-4 lg:col-span-7">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Database size={16} className="text-accent" />
                    <span className="font-mono text-sm font-semibold text-zinc-100">
                      {datasetInfo?.dataset_name || 'documents.csv'}
                    </span>
                    {datasetInfo?.is_demo_data && (
                      <span className="rounded border border-accent/40 bg-accent/10 px-2 py-0.5 font-mono text-[10px] text-accent uppercase">
                        Standard Corpus
                      </span>
                    )}
                    {datasetPath && !datasetInfo?.is_demo_data && (
                      <span className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 font-mono text-[10px] text-amber-400 uppercase">
                        Custom Dataset
                      </span>
                    )}
                  </div>
                  <span className="font-mono text-xs text-zinc-500">
                    80/20 Stratified Split
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-md border border-edge bg-carbon p-3">
                    <span className="text-[11px] text-zinc-500">Total Samples</span>
                    <p className="mono mt-1 text-lg font-semibold text-zinc-100">
                      {datasetInfo?.total_rows ?? '—'}
                    </p>
                  </div>
                  <div className="rounded-md border border-edge bg-carbon p-3">
                    <span className="text-[11px] text-zinc-500">Training (80%)</span>
                    <p className="mono mt-1 text-lg font-semibold text-zinc-100">
                      {datasetInfo?.training_samples ?? '—'}
                    </p>
                  </div>
                  <div className="rounded-md border border-edge bg-carbon p-3">
                    <span className="text-[11px] text-zinc-500">Held-Out Test (20%)</span>
                    <p className="mono mt-1 text-lg font-semibold text-zinc-100">
                      {datasetInfo?.testing_samples ?? '—'}
                    </p>
                  </div>
                  <div className="rounded-md border border-edge bg-carbon p-3">
                    <span className="text-[11px] text-zinc-500">Columns</span>
                    <p className="mono mt-1 text-lg font-semibold text-zinc-100">
                      {datasetInfo?.columns?.length ?? '—'}
                    </p>
                  </div>
                </div>

                <div>
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                    Class Distribution & Supervision Target
                  </p>
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="flex items-center gap-2 rounded-md border border-edge bg-carbon px-3 py-1.5 text-xs">
                      <span className="h-2 w-2 rounded-full bg-accent" />
                      <span className="text-zinc-400">Genuine (Label 0):</span>
                      <span className="mono font-semibold text-zinc-100">
                        {datasetInfo?.class_distribution?.['Genuine (0)'] ?? '—'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 rounded-md border border-edge bg-carbon px-3 py-1.5 text-xs">
                      <span className="h-2 w-2 rounded-full bg-red-400" />
                      <span className="text-zinc-400">Fake / Tampered (Label 1):</span>
                      <span className="mono font-semibold text-zinc-100">
                        {datasetInfo?.class_distribution?.['Fake (1)'] ?? '—'}
                      </span>
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-zinc-500">
                    Feature text column: <span className="mono text-zinc-300">{datasetInfo?.text_column || 'text'}</span> · Target label:{' '}
                    <span className="mono text-zinc-300">{datasetInfo?.label_column || 'label'}</span>
                  </p>
                </div>
              </div>

              {/* Upload area */}
              <div className="flex flex-col justify-between rounded-lg border border-edge/80 bg-carbon p-4 lg:col-span-5">
                <div>
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                    <UploadCloud size={15} className="text-accent" />
                    Upload Custom Training Dataset
                  </div>
                  <p className="mt-1 text-xs text-zinc-500">
                    Supply a custom CSV file with document text and binary supervision labels (0 = Genuine, 1 = Fake).
                  </p>

                  <div className="mt-4">
                    <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-zinc-700 bg-surface/50 p-4 transition-colors hover:border-accent hover:bg-surface">
                      <FileSpreadsheet size={24} className="mb-1 text-zinc-400" />
                      <span className="text-xs font-medium text-zinc-200">
                        {uploadingDataset ? 'Uploading dataset…' : 'Choose CSV dataset'}
                      </span>
                      <span className="mt-0.5 text-[10px] text-zinc-500">.csv format only</span>
                      <input
                        type="file"
                        accept=".csv"
                        className="hidden"
                        disabled={uploadingDataset || training}
                        onChange={handleDatasetUpload}
                      />
                    </label>
                  </div>

                  {uploadMsg && (
                    <div className="mt-3 rounded border border-accent/40 bg-accent/10 px-3 py-2 text-xs text-accent">
                      {uploadMsg}
                    </div>
                  )}
                  {uploadErr && (
                    <div className="mt-3 rounded border border-red-900/60 bg-red-950/30 px-3 py-2 text-xs text-red-400">
                      {uploadErr}
                    </div>
                  )}
                </div>

                <div className="mt-4 flex items-center justify-between border-t border-edge/60 pt-3">
                  <span className="text-[11px] text-zinc-500">
                    {datasetPath ? 'Custom dataset staged' : 'Standard corpus loaded'}
                  </span>
                  {datasetPath && (
                    <button
                      type="button"
                      onClick={() => {
                        setDatasetPath(null)
                        loadDataset()
                        setUploadMsg(null)
                      }}
                      className="text-[11px] text-zinc-400 underline hover:text-zinc-200"
                    >
                      Reset to default
                    </button>
                  )}
                </div>
              </div>
            </div>
          </Card>

          {/* --- Metrics table --- */}
          <Card
            title="Evaluation Metrics"
            subtitle={`${models.length} model(s) evaluated on the held-out test dataset`}
            bodyClassName="p-0"
          >
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead>
                  <tr className="border-b border-edge text-left text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                    <th className="px-5 py-3">Model</th>
                    <th className="px-5 py-3">Accuracy</th>
                    <th className="px-5 py-3">Precision</th>
                    <th className="px-5 py-3">Recall</th>
                    <th className="px-5 py-3">F1</th>
                    <th className="px-5 py-3">ROC-AUC</th>
                    <th className="px-5 py-3 text-right">Training Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#171717]">
                  {models.map((model) => (
                    <tr key={model.id ?? model.model_name} className="hover:bg-raised/50">
                      <td className="px-5 py-3.5">
                        <span className="flex items-center gap-2.5 font-medium text-zinc-100">
                          <Activity size={15} className="text-accent shrink-0" />
                          {model.model_name}
                        </span>
                      </td>
                      <td className="px-5 py-3.5"><Metric value={model.accuracy} /></td>
                      <td className="px-5 py-3.5"><Metric value={model.precision} /></td>
                      <td className="px-5 py-3.5"><Metric value={model.recall} /></td>
                      <td className="px-5 py-3.5"><Metric value={model.f1} /></td>
                      <td className="px-5 py-3.5"><Metric value={model.roc_auc} /></td>
                      <td className="px-5 py-3.5 text-right">
                        {model.training_time != null ? (
                          <span className="mono text-xs text-zinc-400">
                            {Number(model.training_time).toFixed(2)}s
                          </span>
                        ) : (
                          <span className="text-zinc-600">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* --- Charts --- */}
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card
              title="Accuracy Comparison"
              subtitle="Per-model test-set accuracy"
              bodyClassName="p-5"
            >
              {models.length ? (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
                      <CartesianGrid stroke="#1b1b1b" vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: '#71717a', fontSize: 11 }}
                        axisLine={{ stroke: '#232323' }}
                        tickLine={false}
                        interval={0}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={(v) => `${Math.round(v * 100)}%`}
                        tick={{ fill: '#71717a', fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip content={<ChartTooltip valueFormatter={(v) => `${(v * 100).toFixed(2)}%`} />} cursor={{ fill: '#161616' }} />
                      <Bar dataKey="Accuracy" name="Accuracy" radius={[3, 3, 0, 0]}>
                        {chartData.map((_, index) => (
                          <Cell key={index} fill={MODEL_ACCENT[index % MODEL_ACCENT.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState />
              )}
            </Card>

            <Card
              title="Precision · Recall · F1"
              subtitle="Grouped comparison across each metric"
              bodyClassName="p-5"
            >
              {models.length ? (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
                      <CartesianGrid stroke="#1b1b1b" vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: '#71717a', fontSize: 11 }}
                        axisLine={{ stroke: '#232323' }}
                        tickLine={false}
                        interval={0}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={(v) => `${Math.round(v * 100)}%`}
                        tick={{ fill: '#71717a', fontSize: 12 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip content={<ChartTooltip valueFormatter={(v) => `${(v * 100).toFixed(2)}%`} />} cursor={{ fill: '#161616' }} />
                      <Legend
                        iconType="circle"
                        iconSize={8}
                        wrapperStyle={{ fontSize: 12, color: '#a1a1aa' }}
                      />
                      <Bar dataKey="Precision" fill="#22c55e" radius={[2, 2, 0, 0]} maxBarSize={22} />
                      <Bar dataKey="Recall" fill="#4ade80" radius={[2, 2, 0, 0]} maxBarSize={22} />
                      <Bar dataKey="F1" fill="#e2e8f0" radius={[2, 2, 0, 0]} maxBarSize={22} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState />
              )}
            </Card>
          </div>

          {/* --- Confusion matrices --- */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {models.map((model) => {
              const matrix = buildConfusionMatrix(model)
              return (
                <Card
                  key={model.id ?? model.model_name}
                  title={model.model_name}
                  subtitle="Hold-out confusion matrix*"
                  bodyClassName="p-5"
                >
                  <div className="grid grid-cols-2 gap-2.5">
                    <MatrixCell label="Predicted Positive" value={matrix[0].value} accent="#22c55e" />
                    <MatrixCell label="Predicted Negative" value={matrix[3].value} accent="#3b82f6" />
                    <MatrixCell label="False Positive" value={matrix[1].value} accent="#f59e0b" />
                    <MatrixCell label="False Negative" value={matrix[2].value} accent="#ef4444" />
                  </div>
                  <div className="mt-2.5 flex items-center gap-2">
                    <span className="h-1.5 w-full rounded-full bg-carbon">
                      <span
                        className="block h-full rounded-full"
                        style={{ width: `${(Number(model.roc_auc) || 0) * 100}%`, backgroundColor: '#22c55e' }}
                      />
                    </span>
                    <span className="mono shrink-0 text-[11px] text-zinc-400">
                      AUC {(Number(model.roc_auc) || 0).toFixed(3)}
                    </span>
                  </div>
                </Card>
              )
            })}
          </div>

          <p className="px-1 text-[11px] leading-relaxed text-zinc-600">
            * Confusion-matrix cell rates are estimated from the reported
            accuracy, precision and recall on a balanced held-out test set. TP /
            TN / FP / FN cell values are normalized to the population rate.
          </p>
        </>
      )}
    </div>
  )
}

function Metric({ value }) {
  const number = Number(value)
  if (Number.isNaN(number)) return <span className="text-zinc-600">—</span>
  return (
    <span className="mono inline-flex items-center gap-2 font-medium text-zinc-100">
      {(number * 100).toFixed(1)}%
      <span className="h-1 w-12 overflow-hidden rounded-full bg-zinc-800">
        <span
          className="block h-full rounded-full bg-accent"
          style={{ width: `${Math.min(100, number * 100)}%` }}
        />
      </span>
    </span>
  )
}

function MatrixCell({ label, value, accent }) {
  return (
    <div className="rounded-md border border-edge bg-carbon px-3 py-3 text-center">
      <p className="text-[10px] uppercase tracking-wider text-zinc-600">{label}</p>
      <p className="mono mt-1 text-xl font-semibold" style={{ color: accent }}>
        {(value * 100).toFixed(1)}%
      </p>
      <p className="text-[10px] text-zinc-600">of population</p>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex h-64 items-center justify-center">
      <p className="text-xs text-zinc-600">
        No models trained yet. Run "Train Models" to populate metrics.
      </p>
    </div>
  )
}