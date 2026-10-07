const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'
).replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request(path, { method = 'GET', body = null, timeout = 60000 } = {}) {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeout)

  try {
    const config = { method, signal: controller.signal }
    config.headers = {}

    if (body instanceof FormData) {
      config.body = body
    } else if (body !== null && body !== undefined) {
      config.body = JSON.stringify(body)
      config.headers['Content-Type'] = 'application/json'
    }

    const response = await fetch(`${API_BASE_URL}${path}`, config)

    const contentType = response.headers.get('content-type') || ''
    let data = null
    if (contentType.includes('application/json')) {
      try {
        data = await response.json()
      } catch {
        data = null
      }
    }

    if (!response.ok) {
      const detail = data?.detail
      let message
      if (typeof detail === 'string') {
        message = detail
      } else if (Array.isArray(detail)) {
        message = detail.map((entry) => entry?.msg || String(entry)).join('; ')
      } else if (data?.message) {
        message = data.message
      } else {
        message = `Request failed with status ${response.status}`
      }
      throw new ApiError(message, response.status, detail)
    }

    return data
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error?.name === 'AbortError') {
      throw new ApiError(
        'Request timed out. The backend may be busy or unavailable.',
        408,
        null,
      )
    }
    throw new ApiError(
      `Network error — could not reach the DocuGuard API at ${API_BASE_URL}. Is the backend running?`,
      0,
      null,
    )
  } finally {
    window.clearTimeout(timer)
  }
}

/**
 * DocuGuard API service. Talks to the FastAPI backend under
 * http://localhost:8000/api/v1 (override with VITE_API_BASE_URL).
 */
const api = {
  /** Upload a single document (multipart). Returns { document_id, filename }. */
  uploadDocument(file) {
    const form = new FormData()
    form.append('file', file)
    return request('/upload', { method: 'POST', body: form, timeout: 180000 })
  },

  /** Run the full forensic pipeline on an uploaded document. */
  analyzeDocument(documentId) {
    return request(`/analyze/${documentId}`, { method: 'POST', timeout: 600000 })
  },

  /** Return the complete persisted result of a previous analysis. */
  getResults(analysisId) {
    return request(`/results/${analysisId}`)
  },

  /** Paginated, filterable analysis history. */
  getHistory(params = {}) {
    const { skip = 0, limit = 20, decision, search } = params
    const query = new URLSearchParams()
    query.set('skip', String(skip))
    query.set('limit', String(limit))
    if (decision) query.set('decision', decision)
    if (search && search.trim()) query.set('search', search.trim())
    return request(`/history?${query.toString()}`)
  },

  /** Aggregated statistics for the dashboard. */
  getDashboardStats() {
    return request('/dashboard')
  },

  /** Persisted evaluation metrics for every trained model. */
  getModelPerformance() {
    return request('/models/performance')
  },

  /** Trigger model retraining + evaluation. */
  trainModels() {
    return request('/models/train', { method: 'POST', timeout: 900000 })
  },

  /** Generate a PDF forensic report for an analysis. */
  generateReport(analysisId) {
    return request(`/report/${analysisId}`, { timeout: 180000 })
  },

  /** Delete an analysis and its related rows. */
  deleteHistory(analysisId) {
    return request(`/history/${analysisId}`, { method: 'DELETE' })
  },

  /** Basic service health check. */
  getHealth() {
    return request('/health', { timeout: 8000 })
  },
}

export default api