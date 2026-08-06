import service from './index'

export const listC0Scenarios = () => {
  return service.get('/api/finance/c0/scenarios')
}

export const prepareC0 = (scenarioId, runMode = 'single') => {
  const payload = { run_mode: runMode }
  if (runMode === 'single') payload.scenario_ids = [scenarioId]
  return service.post('/api/finance/c0/prepare', payload)
}

export const runC0 = (runId, dryRun = false, background = false) => {
  return service.post('/api/finance/c0/run', {
    run_id: runId,
    dry_run: dryRun,
    background
  }, {
    // A real C0 run contains 20 sequential model calls.
    timeout: dryRun ? 300000 : 0
  })
}

export const getC0Status = (runId) => {
  return service.get(`/api/finance/c0/${runId}`)
}

export const getC0Preview = (runId) => {
  return service.get(`/api/finance/c0/${runId}/preview`)
}

export const getC0Predictions = (runId) => {
  return service.get(`/api/finance/c0/${runId}/predictions`)
}

export const getC0Outcome = (runId) => {
  return service.get(`/api/finance/c0/${runId}/outcome`)
}

export const getC0CsvDownloadUrl = (runId, kind) => {
  const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001').replace(/\/$/, '')
  return `${baseUrl}/api/finance/c0/${encodeURIComponent(runId)}/csv/${encodeURIComponent(kind)}`
}

export const prepareS1Reddit = ({ scenarioId, projectId = '', graphId = '', sourceMode = 'auto', socialRounds = 6 }) => {
  const payload = { scenario_id: scenarioId, source_mode: sourceMode, social_rounds: socialRounds }
  if (projectId.trim()) payload.project_id = projectId.trim()
  if (graphId.trim()) payload.graph_id = graphId.trim()
  return service.post('/api/finance/s1/reddit/prepare', payload, { timeout: 60000 })
}

export const getS1ScenarioSeed = (scenarioId) => {
  return service.get(`/api/finance/s1/reddit/scenarios/${encodeURIComponent(scenarioId)}/seed`)
}

export const runS1Reddit = (runId) => {
  return service.post('/api/finance/s1/reddit/run', { run_id: runId }, { timeout: 30000 })
}

export const updateS1RedditSettings = (runId, socialRounds) => {
  return service.patch(`/api/finance/s1/reddit/${encodeURIComponent(runId)}/settings`, {
    social_rounds: socialRounds
  })
}

export const getS1RedditStatus = (runId) => {
  return service.get(`/api/finance/s1/reddit/${encodeURIComponent(runId)}`)
}

export const getS1RedditPredictions = (runId, stage = 'all') => {
  return service.get(`/api/finance/s1/reddit/${encodeURIComponent(runId)}/predictions`, { params: { stage } })
}

export const getS1RedditMetrics = (runId) => {
  return service.get(`/api/finance/s1/reddit/${encodeURIComponent(runId)}/metrics`)
}

export const getS1RedditActions = (runId, limit = 100) => {
  return service.get(`/api/finance/s1/reddit/${encodeURIComponent(runId)}/actions`, { params: { limit } })
}

export const getS1RedditMapping = (runId) => {
  return service.get(`/api/finance/s1/reddit/${encodeURIComponent(runId)}/mapping`)
}

export const getS1RedditCsvDownloadUrl = (runId, kind) => {
  const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001').replace(/\/$/, '')
  return `${baseUrl}/api/finance/s1/reddit/${encodeURIComponent(runId)}/csv/${encodeURIComponent(kind)}`
}

export const prepareS1RedditBatch = (socialRounds = 6) => {
  return service.post('/api/finance/s1/reddit/batch/prepare', { social_rounds: socialRounds })
}

export const runS1RedditBatch = (batchId) => {
  return service.post('/api/finance/s1/reddit/batch/run', { batch_id: batchId })
}

export const getS1RedditBatchStatus = (batchId) => {
  return service.get(`/api/finance/s1/reddit/batch/${encodeURIComponent(batchId)}`)
}

export const getS1RedditBatchCsvDownloadUrl = (batchId) => {
  const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001').replace(/\/$/, '')
  return `${baseUrl}/api/finance/s1/reddit/batch/${encodeURIComponent(batchId)}/csv`
}
