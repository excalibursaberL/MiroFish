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
