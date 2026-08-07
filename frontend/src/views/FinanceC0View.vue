<template>
  <div class="finance-page">
    <header class="topbar">
      <button class="brand-button" type="button" @click="router.push('/')">MIROFISH</button>
      <div class="route-title">
        <span>A 股金融适配层</span>
        <strong>C0 独立判断实验</strong>
      </div>
      <div class="header-status">
        <span class="status-dot" :class="statusTone"></span>
        <span>{{ statusLabel }}</span>
      </div>
    </header>

    <main>
      <section class="intro-band">
        <div>
          <p class="eyebrow">FINANCE ADAPTER / CONTROL GROUP</p>
          <h1>{{ experimentMode === 'all' ? '全场景 C0 批量实验' : '单场景 C0 实验工作台' }}</h1>
          <p class="intro-copy">{{ experimentMode === 'all' ? '一次冻结全部匿名场景，在后台依次完成预测并生成可分析的 CSV。' : '选择一条匿名历史场景，让筛选出的 10 个投资者 Agent 在看不到彼此观点的情况下分别完成一次预测。' }}</p>
        </div>
        <div class="facts" aria-label="实验固定参数">
          <div><strong>{{ experimentMode === 'all' ? scenarios.length : 1 }}</strong><span>匿名场景</span></div>
          <div><strong>5 + 1</strong><span>历史种子 + 当前事件</span></div>
          <div><strong>10</strong><span>独立 Agent</span></div>
          <div><strong>0</strong><span>社会互动</span></div>
        </div>
      </section>

      <ol class="phase-strip" aria-label="C0 实验流程">
        <li v-for="(phase, index) in phases" :key="phase" :class="phaseClass(index + 1)">
          <span>{{ String(index + 1).padStart(2, '0') }}</span>
          <b>{{ phase }}</b>
        </li>
      </ol>

      <div class="workspace">
        <aside class="control-panel">
          <section class="control-section">
            <div class="section-heading">
              <span>01</span>
              <h2>选择运行范围</h2>
            </div>

            <label class="field-label">运行模式</label>
            <div class="mode-control" role="group" aria-label="运行模式">
              <button type="button" :class="{ active: experimentMode === 'single' }" :disabled="hasRun || busy" @click="experimentMode = 'single'">单场景</button>
              <button type="button" :class="{ active: experimentMode === 'all' }" :disabled="hasRun || busy" @click="experimentMode = 'all'">全部场景</button>
            </div>

            <template v-if="experimentMode === 'single'">
              <label class="field-label scenario-field" for="scenario">匿名场景</label>
              <select id="scenario" v-model="selectedScenarioId" :disabled="hasRun || loadingScenarios">
              <option v-for="scenario in scenarios" :key="scenario.scenario_id" :value="scenario.scenario_id">
                {{ scenario.scenario_id }} · {{ scenario.symbol }}
              </option>
              </select>
            </template>

            <div v-else class="batch-summary">
              <strong>{{ scenarios.length }} 个场景 · {{ scenarios.length * 10 }} 次预测</strong>
              <span>后台顺序执行；运行中可以关闭页面，后端不可停止。</span>
            </div>

            <div v-if="experimentMode === 'single' && selectedScenario" class="scenario-summary">
              <dl>
                <div><dt>资产</dt><dd>{{ selectedScenario.symbol }}</dd></div>
                <div><dt>企业</dt><dd>{{ selectedScenario.name }}</dd></div>
                <div><dt>预测窗口</dt><dd>{{ horizonLabel(selectedScenario.horizon) }}</dd></div>
                <div><dt>历史种子</dt><dd>{{ selectedScenario.seed_count }} 条</dd></div>
              </dl>
              <p>{{ selectedScenario.current_event_text }}</p>
            </div>

            <button class="primary-button" type="button" :disabled="!canPrepare" @click="prepareExperiment">
              {{ preparing ? '正在准备…' : (experimentMode === 'all' ? '准备全部场景' : '准备这一条测试数据') }}
            </button>
          </section>

          <section class="control-section">
            <div class="section-heading">
              <span>02</span>
              <h2>运行控制</h2>
            </div>

            <div v-if="manifest" class="run-meta">
              <span>RUN ID</span>
              <code>{{ runId }}</code>
              <dl>
                <div><dt>场景</dt><dd>{{ manifest.scenario_count }}</dd></div>
                <div><dt>Agent</dt><dd>{{ manifest.agent_count }}</dd></div>
                <div><dt>中性区间</dt><dd>±{{ formatThresholdPercent(manifest.prediction_target?.neutral_threshold) }}</dd></div>
                <div><dt>已完成</dt><dd>{{ completedCount }}/{{ expectedCount }}</dd></div>
                <div><dt>成功</dt><dd>{{ manifest.successful_prediction_count || 0 }}</dd></div>
                <div><dt>失败</dt><dd>{{ manifest.failed_prediction_count || 0 }}</dd></div>
              </dl>
            </div>
            <p v-else class="muted-copy">准备场景后，这里会显示本次运行编号和 Agent 进度。</p>

            <button class="secondary-button" type="button" :disabled="!hasRun || busy || manifest?.status === 'completed'" @click="performDryRun">
              {{ dryRunning ? '正在检查…' : '执行 Prompt 检查' }}
            </button>

            <label class="check-row" :class="{ disabled: !dryRunReady }">
              <input v-model="promptConfirmed" type="checkbox" :disabled="!dryRunReady || busy">
              <span>我已确认 Prompt 不含评测答案和其他 Agent 的观点</span>
            </label>

            <button class="run-button" type="button" :disabled="!canRun" @click="executeExperiment">
              {{ running ? `正在运行 ${completedCount}/${expectedCount}` : (manifest?.status === 'failed' && isBatchRun ? `继续剩余 ${expectedCount - completedCount} 次预测` : (isBatchRun ? `一键运行全部 ${expectedCount} 次预测` : '正式运行 10 个 Agent')) }}
            </button>

            <button v-if="hasRun && !running" class="text-button" type="button" @click="resetWorkbench">重新选择场景</button>
          </section>

          <div v-if="errorMessage" class="error-box">
            <strong>操作失败</strong>
            <p>{{ errorMessage }}</p>
          </div>
        </aside>

        <section class="result-panel">
          <div class="panel-toolbar">
            <div class="tab-list" role="tablist">
              <button v-for="tab in tabs" :key="tab.id" type="button" :class="{ active: activeTab === tab.id }" @click="activeTab = tab.id">
                {{ tab.label }}
                <span v-if="tab.id === 'results'">{{ predictions.length }}</span>
              </button>
            </div>
            <button class="refresh-button" type="button" :disabled="!hasRun || refreshing" title="刷新状态" @click="refreshRun">
              {{ refreshing ? '刷新中…' : '刷新' }}
            </button>
          </div>

          <div v-if="activeTab === 'prompt'" class="panel-content prompt-view">
            <div v-if="preview?.prompt">
              <div class="content-header">
                <div>
                  <p class="content-kicker">冻结后的首个 Agent 输入</p>
                  <h2>{{ preview.prompt.agent_role }} / Agent {{ preview.prompt.agent_id }}</h2>
                </div>
                <div class="segmented-control">
                  <button type="button" :class="{ active: promptPart === 'system' }" @click="promptPart = 'system'">System</button>
                  <button type="button" :class="{ active: promptPart === 'user' }" @click="promptPart = 'user'">User</button>
                </div>
              </div>
              <pre>{{ preview.prompt[promptPart] }}</pre>
            </div>
            <div v-else class="empty-state">
              <strong>尚未生成 Prompt</strong>
              <p>先在左侧选择并准备一条测试数据。</p>
            </div>
          </div>

          <div v-else-if="activeTab === 'results'" class="panel-content result-view">
            <div class="progress-block">
              <div class="progress-copy">
                <div>
                  <p class="content-kicker">INDEPENDENT FORECASTS</p>
                  <h2>{{ progressTitle }}</h2>
                </div>
                <strong>{{ progressPercent }}%</strong>
              </div>
              <div class="progress-track"><span :style="{ width: `${progressPercent}%` }"></span></div>
              <div class="direction-summary">
                <span class="up">上涨 {{ directionCounts.up }}</span>
                <span class="neutral">中性 {{ directionCounts.neutral }}</span>
                <span class="down">下跌 {{ directionCounts.down }}</span>
                <span class="failed">异常 {{ failedCount }}</span>
              </div>
            </div>

            <div v-if="hasRun && (predictions.length || manifest?.status === 'completed')" class="csv-actions">
              <a :href="csvUrl('predictions')">下载预测 CSV</a>
              <a v-if="manifest?.status === 'completed'" :href="csvUrl('evaluation')">下载评测 CSV</a>
              <a v-if="manifest?.status === 'completed'" :href="csvUrl('agent_token_usage')">下载 Agent Token CSV</a>
              <span>评测 CSV 仅在全部 Agent 完成后生成。</span>
            </div>

            <section v-if="outcome" class="outcome-strip" aria-label="研究者评测结果">
              <div class="outcome-heading">
                <div>
                  <p class="content-kicker">RESEARCHER-ONLY GROUND TRUTH</p>
                  <h2>真实市场结果</h2>
                </div>
                <span>仅在实验完成后读取，不进入 Agent Prompt</span>
              </div>
              <div class="outcome-values">
                <div>
                  <span>Astock 原始标签</span>
                  <strong :class="outcome.astock_direction">{{ directionLabel(outcome.astock_direction) }}</strong>
                  <b>{{ formatSignedPercent(outcome.astock_change_return) }}</b>
                </div>
                <div>
                  <span>未来 5 日累计收盘结果</span>
                  <strong :class="outcome.five_day_close_direction">{{ directionLabel(outcome.five_day_close_direction) }}</strong>
                  <b>{{ formatSignedPercent(outcome.five_day_close_return) }}</b>
                </div>
              </div>
              <p>Astock 的 label/CHANGE 与本实验采用的“五日端到端收盘收益”是两种不同统计口径，因此方向可能不同。五日收益在 ±{{ formatThresholdPercent(outcome.five_day_neutral_threshold) }} 内判为中性；中性表示价格变化很小，不表示 Agent 没有把握。</p>
            </section>

            <div v-if="predictions.length" class="table-wrap">
              <table :class="{ 'batch-table': isBatchRun }">
                <thead>
                  <tr>
                    <th v-if="isBatchRun" class="scenario-head">场景</th>
                    <th class="agent-head">Agent</th><th class="role-head">角色</th><th>方向</th><th>上涨</th><th>中性</th><th>下跌</th><th>置信度</th><th>依据</th><th class="attempt-head">调用</th><th class="record-head">状态</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="prediction in predictions" :key="`${prediction.scenario_id}-${prediction.agent_id}`">
                    <td v-if="isBatchRun" class="mono">{{ prediction.scenario_id }}</td>
                    <td class="mono">{{ formatAgentId(prediction.agent_id) }}</td>
                    <td>{{ prediction.agent_role_label || prediction.agent_role }}</td>
                    <td><span class="direction-badge" :class="prediction.direction">{{ directionLabel(prediction.direction) }}</span></td>
                    <td>{{ formatProbability(prediction.up_probability) }}</td>
                    <td>{{ formatProbability(prediction.neutral_probability) }}</td>
                    <td>{{ formatProbability(prediction.down_probability) }}</td>
                    <td>{{ formatProbability(prediction.confidence) }}</td>
                    <td class="reason-cell">{{ prediction.reason || prediction.error || '—' }}</td>
                    <td>{{ prediction.attempt_count || 1 }} 次</td>
                    <td><span class="record-status" :class="prediction.status">{{ prediction.status }}</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div v-else class="empty-state compact">
              <strong>尚无预测结果</strong>
              <p>完成 Prompt 检查后，在左侧启动独立 Agent。</p>
            </div>
          </div>

          <div v-else class="panel-content log-view">
            <div class="content-header">
              <div><p class="content-kicker">LOCAL OPERATION LOG</p><h2>实验操作记录</h2></div>
            </div>
            <ol v-if="operationLogs.length" class="log-list">
              <li v-for="(entry, index) in operationLogs" :key="`${entry.time}-${index}`">
                <time>{{ entry.time }}</time><span>{{ entry.message }}</span>
              </li>
            </ol>
            <div v-else class="empty-state compact"><p>还没有操作记录。</p></div>
          </div>
        </section>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  getC0CsvDownloadUrl,
  getC0Outcome,
  getC0Predictions,
  getC0Preview,
  getC0Status,
  listC0Scenarios,
  prepareC0,
  runC0
} from '../api/finance'

const router = useRouter()
const STORAGE_KEY = 'mirofish_finance_c0_run_id'

const scenarios = ref([])
const experimentMode = ref('single')
const selectedScenarioId = ref('')
const runId = ref('')
const manifest = ref(null)
const preview = ref(null)
const predictions = ref([])
const outcome = ref(null)
const activeTab = ref('prompt')
const promptPart = ref('user')
const promptConfirmed = ref(false)
const loadingScenarios = ref(false)
const preparing = ref(false)
const dryRunning = ref(false)
const running = ref(false)
const refreshing = ref(false)
const errorMessage = ref('')
const operationLogs = ref([])
const terminalLogKey = ref('')
let pollTimer = null

const phases = ['选择数据', '冻结输入', '检查 Prompt', '运行 Agent', '查看结果']
const tabs = [
  { id: 'prompt', label: 'Prompt 预览' },
  { id: 'results', label: 'Agent 结果' },
  { id: 'logs', label: '操作记录' }
]

const selectedScenario = computed(() => scenarios.value.find(item => item.scenario_id === selectedScenarioId.value))
const hasRun = computed(() => Boolean(runId.value && manifest.value))
const busy = computed(() => preparing.value || dryRunning.value || running.value)
const isBatchRun = computed(() => (manifest.value?.run_mode || experimentMode.value) === 'all')
const canPrepare = computed(() => !hasRun.value && !busy.value && (experimentMode.value === 'all' ? scenarios.value.length > 0 : Boolean(selectedScenarioId.value)))
const dryRunReady = computed(() => ['dry_run', 'running', 'completed', 'failed'].includes(manifest.value?.status))
const canRun = computed(() => hasRun.value && dryRunReady.value && promptConfirmed.value && !busy.value && manifest.value?.status !== 'completed')
const expectedCount = computed(() => manifest.value?.expected_prediction_count || manifest.value?.agent_count || 10)
const completedCount = computed(() => manifest.value?.completed_prediction_count ?? manifest.value?.prediction_count ?? predictions.value.length)
const failedCount = computed(() => predictions.value.filter(item => item.status !== 'ok').length)
const progressPercent = computed(() => expectedCount.value ? Math.min(100, Math.round((completedCount.value / expectedCount.value) * 100)) : 0)
const directionCounts = computed(() => predictions.value.reduce((counts, item) => {
  if (item.status === 'ok' && counts[item.direction] !== undefined) counts[item.direction] += 1
  return counts
}, { up: 0, neutral: 0, down: 0 }))

const phaseLevel = computed(() => {
  if (manifest.value?.status === 'completed') return 5
  if (['queued', 'running'].includes(manifest.value?.status)) return 4
  if (manifest.value?.status === 'failed') return 4
  if (manifest.value?.status === 'dry_run') return 3
  if (hasRun.value) return 2
  return 1
})

const statusLabel = computed(() => ({
  prepared: '输入已冻结', dry_run: 'Prompt 已检查', queued: '后台任务排队中', running: 'Agent 运行中', completed: '实验完成', failed: '运行中断'
}[manifest.value?.status] || (loadingScenarios.value ? '读取数据中' : '等待准备')))
const statusTone = computed(() => manifest.value?.status === 'completed' ? 'completed' : (manifest.value?.status === 'failed' || errorMessage.value ? 'error' : (running.value ? 'running' : 'idle')))
const progressTitle = computed(() => {
  if (manifest.value?.status === 'completed') return `${expectedCount.value} 次独立预测已完成`
  if (manifest.value?.status === 'queued') return '批量任务已经进入后台队列'
  if (manifest.value?.status === 'running') return `正在处理第 ${Math.min(expectedCount.value, completedCount.value + 1)} 次预测`
  if (manifest.value?.status === 'failed') return `运行在 ${completedCount.value}/${expectedCount.value} 处中断`
  return '等待正式运行'
})

const phaseClass = (level) => ({ active: phaseLevel.value === level, complete: phaseLevel.value > level })

const addLog = (message) => {
  operationLogs.value.unshift({
    time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
    message
  })
}

const loadScenarios = async () => {
  loadingScenarios.value = true
  try {
    const response = await listC0Scenarios()
    scenarios.value = response.data || []
    if (!selectedScenarioId.value && scenarios.value.length) selectedScenarioId.value = scenarios.value[0].scenario_id
    addLog(`已读取 ${scenarios.value.length} 个匿名候选场景。`)
  } catch (error) {
    errorMessage.value = error.message
  } finally {
    loadingScenarios.value = false
  }
}

const prepareExperiment = async () => {
  preparing.value = true
  errorMessage.value = ''
  predictions.value = []
  outcome.value = null
  try {
    const response = await prepareC0(selectedScenarioId.value, experimentMode.value)
    manifest.value = response.data
    runId.value = response.data.run_id
    localStorage.setItem(STORAGE_KEY, runId.value)
    addLog(`${response.data.scenario_count} 个场景已冻结，生成 ${response.data.expected_prediction_count} 份 Agent 输入。`)
    await loadPreview()
    activeTab.value = 'prompt'
  } catch (error) {
    errorMessage.value = error.message
    addLog(`准备失败：${error.message}`)
  } finally {
    preparing.value = false
  }
}

const loadPreview = async () => {
  if (!runId.value) return
  const response = await getC0Preview(runId.value)
  preview.value = response.data
}

const performDryRun = async () => {
  dryRunning.value = true
  errorMessage.value = ''
  try {
    const response = await runC0(runId.value, true)
    manifest.value = response.data
    await loadPreview()
    addLog('Prompt dry-run 完成，没有调用真实 LLM。')
    activeTab.value = 'prompt'
  } catch (error) {
    errorMessage.value = error.message
    addLog(`Prompt 检查失败：${error.message}`)
  } finally {
    dryRunning.value = false
  }
}

const executeExperiment = async () => {
  const background = isBatchRun.value
  const resuming = background && manifest.value?.status === 'failed' && completedCount.value > 0
  let backgroundAccepted = false
  running.value = true
  errorMessage.value = ''
  activeTab.value = 'results'
  addLog(resuming
    ? `继续批量运行：保留已有 ${completedCount.value} 条预测，处理剩余 ${expectedCount.value - completedCount.value} 条。`
    : `正式运行已启动：${manifest.value.scenario_count} 个场景，${expectedCount.value} 次独立预测。`)
  startPolling()
  try {
    const response = await runC0(runId.value, false, background)
    manifest.value = response.data
    if (background) {
      backgroundAccepted = true
      addLog('全部场景已交给后台执行，可以保留后端运行并关闭本页面。')
      return
    }
    await Promise.all([loadPredictions(), loadOutcome()])
    addLog(`实验完成：成功 ${manifest.value.successful_prediction_count || 0}，失败 ${manifest.value.failed_prediction_count || 0}。`)
  } catch (error) {
    errorMessage.value = error.message
    addLog(`正式运行请求失败：${error.message}`)
    await refreshRun(false)
  } finally {
    if (!backgroundAccepted) {
      running.value = false
      stopPolling()
    }
  }
}

const loadPredictions = async () => {
  if (!runId.value) return
  const response = await getC0Predictions(runId.value)
  predictions.value = response.data || []
}

const loadOutcome = async () => {
  if (!runId.value || manifest.value?.status !== 'completed' || isBatchRun.value) return
  try {
    const response = await getC0Outcome(runId.value)
    outcome.value = response.data
  } catch (error) {
    outcome.value = null
    addLog(`真实结果读取失败：${error.message}`)
  }
}

const refreshRun = async (showSpinner = true) => {
  if (!runId.value) return
  if (showSpinner) refreshing.value = true
  try {
    const [statusResponse] = await Promise.all([
      getC0Status(runId.value),
      loadPredictions()
    ])
    manifest.value = statusResponse.data
    if (manifest.value.status === 'completed') {
      await loadOutcome()
      running.value = false
      stopPolling()
      const logKey = `${runId.value}:completed`
      if (terminalLogKey.value !== logKey) {
        terminalLogKey.value = logKey
        addLog(`实验完成：成功 ${manifest.value.successful_prediction_count || 0}，失败 ${manifest.value.failed_prediction_count || 0}。`)
      }
    } else if (manifest.value.status === 'failed') {
      running.value = false
      stopPolling()
      errorMessage.value = manifest.value.background_error || '后台运行已中断'
      const logKey = `${runId.value}:failed`
      if (terminalLogKey.value !== logKey) {
        terminalLogKey.value = logKey
        addLog(`后台运行中断：${errorMessage.value}`)
      }
    }
  } catch (error) {
    if (showSpinner) errorMessage.value = error.message
  } finally {
    if (showSpinner) refreshing.value = false
  }
}

const startPolling = () => {
  stopPolling()
  pollTimer = window.setInterval(() => refreshRun(false), 1200)
}

const stopPolling = () => {
  if (pollTimer) window.clearInterval(pollTimer)
  pollTimer = null
}

const restoreRun = async () => {
  const storedRunId = localStorage.getItem(STORAGE_KEY)
  if (!storedRunId) return
  runId.value = storedRunId
  try {
    const response = await getC0Status(storedRunId)
    manifest.value = response.data
    experimentMode.value = response.data.run_mode || 'single'
    selectedScenarioId.value = response.data.scenario_ids?.[0] || selectedScenarioId.value
    await Promise.all([loadPreview(), loadPredictions(), loadOutcome()])
    promptConfirmed.value = ['queued', 'running', 'completed', 'failed'].includes(manifest.value.status)
    if (['queued', 'running'].includes(manifest.value.status)) {
      running.value = true
      startPolling()
    }
    addLog(`已恢复运行 ${storedRunId}。`)
  } catch {
    localStorage.removeItem(STORAGE_KEY)
    runId.value = ''
  }
}

const resetWorkbench = () => {
  stopPolling()
  localStorage.removeItem(STORAGE_KEY)
  runId.value = ''
  manifest.value = null
  preview.value = null
  predictions.value = []
  outcome.value = null
  terminalLogKey.value = ''
  promptConfirmed.value = false
  errorMessage.value = ''
  activeTab.value = 'prompt'
  addLog('工作台已重置；磁盘上的历史运行文件仍然保留。')
}

const directionLabel = (direction) => ({ up: '上涨', neutral: '中性', down: '下跌' }[direction] || '—')
const formatProbability = (value) => value === null || value === undefined ? '—' : `${Math.round(Number(value) * 100)}%`
const formatThresholdPercent = (value) => `${((Number(value) || 0.017) * 100).toFixed(1)}%`
const formatSignedPercent = (value) => {
  if (value === null || value === undefined) return '—'
  const number = Number(value) * 100
  return `${number >= 0 ? '+' : ''}${number.toFixed(2)}%`
}
const formatAgentId = (value) => `A${String(Number(value) + 1).padStart(2, '0')}`
const horizonLabel = (value) => value === 'next_5_trading_days' ? '未来 5 个交易日' : value
const csvUrl = (kind) => getC0CsvDownloadUrl(runId.value, kind)

onMounted(async () => {
  await loadScenarios()
  await restoreRun()
})

onBeforeUnmount(stopPolling)
</script>

<style scoped>
.finance-page {
  min-height: 100vh;
  color: #17211d;
  background: #f4f6f5;
  font-family: "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
}

button, select { font: inherit; }
button { letter-spacing: 0; }

.topbar {
  height: 60px;
  display: grid;
  grid-template-columns: 180px 1fr auto;
  align-items: center;
  gap: 24px;
  padding: 0 32px;
  color: #fff;
  background: #111714;
  border-bottom: 1px solid #324039;
}

.brand-button {
  width: max-content;
  padding: 0;
  color: #fff;
  background: transparent;
  border: 0;
  font-family: "JetBrains Mono", monospace;
  font-weight: 800;
  cursor: pointer;
}

.route-title { display: flex; align-items: baseline; gap: 12px; font-size: 13px; color: #9fafaa; }
.route-title strong { color: #fff; font-size: 14px; }
.header-status { display: flex; align-items: center; gap: 9px; font-size: 13px; }
.status-dot { width: 8px; height: 8px; background: #97a19d; }
.status-dot.running { background: #e8a33d; animation: pulse 1.3s infinite; }
.status-dot.completed { background: #33a06f; }
.status-dot.error { background: #c94d42; }

main { width: min(1480px, calc(100% - 56px)); margin: 0 auto; padding: 30px 0 44px; }
.intro-band { display: flex; justify-content: space-between; align-items: end; gap: 36px; padding: 0 0 27px; border-bottom: 1px solid #cbd2ce; }
.eyebrow, .content-kicker { margin: 0 0 8px; color: #2c7659; font: 700 11px/1.4 "JetBrains Mono", monospace; }
h1 { margin: 0; font-size: 31px; line-height: 1.3; letter-spacing: 0; }
.intro-copy { max-width: 720px; margin: 10px 0 0; color: #5d6964; font-size: 15px; }
.facts { display: grid; grid-template-columns: repeat(4, 112px); border: 1px solid #cbd2ce; background: #fff; }
.facts div { min-height: 74px; padding: 12px 14px; border-right: 1px solid #dce1de; }
.facts div:last-child { border-right: 0; }
.facts strong { display: block; font: 700 21px/1.2 "JetBrains Mono", monospace; }
.facts span { display: block; margin-top: 5px; color: #65706c; font-size: 11px; }

.phase-strip { display: grid; grid-template-columns: repeat(5, 1fr); margin: 22px 0; padding: 0; list-style: none; border: 1px solid #cbd2ce; background: #fff; }
.phase-strip li { min-height: 54px; display: flex; align-items: center; gap: 11px; padding: 10px 16px; color: #7d8783; border-right: 1px solid #dce1de; }
.phase-strip li:last-child { border-right: 0; }
.phase-strip span { font: 700 11px "JetBrains Mono", monospace; }
.phase-strip b { font-size: 13px; }
.phase-strip li.active { color: #18241f; background: #fff4df; box-shadow: inset 0 -3px #d98924; }
.phase-strip li.complete { color: #28664f; background: #eef7f2; }

.workspace { min-height: 660px; display: grid; grid-template-columns: 350px minmax(0, 1fr); border: 1px solid #bdc6c1; background: #fff; }
.control-panel { border-right: 1px solid #bdc6c1; background: #fafbfa; }
.control-section { padding: 23px 24px; border-bottom: 1px solid #d9dfdc; }
.section-heading { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
.section-heading span { color: #b66c1c; font: 700 11px "JetBrains Mono", monospace; }
.section-heading h2, .content-header h2, .progress-copy h2 { margin: 0; font-size: 16px; line-height: 1.4; }
.field-label { display: block; margin-bottom: 7px; color: #55615c; font-size: 12px; font-weight: 700; }
select { width: 100%; height: 40px; padding: 0 34px 0 11px; color: #1a2520; background: #fff; border: 1px solid #aeb9b3; border-radius: 3px; }
.scenario-field { margin-top: 17px; }
.mode-control { display: grid; grid-template-columns: 1fr 1fr; padding: 2px; border: 1px solid #b9c3be; border-radius: 4px; background: #e9eeeb; }
.mode-control button { min-height: 35px; color: #5d6963; background: transparent; border: 0; border-radius: 2px; cursor: pointer; }
.mode-control button.active { color: #173c2e; background: #fff; box-shadow: 0 1px 3px rgba(17, 35, 27, .12); font-weight: 700; }
.batch-summary { margin-top: 16px; padding: 13px 14px; border-left: 3px solid #b66c1c; background: #fff6e8; }
.batch-summary strong, .batch-summary span { display: block; }
.batch-summary strong { color: #4a3523; font-size: 13px; }
.batch-summary span { margin-top: 5px; color: #735e4b; font-size: 11px; line-height: 1.5; }

.scenario-summary { margin-top: 14px; padding-top: 13px; border-top: 1px solid #dde2df; }
.scenario-summary dl, .run-meta dl { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; margin: 0; }
.scenario-summary dl div, .run-meta dl div { min-width: 0; }
dt { color: #7b8581; font-size: 11px; }
dd { margin: 2px 0 0; overflow-wrap: anywhere; font: 600 12px/1.4 "JetBrains Mono", "Microsoft YaHei", monospace; }
.scenario-summary p { max-height: 108px; margin: 13px 0 0; padding: 11px; overflow: auto; color: #4f5b56; background: #f0f3f1; font-size: 12px; line-height: 1.65; }

.primary-button, .secondary-button, .run-button { width: 100%; min-height: 42px; margin-top: 16px; padding: 9px 12px; border-radius: 3px; font-weight: 700; cursor: pointer; }
.primary-button { color: #fff; background: #1d6349; border: 1px solid #1d6349; }
.secondary-button { color: #1d2924; background: #fff; border: 1px solid #839089; }
.run-button { color: #fff; background: #b96120; border: 1px solid #b96120; }
button:disabled { cursor: not-allowed; opacity: .45; }
.text-button { display: block; margin: 13px auto 0; padding: 4px 8px; color: #53605a; background: transparent; border: 0; text-decoration: underline; cursor: pointer; }

.run-meta { padding: 12px; background: #edf2ef; border-left: 3px solid #39725b; }
.run-meta > span { display: block; color: #69746f; font: 700 10px "JetBrains Mono", monospace; }
.run-meta > code { display: block; margin: 4px 0 12px; overflow-wrap: anywhere; color: #1e3d31; font-size: 12px; }
.muted-copy { margin: 0; color: #74807b; font-size: 12px; line-height: 1.6; }
.check-row { display: flex; gap: 9px; align-items: flex-start; margin-top: 15px; color: #45514c; font-size: 12px; line-height: 1.5; cursor: pointer; }
.check-row.disabled { opacity: .5; cursor: not-allowed; }
.check-row input { width: 16px; height: 16px; margin-top: 1px; accent-color: #2b7357; }
.error-box { margin: 18px; padding: 13px 15px; color: #842d28; background: #fff0ee; border-left: 4px solid #bd443c; }
.error-box p { margin: 5px 0 0; font-size: 12px; line-height: 1.5; }

.result-panel { min-width: 0; display: grid; grid-template-rows: 52px minmax(0, 1fr); }
.panel-toolbar { display: flex; justify-content: space-between; align-items: stretch; padding: 0 18px; border-bottom: 1px solid #cbd2ce; }
.tab-list { display: flex; gap: 3px; }
.tab-list button { min-width: 120px; padding: 0 13px; color: #65716c; background: transparent; border: 0; border-bottom: 3px solid transparent; font-weight: 700; cursor: pointer; }
.tab-list button.active { color: #1d2a24; border-bottom-color: #277052; }
.tab-list span { margin-left: 5px; padding: 1px 5px; color: #fff; background: #65716c; border-radius: 8px; font-size: 10px; }
.refresh-button { padding: 0 9px; color: #40504a; background: transparent; border: 0; cursor: pointer; }
.panel-content { min-width: 0; padding: 25px 27px; overflow: auto; }
.content-header { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 17px; }
.content-kicker { margin-bottom: 5px; }
.segmented-control { display: flex; padding: 2px; background: #e8ecea; border: 1px solid #ccd4d0; border-radius: 4px; }
.segmented-control button { min-width: 75px; padding: 6px 10px; color: #65716c; background: transparent; border: 0; border-radius: 2px; cursor: pointer; }
.segmented-control button.active { color: #17211d; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.08); }
pre { min-height: 500px; margin: 0; padding: 20px; overflow: auto; color: #dfe9e4; background: #18231e; border: 1px solid #101713; font: 12px/1.75 "JetBrains Mono", "Microsoft YaHei", monospace; white-space: pre-wrap; overflow-wrap: anywhere; }

.progress-block { padding-bottom: 20px; border-bottom: 1px solid #d9dfdc; }
.progress-copy { display: flex; justify-content: space-between; align-items: end; }
.progress-copy > strong { color: #28664f; font: 700 25px "JetBrains Mono", monospace; }
.progress-track { height: 8px; margin-top: 14px; overflow: hidden; background: #e2e7e4; }
.progress-track span { display: block; height: 100%; background: #318061; transition: width .25s ease; }
.direction-summary { display: flex; gap: 10px; margin-top: 12px; }
.direction-summary span { padding: 4px 8px; font-size: 11px; border-left: 3px solid; background: #f3f5f4; }
.direction-summary .up { color: #a43a34; border-color: #c94d42; }
.direction-summary .neutral { color: #5c6762; border-color: #8b9590; }
.direction-summary .down { color: #27618b; border-color: #3b7cab; }
.direction-summary .failed { color: #8a5b1d; border-color: #d28c30; }
.csv-actions { display: flex; align-items: center; gap: 10px; margin-top: 14px; }
.csv-actions a { padding: 7px 10px; color: #fff; background: #2d6e55; border-radius: 3px; font-size: 12px; font-weight: 700; text-decoration: none; }
.csv-actions span { color: #737e79; font-size: 11px; }

.outcome-strip { margin-top: 18px; padding: 17px 19px; border: 1px solid #cbd3cf; border-left: 4px solid #b66c1c; background: #fbfaf7; }
.outcome-heading { display: flex; justify-content: space-between; align-items: end; gap: 20px; }
.outcome-heading h2 { margin: 0; font-size: 15px; }
.outcome-heading > span { color: #707a75; font-size: 11px; }
.outcome-values { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; margin-top: 14px; background: #d7ddda; border: 1px solid #d7ddda; }
.outcome-values > div { display: grid; grid-template-columns: 1fr auto auto; align-items: center; gap: 12px; padding: 12px 14px; background: #fff; }
.outcome-values span { color: #59655f; font-size: 12px; }
.outcome-values strong { font-size: 14px; }
.outcome-values strong.up { color: #a43a34; }
.outcome-values strong.neutral { color: #5c6762; }
.outcome-values strong.down { color: #27618b; }
.outcome-values b { font: 700 13px "JetBrains Mono", monospace; }
.outcome-strip > p { margin: 11px 0 0; color: #68736e; font-size: 11px; line-height: 1.6; }

.table-wrap { margin-top: 20px; overflow: auto; border: 1px solid #d4dad7; }
table { width: 100%; min-width: 1120px; border-collapse: collapse; table-layout: fixed; font-size: 12px; }
table.batch-table { min-width: 1210px; }
th, td { padding: 10px 9px; text-align: left; vertical-align: top; border-bottom: 1px solid #e0e5e2; }
th { position: sticky; top: 0; z-index: 1; color: #5c6862; background: #edf1ef; font-size: 10px; }
th:nth-child(1) { width: 56px; } th:nth-child(2) { width: 142px; } th:nth-child(3) { width: 68px; }
th:nth-child(4), th:nth-child(5), th:nth-child(6), th:nth-child(7) { width: 66px; }
th:nth-child(9) { width: 58px; } th:nth-child(10) { width: 84px; }
th.scenario-head { width: 78px; }
th.agent-head { width: 56px; }
th.role-head { width: 142px; }
th.attempt-head { width: 58px; }
th.record-head { width: 84px; }
.mono { font-family: "JetBrains Mono", monospace; }
.reason-cell { line-height: 1.55; overflow-wrap: anywhere; }
.direction-badge, .record-status { display: inline-block; padding: 3px 6px; border-radius: 2px; font-size: 10px; font-weight: 700; }
.direction-badge.up { color: #942d28; background: #fdecea; }
.direction-badge.neutral { color: #56615c; background: #edf0ef; }
.direction-badge.down { color: #225a83; background: #e9f2f8; }
.record-status.ok { color: #22634a; background: #e8f5ee; }
.record-status.error, .record-status.parse_error { color: #8b302a; background: #fdecea; }

.log-list { margin: 0; padding: 0; list-style: none; border-top: 1px solid #d4dad7; }
.log-list li { display: grid; grid-template-columns: 92px 1fr; gap: 14px; padding: 13px 5px; border-bottom: 1px solid #e1e5e3; }
.log-list time { color: #71807a; font: 11px "JetBrains Mono", monospace; }
.log-list span { font-size: 13px; }
.empty-state { min-height: 500px; display: grid; place-content: center; text-align: center; color: #7a8580; }
.empty-state strong { color: #3c4943; font-size: 16px; }
.empty-state p { margin: 7px 0 0; font-size: 13px; }
.empty-state.compact { min-height: 270px; }

@keyframes pulse { 50% { opacity: .35; } }

@media (max-width: 1100px) {
  .facts { grid-template-columns: repeat(2, 112px); }
  .facts div:nth-child(2) { border-right: 0; }
  .facts div:nth-child(-n+2) { border-bottom: 1px solid #dce1de; }
  .workspace { grid-template-columns: 320px minmax(0, 1fr); }
}
</style>
