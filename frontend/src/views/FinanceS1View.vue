<template>
  <div class="s1-page">
    <header class="topbar">
      <button class="brand-button" type="button" @click="router.push('/')">MIROFISH</button>
      <div class="route-title">
        <span>A 股金融适配层</span>
        <strong>S1 Reddit 社会互动实验</strong>
      </div>
      <div class="header-status">
        <span class="status-dot" :class="statusTone"></span>
        <span>{{ statusLabel }}</span>
      </div>
    </header>

    <main>
      <section class="intro-band">
        <div>
          <p class="eyebrow">FINANCE ADAPTER / SOCIAL CONDITION</p>
          <h1>Reddit 社会互动实验工作台</h1>
          <p class="intro-copy">同一批 Agent 在社会互动前后各预测一次。两次预测之间唯一新增的信息，是 Reddit 中其他投资者的公开观点。</p>
        </div>
        <div class="facts" aria-label="S1 实验固定参数">
          <div><strong>20</strong><span>投资者 Agent</span></div>
          <div><strong>{{ sourceCount }}</strong><span>动态信息源</span></div>
          <div><strong>5 + 1</strong><span>历史 + 当前事件</span></div>
          <div><strong>{{ socialRounds }}</strong><span>社会互动轮</span></div>
        </div>
      </section>

      <ol class="phase-strip" aria-label="S1 实验流程">
        <li v-for="(phase, index) in phases" :key="phase" :class="phaseClass(index + 1)">
          <span>{{ String(index + 1).padStart(2, '0') }}</span><b>{{ phase }}</b>
        </li>
      </ol>

      <div class="workspace">
        <aside class="control-panel">
          <section class="control-section">
            <div class="section-heading"><span>01</span><h2>选择运行范围</h2></div>
            <div class="mode-control" role="group" aria-label="运行范围">
              <button type="button" :class="{ active: runMode === 'single' }" :disabled="busy" @click="runMode = 'single'">当前场景</button>
              <button type="button" :class="{ active: runMode === 'all' }" :disabled="busy" @click="runMode = 'all'">全部已构图场景</button>
            </div>
            <p v-if="runMode === 'all'" class="mode-help">从本地 Zep 图谱清单读取已完成场景，并严格串行运行。目前清单中的场景不会重新建图。</p>
            <template v-else>
            <label class="field-label" for="scenario">场景</label>
            <select id="scenario" v-model="selectedScenarioId" :disabled="hasRun || busy || loadingScenarios">
              <option v-for="scenario in scenarios" :key="scenario.scenario_id" :value="scenario.scenario_id">
                {{ scenario.scenario_id }} · {{ scenario.symbol }}
              </option>
            </select>
            <div v-if="selectedScenario" class="scenario-summary">
              <dl>
                <div><dt>资产</dt><dd>{{ selectedScenario.symbol }}</dd></div>
                <div><dt>企业代号</dt><dd>{{ selectedScenario.name }}</dd></div>
                <div><dt>种子</dt><dd>{{ selectedScenario.seed_count }} 条</dd></div>
              </dl>
              <p>{{ selectedScenario.current_event_text }}</p>
            </div>
            </template>
          </section>

          <section v-if="runMode === 'single'" class="control-section">
            <div class="section-heading"><span>02</span><h2>选择事件并构建图谱</h2></div>
            <p class="mode-help">S1 使用 5 条历史种子和 1 条当前事件。默认全部纳入，保证图谱能识别每条事件的发布主体。</p>
            <div v-if="seedDocument?.events?.length" class="event-pick-list">
              <label v-for="event in seedDocument.events" :key="event.event_id" class="event-pick-row">
                <input v-model="selectedGraphEventIds" type="checkbox" :value="event.event_id" :disabled="hasRun || busy || event.phase === 'current'">
                <span class="event-pick-round">{{ event.phase === 'current' ? '当前' : `种子 ${event.seed_rank}` }}</span>
                <span class="event-pick-text">{{ event.text }}</span>
              </label>
            </div>
            <p v-else class="hint">正在读取当前场景的安全事件文本…</p>
            <p v-if="seedDocument?.events?.length && !allGraphEventsSelected" class="hint warning">正式 S1 需要六条事件全部选中；当前事件不可取消。</p>
            <div v-if="graphProgress" class="build-progress"><div class="progress-track"><span :style="{ width: `${graphProgress.progress}%` }"></span></div><small>{{ graphProgress.message }} {{ graphProgress.progress }}%</small></div>
            <button v-if="sourceMode === 'graph' && !projectId.trim() && !graphId.trim()" class="graph-button" type="button" :disabled="!canBuildGraph" @click="buildGraphForScenario">
              {{ graphBuilding ? '正在构建匿名图谱…' : '从当前事件新建图谱' }}
            </button>
            <p v-else class="hint">{{ sourceMode === 'graph' ? '当前已选择图谱，冻结时会读取该图谱实体。' : '文本回退模式不创建 Zep 图谱，仅用于调试归因。' }}</p>
          </section>

          <section v-if="runMode === 'single'" class="control-section">
            <div class="section-heading"><span>03</span><h2>信息源模式</h2></div>
            <div class="mode-control" role="group" aria-label="信息源模式">
              <button type="button" :class="{ active: sourceMode === 'graph' }" :disabled="hasRun || busy" @click="sourceMode = 'graph'">Zep 图谱</button>
              <button type="button" :class="{ active: sourceMode === 'scenario' }" :disabled="hasRun || busy" @click="sourceMode = 'scenario'">文本回退</button>
            </div>
            <p class="mode-help">
              {{ sourceMode === 'graph' ? '正式实验：使用 MiroFish 已构建的图谱实体。' : '调试模式：从匿名事件中的“公告/媒体电”等表述解析发布者。' }}
            </p>
            <template v-if="sourceMode === 'graph'">
              <label class="field-label" for="project-id">已完成图谱的 MiroFish 项目</label>
              <select id="project-id" v-model="projectId" :disabled="hasRun || busy || loadingProjects">
                <option value="">请选择项目</option>
                <option v-for="project in graphProjects" :key="project.project_id" :value="project.project_id">{{ project.name || '未命名项目' }} · {{ project.project_id }}</option>
              </select>
              <p v-if="!loadingProjects && !graphProjects.length" class="hint warning">没有找到已完成图谱的项目，请先回到 MiroFish 完成图谱构建。</p>
              <p v-if="selectedProject" class="project-hint">graph: {{ selectedProject.graph_id }}</p>
              <label class="field-label spaced" for="graph-id">或直接填写 graph ID</label>
              <input id="graph-id" v-model="graphId" :disabled="hasRun || busy" placeholder="mirofish_xxxxxxxxxxxxxxxx">
            </template>
            <p v-if="sourceMode === 'graph' && !projectId.trim() && !graphId.trim()" class="hint warning">图谱模式需要填写项目 ID 或 graph ID。</p>
            <p v-else class="hint">准备阶段会冻结事件与发布者映射，不会调用预测 LLM。</p>
            <button class="primary-button" type="button" :disabled="!canPrepare" @click="prepareExperiment">
              {{ preparing ? '正在准备…' : '冻结输入并解析信息源' }}
            </button>
          </section>

          <section class="control-section">
            <div class="section-heading"><span>04</span><h2>互动与运行控制</h2></div>
            <div class="round-config">
              <div><label class="field-label" for="social-rounds">社会互动轮数</label><input id="social-rounds" v-model.number="socialRounds" type="number" min="1" max="12" :disabled="settingsLocked" @change="settingsDirty = runMode === 'single' && hasRun"><small>每轮代表一次互动机会，不再映射成现实中的分钟。正式试验建议固定为 6 轮。</small></div>
            </div>
            <p v-if="settingsDirty" class="hint warning">参数已修改，请点击“应用互动参数”后再启动实验。</p>
            <button v-if="runMode === 'single' && manifest?.status === 'prepared'" class="primary-button" type="button" :disabled="!settingsDirty || updatingSettings" @click="applySettings">
              {{ updatingSettings ? '正在应用…' : '应用互动参数' }}
            </button>
            <div v-if="runMode === 'single' && manifest" class="run-meta">
              <span>RUN ID</span><code>{{ runId }}</code>
              <dl>
                <div><dt>信息源</dt><dd>{{ manifest.source_agent_count }}</dd></div>
                <div><dt>总账号</dt><dd>{{ manifest.agent_count_total }}</dd></div>
                <div><dt>已完成</dt><dd>{{ completedCount }}/{{ expectedCount }}</dd></div>
                <div><dt>来源模式</dt><dd>{{ sourceModeLabel(manifest.source_mode) }}</dd></div>
                <div><dt>互动轮数</dt><dd>{{ manifest.social_rounds }}</dd></div>
              </dl>
            </div>
            <p v-else-if="runMode === 'single'" class="muted-copy">冻结输入后，这里会显示运行编号和互动进度。</p>
            <button v-if="runMode === 'single'" class="run-button" type="button" :disabled="!canRun" @click="executeExperiment">
              {{ running ? `正在运行 ${completedCount}/${expectedCount}` : '启动 Reddit 社会互动' }}
            </button>
            <button v-else class="run-button" type="button" :disabled="busy || batchRunning" @click="runAllScenarios">
              {{ batchRunning ? `正在串行运行 ${batchCompleted}/${batchTotal}` : '一键运行全部已构图场景' }}
            </button>
            <div v-if="runMode === 'all' && batchManifest" class="run-meta batch-meta">
              <span>BATCH ID</span><code>{{ batchManifest.batch_id }}</code>
              <dl>
                <div><dt>当前场景</dt><dd>{{ batchManifest.current_scenario_id || '—' }}</dd></div>
                <div><dt>完成</dt><dd>{{ batchCompleted }}/{{ batchTotal }}</dd></div>
                <div><dt>失败</dt><dd>{{ batchManifest.failed_scenario_count || 0 }}</dd></div>
                <div><dt>状态</dt><dd>{{ batchManifest.status }}</dd></div>
              </dl>
              <a v-if="['completed', 'partial_failed'].includes(batchManifest.status)" class="batch-download" :href="getS1RedditBatchCsvDownloadUrl(batchManifest.batch_id)">下载场景指标汇总 CSV</a>
            </div>
            <button v-if="runMode === 'single' && hasRun && !running" class="text-button" type="button" @click="resetWorkbench">重新选择场景</button>
          </section>

          <div v-if="errorMessage" class="error-box"><strong>操作失败</strong><p>{{ errorMessage }}</p></div>
        </aside>

        <section class="result-panel">
          <div class="panel-toolbar">
            <div class="tab-list" role="tablist">
              <button v-for="tab in tabs" :key="tab.id" type="button" :class="{ active: activeTab === tab.id }" @click="activeTab = tab.id">
                {{ tab.label }}<span v-if="tab.id === 'results'">{{ predictions.length }}</span>
              </button>
            </div>
            <button class="refresh-button" type="button" :disabled="!hasRun || refreshing" @click="refreshRun">{{ refreshing ? '刷新中…' : '刷新' }}</button>
          </div>

          <div v-if="activeTab === 'overview'" class="panel-content overview-view">
            <div class="progress-block">
              <div class="progress-copy"><div><p class="content-kicker">REDDIT SOCIAL INTERACTION</p><h2>{{ progressTitle }}</h2></div><strong>{{ progressPercent }}%</strong></div>
              <div class="progress-track"><span :style="{ width: `${progressPercent}%` }"></span></div>
              <div class="direction-summary"><span>历史记忆 {{ mapping?.history_memory?.length || 0 }} 条</span><span>互动 {{ manifest?.social_rounds || socialRounds }} 轮</span><span>Pre / Post 各 20 次预测</span></div>
            </div>
            <div class="section-grid">
              <section class="mapping-card">
                <div class="content-header"><div><p class="content-kicker">ENTITY → ACCOUNT</p><h2>信息源映射</h2></div></div>
                <div v-if="mapping?.publishers?.length" class="source-list">
                  <div v-for="source in mapping.publishers" :key="source.source_entity_id" class="source-row">
                    <span class="source-id">A{{ String(Number(source.agent_id) + 1).padStart(2, '0') }}</span>
                    <div><strong>{{ source.name }}</strong><small>{{ source.source_type }} · {{ sourceOriginLabel(source.source_origin) }}</small></div>
                  </div>
                </div>
                <div v-else class="empty-state compact"><p>准备后显示实体到 Reddit 账号的映射。</p></div>
              </section>
              <section class="mapping-card timeline-card">
                <div class="content-header"><div><p class="content-kicker">MEMORY + CURRENT EVENT</p><h2>Agent 实际读取的信息</h2></div></div>
                <ol v-if="mapping?.history_memory?.length" class="event-list">
                  <li v-for="(event, index) in mapping.history_memory" :key="event.event_id">
                    <span class="round-badge">M{{ index + 1 }}</span>
                    <div><strong>互动前只读历史记忆</strong><p>{{ event.text }}</p></div>
                  </li>
                  <li v-if="mapping.current_event"><span class="round-badge">NOW</span><div><strong>当前公开帖 · {{ mapping.current_event.publisher_name }}</strong><p>{{ mapping.current_event.text }}</p></div></li>
                </ol>
                <div v-else class="empty-state compact"><p>准备后显示历史记忆和当前公开事件。</p></div>
              </section>
            </div>
          </div>

          <div v-else-if="activeTab === 'results'" class="panel-content result-view">
            <div class="progress-block"><div class="progress-copy"><div><p class="content-kicker">PRE / POST COMPARISON</p><h2>{{ progressTitle }}</h2></div><strong>{{ progressPercent }}%</strong></div><div class="progress-track"><span :style="{ width: `${progressPercent}%` }"></span></div><div class="direction-summary"><span>有效配对 {{ pairedPredictions.length }}</span><span class="up">改判 {{ changedCount }}</span><span>改判率 {{ formatProbability(metrics?.group_change?.direction_flip_rate) }}</span><span class="failed">异常 {{ failedCount }}</span></div></div>
            <div v-if="hasRun && predictions.length" class="csv-actions"><a :href="csvUrl('predictions')">下载两阶段预测</a><a :href="csvUrl('agent_changes')">下载 Agent 变化</a><a v-if="manifest?.status === 'completed'" :href="csvUrl('evaluation')">下载评测</a></div>
            <div v-if="pairedPredictions.length" class="table-wrap"><table><thead><tr><th>Agent</th><th>角色</th><th>Pre 方向</th><th>Post 方向</th><th>是否改判</th><th>预期收益变化</th><th>置信度变化</th><th>社交动作</th></tr></thead><tbody><tr v-for="pair in pairedPredictions" :key="pair.agent_id"><td class="mono">{{ formatAgentId(pair.agent_id) }}</td><td>{{ pair.role }}</td><td><span class="direction-badge" :class="pair.pre?.direction">{{ directionLabel(pair.pre?.direction) }}</span></td><td><span class="direction-badge" :class="pair.post?.direction">{{ directionLabel(pair.post?.direction) }}</span></td><td>{{ pair.changed ? '是' : '否' }}</td><td>{{ formatDelta(pair.returnDelta, true) }}</td><td>{{ formatDelta(pair.confidenceDelta) }}</td><td>{{ pair.post?.social_action_count || 0 }}</td></tr></tbody></table></div><div v-else class="empty-state compact"><strong>尚无配对结果</strong><p>完整运行后会同时显示互动前和互动后的预测。</p></div>
          </div>

          <div v-else-if="activeTab === 'metrics'" class="panel-content">
            <div v-if="metrics?.group_change" class="metric-grid">
              <div><span>方向改判率</span><strong>{{ formatProbability(metrics.group_change.direction_flip_rate) }}</strong></div>
              <div><span>平均观点变化 JS</span><strong>{{ formatNumber(metrics.group_change.mean_distribution_js_divergence) }}</strong></div>
              <div><span>Pre 共识率</span><strong>{{ formatProbability(metrics.pre_social?.consensus_rate) }}</strong></div>
              <div><span>Post 共识率</span><strong>{{ formatProbability(metrics.post_social?.consensus_rate) }}</strong></div>
              <div><span>Pre 群体熵</span><strong>{{ formatNumber(metrics.pre_social?.direction_entropy_bits) }}</strong></div>
              <div><span>Post 群体熵</span><strong>{{ formatNumber(metrics.post_social?.direction_entropy_bits) }}</strong></div>
            </div>
            <div v-if="actions.length" class="table-wrap action-table"><table><thead><tr><th>轮次</th><th>Agent</th><th>动作</th><th>时间</th><th>动作内容</th></tr></thead><tbody><tr v-for="action in actions" :key="action.trace_id"><td>R{{ action.round }}</td><td>{{ action.agent_class === 'investor' ? formatAgentId(action.agent_id) : '信息源' }}</td><td>{{ action.action_type }}</td><td class="mono">{{ action.timestamp }}</td><td class="action-detail">{{ actionSummary(action) }}</td></tr></tbody></table></div>
            <div v-else class="empty-state compact"><strong>尚无互动记录</strong><p>完成后端模拟后，这里展示最近 100 条 OASIS 原始动作；完整轨迹保存在 JSONL。</p></div>
          </div>

          <div v-else class="panel-content log-view"><div class="content-header"><div><p class="content-kicker">LOCAL OPERATION LOG</p><h2>实验操作记录</h2></div></div><ol v-if="operationLogs.length" class="log-list"><li v-for="(entry, index) in operationLogs" :key="`${entry.time}-${index}`"><time>{{ entry.time }}</time><span>{{ entry.message }}</span></li></ol><div v-else class="empty-state compact"><p>还没有操作记录。</p></div></div>
        </section>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  getS1RedditCsvDownloadUrl,
  getS1RedditActions,
  getS1RedditBatchStatus,
  getS1RedditBatchCsvDownloadUrl,
  getS1RedditMapping,
  getS1RedditMetrics,
  getS1RedditPredictions,
  getS1RedditStatus,
  getS1ScenarioSeed,
  listC0Scenarios,
  prepareS1Reddit,
  prepareS1RedditBatch,
  runS1Reddit,
  runS1RedditBatch,
  updateS1RedditSettings
} from '../api/finance'
import { buildGraph, generateOntology, getProject, getTaskStatus, listProjects } from '../api/graph'

const route = useRoute()
const router = useRouter()
const STORAGE_KEY = 'mirofish_finance_s1_reddit_run_id'
const BATCH_STORAGE_KEY = 'mirofish_finance_s1_reddit_batch_id'
const runMode = ref('single')
const scenarios = ref([])
const selectedScenarioId = ref('')
const sourceMode = ref('graph')
const projectId = ref(String(route.query.projectId || ''))
const graphId = ref('')
const projects = ref([])
const loadingProjects = ref(false)
const seedDocument = ref(null)
const selectedGraphEventIds = ref([])
const graphBuilding = ref(false)
const graphProgress = ref(null)
const graphTaskId = ref('')
const socialRounds = ref(6)
const runId = ref('')
const manifest = ref(null)
const mapping = ref(null)
const predictions = ref([])
const metrics = ref(null)
const actions = ref([])
const batchManifest = ref(null)
const batchRunning = ref(false)
const activeTab = ref('overview')
const loadingScenarios = ref(false)
const preparing = ref(false)
const running = ref(false)
const updatingSettings = ref(false)
const settingsDirty = ref(false)
const refreshing = ref(false)
const errorMessage = ref('')
const operationLogs = ref([])
const terminalLogKey = ref('')
let pollTimer = null

const phases = ['历史记忆', '当前事件', 'Pre 预测', '社会互动', 'Post 预测', '比较归档']
const tabs = [{ id: 'overview', label: '实验输入' }, { id: 'results', label: 'Agent 变化' }, { id: 'metrics', label: '指标与互动' }, { id: 'logs', label: '操作记录' }]
const selectedScenario = computed(() => scenarios.value.find(item => item.scenario_id === selectedScenarioId.value))
const graphProjects = computed(() => projects.value.filter(item => item.status === 'graph_completed' && item.graph_id))
const selectedProject = computed(() => graphProjects.value.find(item => item.project_id === projectId.value))
const hasRun = computed(() => Boolean(runId.value && manifest.value))
const busy = computed(() => preparing.value || running.value || graphBuilding.value || batchRunning.value)
const settingsLocked = computed(() => preparing.value || running.value || graphBuilding.value || batchRunning.value || ['completed', 'failed'].includes(manifest.value?.status))
const allGraphEventsSelected = computed(() => Boolean(seedDocument.value?.events?.length) && selectedGraphEventIds.value.length === seedDocument.value.events.length)
const canPrepare = computed(() => Boolean(selectedScenarioId.value) && !hasRun.value && !busy.value && (sourceMode.value !== 'graph' || projectId.value.trim() || graphId.value.trim()))
const canBuildGraph = computed(() => sourceMode.value === 'graph' && Boolean(seedDocument.value?.events?.length) && allGraphEventsSelected.value && !hasRun.value && !busy.value && !projectId.value.trim() && !graphId.value.trim())
const canRun = computed(() => hasRun.value && !busy.value && !settingsDirty.value && manifest.value?.status === 'prepared')
const expectedCount = computed(() => manifest.value?.expected_prediction_count || 40)
const completedCount = computed(() => manifest.value?.completed_prediction_count ?? predictions.value.length)
const failedCount = computed(() => predictions.value.filter(item => item.status !== 'ok').length)
const sourceCount = computed(() => manifest.value?.source_agent_count || mapping.value?.publisher_account_count || 0)
const progressPercent = computed(() => expectedCount.value ? Math.min(100, Math.round(completedCount.value / expectedCount.value * 100)) : 0)
const pairedPredictions = computed(() => {
  const pre = new Map(predictions.value.filter(item => item.prediction_stage === 'pre_social').map(item => [Number(item.agent_id), item]))
  const post = new Map(predictions.value.filter(item => item.prediction_stage === 'post_social').map(item => [Number(item.agent_id), item]))
  return [...new Set([...pre.keys(), ...post.keys()])].sort((a, b) => a - b).map(agentId => {
    const before = pre.get(agentId); const after = post.get(agentId)
    return { agent_id: agentId, role: before?.agent_role_label || after?.agent_role_label, pre: before, post: after, changed: Boolean(before?.status === 'ok' && after?.status === 'ok' && before.direction !== after.direction), returnDelta: numericDelta(before?.expected_return, after?.expected_return), confidenceDelta: numericDelta(before?.confidence, after?.confidence) }
  })
})
const changedCount = computed(() => pairedPredictions.value.filter(item => item.changed).length)
const batchTotal = computed(() => batchManifest.value?.scenario_count || 0)
const batchCompleted = computed(() => batchManifest.value?.completed_scenario_count || 0)
const phaseLevel = computed(() => { if (manifest.value?.status === 'completed') return 6; if (manifest.value?.current_phase === 'social_interaction') return 4; if (manifest.value?.status === 'running') return 3; if (hasRun.value) return 2; return 1 })
const statusLabel = computed(() => ({ prepared: '输入已冻结', running: 'Reddit 互动运行中', completed: '预测完成', failed: '运行中断' }[manifest.value?.status] || (loadingScenarios.value ? '读取数据中' : '等待准备')))
const statusTone = computed(() => manifest.value?.status === 'completed' ? 'completed' : (manifest.value?.status === 'failed' || errorMessage.value ? 'error' : (running.value ? 'running' : 'idle')))
const progressTitle = computed(() => manifest.value?.status === 'completed' ? '20 个 Agent 的两阶段预测已完成配对' : manifest.value?.status === 'running' ? `正在执行：${phaseLabel(manifest.value.current_phase)}` : '等待启动社会互动实验')

const phaseClass = level => ({ active: phaseLevel.value === level, complete: phaseLevel.value > level })
const addLog = message => operationLogs.value.unshift({ time: new Date().toLocaleTimeString('zh-CN', { hour12: false }), message })
const loadScenarios = async () => { loadingScenarios.value = true; try { const response = await listC0Scenarios(); scenarios.value = response.data || []; if (!selectedScenarioId.value && scenarios.value.length) selectedScenarioId.value = scenarios.value[0].scenario_id; addLog(`已读取 ${scenarios.value.length} 个匿名候选场景。`) } catch (error) { errorMessage.value = error.message } finally { loadingScenarios.value = false } }
const loadProjects = async () => { loadingProjects.value = true; try { const response = await listProjects(); projects.value = response.data || []; addLog(`已读取 ${graphProjects.value.length} 个已完成图谱项目。`) } catch (error) { addLog(`项目列表读取失败：${error.message}`) } finally { loadingProjects.value = false } }
const loadSeedDocument = async () => { if (!selectedScenarioId.value || hasRun.value) return; try { const response = await getS1ScenarioSeed(selectedScenarioId.value); seedDocument.value = response.data; selectedGraphEventIds.value = (response.data?.events || []).map(event => event.event_id) } catch (error) { errorMessage.value = error.message; addLog(`事件读取失败：${error.message}`) } }
const buildGraphForScenario = async () => {
  if (!allGraphEventsSelected.value) { errorMessage.value = 'S1 正式实验需要将 5 条历史种子和当前事件全部纳入图谱。'; return }
  graphBuilding.value = true; errorMessage.value = ''; graphProgress.value = { progress: 0, message: '正在准备匿名事件文本…' }
  try {
    const selectedEvents = seedDocument.value.events.filter(event => selectedGraphEventIds.value.includes(event.event_id))
    const documentText = selectedEvents.map(event => `[${event.phase === 'current' ? 'Current public event' : `Historical event ${event.seed_rank}`} at ${event.event_time}]\n${event.text}`).join('\n\n')
    const formData = new FormData()
    formData.append('files', new File([documentText], `${selectedScenarioId.value}_s1_events.txt`, { type: 'text/plain' }))
    formData.append('simulation_requirement', '构建匿名 A 股社会互动实验图谱：识别事件发布者、上市公司、财经媒体、交易所、监管机构、股东及其关系。实体名称必须保留匿名代号，不能猜测真实身份。')
    formData.append('project_name', `S1 ${selectedScenarioId.value} 事件图谱`)
    formData.append('additional_context', '这是 S1 Reddit 社会互动实验的冻结事件材料。只处理页面选中的事件，不包含评测答案、未来价格、stock_factors 或 evaluator 字段。')
    graphProgress.value = { progress: 5, message: '正在使用 MiroFish 原有流程生成图谱本体…' }
    const ontologyResponse = await generateOntology(formData)
    projectId.value = ontologyResponse.data.project_id
    graphProgress.value = { progress: 15, message: '本体完成，正在向 Zep 写入匿名事件…' }
    const buildResponse = await buildGraph({ project_id: projectId.value, graph_name: `S1 ${selectedScenarioId.value} Event Graph` })
    if (buildResponse.data?.reused && buildResponse.data.graph_id) { graphId.value = buildResponse.data.graph_id; graphProgress.value = { progress: 100, message: '图谱已复用。' } } else { await pollGraphTask(buildResponse.data.task_id) }
    const projectResponse = await getProject(projectId.value)
    graphId.value = projectResponse.data.graph_id || graphId.value
    await loadProjects()
    addLog(`图谱构建完成：${projectId.value} / ${graphId.value}`)
  } catch (error) { errorMessage.value = error.message; graphProgress.value = null; addLog(`图谱构建失败：${error.message}`); projectId.value = '' }
  finally { graphBuilding.value = false }
}
const pollGraphTask = (taskId) => new Promise((resolve, reject) => {
  graphTaskId.value = taskId
  const poll = async () => {
    try {
      const response = await getTaskStatus(taskId); const task = response.data
      graphProgress.value = { progress: task.progress || 0, message: task.message || '图谱处理中…' }
      if (task.status === 'completed') { graphProgress.value = { progress: 100, message: '图谱构建完成。' }; resolve(task); return }
      if (task.status === 'failed') { reject(new Error(task.error || '图谱构建失败')); return }
      window.setTimeout(poll, 2000)
    } catch (error) { reject(error) }
  }
  poll()
})
const loadRunArtifacts = async () => {
  if (!runId.value) return
  const [mappingResponse, predictionResponse, metricsResponse, actionsResponse] = await Promise.all([
    getS1RedditMapping(runId.value), getS1RedditPredictions(runId.value, 'all'),
    getS1RedditMetrics(runId.value), getS1RedditActions(runId.value, 100)
  ])
  mapping.value = mappingResponse.data; predictions.value = predictionResponse.data || []
  metrics.value = metricsResponse.data || null; actions.value = actionsResponse.data || []
}
const applySettings = async () => {
  if (!runId.value || !settingsDirty.value) return
  updatingSettings.value = true
  errorMessage.value = ''
  try {
    const response = await updateS1RedditSettings(runId.value, socialRounds.value)
    manifest.value = response.data
    socialRounds.value = Number(response.data.social_rounds)
    settingsDirty.value = false
    addLog(`已应用互动参数：${socialRounds.value} 轮。`)
  } catch (error) {
    errorMessage.value = error.message
    addLog(`互动参数应用失败：${error.message}`)
  } finally {
    updatingSettings.value = false
  }
}
const prepareExperiment = async () => { preparing.value = true; errorMessage.value = ''; try { const response = await prepareS1Reddit({ scenarioId: selectedScenarioId.value, projectId: projectId.value, graphId: graphId.value, sourceMode: sourceMode.value, socialRounds: socialRounds.value }); manifest.value = response.data; runId.value = response.data.run_id; localStorage.setItem(STORAGE_KEY, runId.value); await loadRunArtifacts(); addLog(`输入已冻结：5 条历史事件进入只读记忆，当前事件将作为初始公开帖，随后进行 ${socialRounds.value} 轮互动。`); activeTab.value = 'overview' } catch (error) { errorMessage.value = error.message; addLog(`准备失败：${error.message}`) } finally { preparing.value = false } }
const executeExperiment = async () => { running.value = true; errorMessage.value = ''; activeTab.value = 'overview'; addLog('Reddit 社会互动已启动，前端将持续轮询后端状态。'); try { const response = await runS1Reddit(runId.value); manifest.value = response.data; startPolling() } catch (error) { running.value = false; errorMessage.value = error.message; addLog(`启动失败：${error.message}`) } }
const refreshRun = async (showSpinner = true) => { if (!runId.value) return; if (showSpinner) refreshing.value = true; try { const statusResponse = await getS1RedditStatus(runId.value); manifest.value = statusResponse.data; await loadRunArtifacts(); if (manifest.value.status === 'completed') { running.value = false; stopPolling(); const key = `${runId.value}:completed`; if (terminalLogKey.value !== key) { terminalLogKey.value = key; addLog(`实验完成：20 个 Agent，成功 ${manifest.value.successful_prediction_count || 0} 个。`) } } else if (manifest.value.status === 'failed') { running.value = false; stopPolling(); errorMessage.value = manifest.value.error || '后台运行已中断'; addLog(`实验中断：${errorMessage.value}`) } } catch (error) { if (showSpinner) errorMessage.value = error.message } finally { if (showSpinner) refreshing.value = false } }
const startPolling = () => { stopPolling(); pollTimer = window.setInterval(() => refreshRun(false), 1500) }
const stopPolling = () => { if (pollTimer) window.clearInterval(pollTimer); pollTimer = null }
const restoreRun = async () => { const storedRunId = localStorage.getItem(STORAGE_KEY); if (!storedRunId) return; runId.value = storedRunId; try { const response = await getS1RedditStatus(storedRunId); manifest.value = response.data; sourceMode.value = manifest.value.source_mode === 'graph' ? 'graph' : 'scenario'; await loadRunArtifacts(); if (['prepared', 'running'].includes(manifest.value.status)) { running.value = manifest.value.status === 'running'; if (running.value) startPolling() } addLog(`已恢复运行 ${storedRunId}。`) } catch { localStorage.removeItem(STORAGE_KEY); runId.value = '' } }
const runAllScenarios = async () => {
  batchRunning.value = true; errorMessage.value = ''; activeTab.value = 'overview'
  try {
    const prepared = await prepareS1RedditBatch(socialRounds.value)
    batchManifest.value = prepared.data
    localStorage.setItem(BATCH_STORAGE_KEY, prepared.data.batch_id)
    const started = await runS1RedditBatch(prepared.data.batch_id)
    batchManifest.value = started.data
    addLog(`批量任务已启动：${prepared.data.scenario_count} 个已构图场景，将逐个串行运行。`)
    startBatchPolling()
  } catch (error) { batchRunning.value = false; errorMessage.value = error.message; addLog(`批量启动失败：${error.message}`) }
}
const refreshBatch = async () => {
  if (!batchManifest.value?.batch_id) return
  try {
    const response = await getS1RedditBatchStatus(batchManifest.value.batch_id)
    batchManifest.value = response.data
    if (['completed', 'partial_failed', 'failed'].includes(response.data.status)) { batchRunning.value = false; stopPolling(); addLog(`批量任务结束：完成 ${response.data.completed_scenario_count}，失败 ${response.data.failed_scenario_count}。`) }
  } catch (error) { errorMessage.value = error.message }
}
const startBatchPolling = () => { stopPolling(); pollTimer = window.setInterval(refreshBatch, 2000) }
const restoreBatch = async () => {
  const batchId = localStorage.getItem(BATCH_STORAGE_KEY); if (!batchId) return
  try { const response = await getS1RedditBatchStatus(batchId); batchManifest.value = response.data; batchRunning.value = ['queued', 'running'].includes(response.data.status); if (batchRunning.value) { runMode.value = 'all'; startBatchPolling() } } catch { localStorage.removeItem(BATCH_STORAGE_KEY) }
}
const resetWorkbench = () => { stopPolling(); localStorage.removeItem(STORAGE_KEY); runId.value = ''; manifest.value = null; mapping.value = null; predictions.value = []; metrics.value = null; actions.value = []; terminalLogKey.value = ''; errorMessage.value = ''; activeTab.value = 'overview'; addLog('工作台已重置；历史运行文件仍保留在磁盘。') }
const directionLabel = direction => ({ up: '上涨', neutral: '中性', down: '下跌' }[direction] || '—')
const sourceOriginLabel = origin => ({ zep_graph: 'Zep 图谱实体', scenario_text: '匿名事件文本', public_feed: '统一公开信息流' }[origin] || origin || '—')
const sourceModeLabel = mode => mode === 'graph' ? 'Zep 图谱' : '文本回退'
const formatProbability = value => value === null || value === undefined ? '—' : `${Math.round(Number(value) * 100)}%`
const formatNumber = value => value === null || value === undefined ? '—' : Number(value).toFixed(3)
const numericDelta = (before, after) => Number.isFinite(Number(before)) && Number.isFinite(Number(after)) && before !== null && after !== null ? Number(after) - Number(before) : null
const formatDelta = (value, asPercent = false) => value === null || value === undefined ? '—' : `${Number(value) >= 0 ? '+' : ''}${asPercent ? `${(Number(value) * 100).toFixed(2)}%` : Number(value).toFixed(3)}`
const phaseLabel = phase => ({ pre_social_prediction: '互动前预测', social_interaction: '社会互动', completed: '结果归档' }[phase] || '准备环境')
const actionSummary = action => { const args = action.action_args || {}; return args.content || args.query || args.post_id || args.comment_id || '—' }
const formatAgentId = value => `A${String(Number(value) + 1).padStart(2, '0')}`
const csvUrl = kind => getS1RedditCsvDownloadUrl(runId.value, kind)

onMounted(async () => { await Promise.all([loadScenarios(), loadProjects()]); await Promise.all([restoreRun(), restoreBatch()]) })
watch(selectedScenarioId, () => { if (!hasRun.value) loadSeedDocument() })
watch(manifest, value => {
  if (!value) {
    settingsDirty.value = false
    return
  }
  if (value.status === 'prepared' && !settingsDirty.value) {
    socialRounds.value = Number(value.social_rounds || 6)
  }
})
onBeforeUnmount(stopPolling)
</script>

<style scoped>
.s1-page { min-height: 100vh; color: #17211d; background: #f4f6f5; font-family: "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif; }
button, select, input { font: inherit; }
button { letter-spacing: 0; }
.topbar { height: 60px; display: grid; grid-template-columns: 180px 1fr auto; align-items: center; gap: 24px; padding: 0 32px; color: #fff; background: #111714; border-bottom: 1px solid #324039; }
.brand-button { width: max-content; padding: 0; color: #fff; background: transparent; border: 0; font-family: "JetBrains Mono", monospace; font-weight: 800; cursor: pointer; }
.route-title { display: flex; align-items: baseline; gap: 12px; font-size: 13px; color: #9fafaa; }.route-title strong { color: #fff; font-size: 14px; }
.header-status { display: flex; align-items: center; gap: 9px; font-size: 13px; }.status-dot { width: 8px; height: 8px; background: #97a19d; }.status-dot.running { background: #e8a33d; animation: pulse 1.3s infinite; }.status-dot.completed { background: #33a06f; }.status-dot.error { background: #c94d42; }
main { width: min(1480px, calc(100% - 56px)); margin: 0 auto; padding: 30px 0 44px; }.intro-band { display: flex; justify-content: space-between; align-items: end; gap: 36px; padding-bottom: 27px; border-bottom: 1px solid #cbd2ce; }.eyebrow, .content-kicker { margin: 0 0 8px; color: #2c7659; font: 700 11px/1.4 "JetBrains Mono", monospace; }h1 { margin: 0; font-size: 31px; line-height: 1.3; }.intro-copy { max-width: 720px; margin: 10px 0 0; color: #5d6964; font-size: 15px; }
.facts { display: grid; grid-template-columns: repeat(4, 112px); border: 1px solid #cbd2ce; background: #fff; }.facts div { min-height: 74px; padding: 12px 14px; border-right: 1px solid #dce1de; }.facts div:last-child { border-right: 0; }.facts strong { display: block; font: 700 21px/1.2 "JetBrains Mono", monospace; }.facts span { display: block; margin-top: 5px; color: #65706c; font-size: 11px; }
.phase-strip { display: grid; grid-template-columns: repeat(6, 1fr); margin: 22px 0; padding: 0; list-style: none; border: 1px solid #cbd2ce; background: #fff; }.phase-strip li { min-height: 54px; display: flex; align-items: center; gap: 11px; padding: 10px 16px; color: #7d8783; border-right: 1px solid #dce1de; }.phase-strip li:last-child { border-right: 0; }.phase-strip span { font: 700 11px "JetBrains Mono", monospace; }.phase-strip b { font-size: 13px; }.phase-strip li.active { color: #18241f; background: #fff4df; box-shadow: inset 0 -3px #d98924; }.phase-strip li.complete { color: #28664f; background: #eef7f2; }
.workspace { min-height: 660px; display: grid; grid-template-columns: 350px minmax(0, 1fr); border: 1px solid #bdc6c1; background: #fff; }.control-panel { border-right: 1px solid #bdc6c1; background: #fafbfa; }.control-section { padding: 23px 24px; border-bottom: 1px solid #d9dfdc; }.section-heading { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }.section-heading span { color: #b66c1c; font: 700 11px "JetBrains Mono", monospace; }.section-heading h2, .content-header h2, .progress-copy h2 { margin: 0; font-size: 16px; line-height: 1.4; }.field-label { display: block; margin: 0 0 7px; color: #55615c; font-size: 12px; font-weight: 700; }.spaced { margin-top: 14px; }select, input { width: 100%; height: 40px; box-sizing: border-box; padding: 0 11px; color: #1a2520; background: #fff; border: 1px solid #aeb9b3; border-radius: 3px; }.mode-control { display: grid; grid-template-columns: 1fr 1fr; margin-bottom: 13px; }.mode-control button { padding: 9px 7px; color: #52605a; background: #fff; border: 1px solid #adb8b2; cursor: pointer; }.mode-control button + button { border-left: 0; }.mode-control button.active { color: #fff; background: #2d7658; border-color: #2d7658; }.mode-help, .hint, .muted-copy { color: #69756f; font-size: 12px; line-height: 1.6; }.hint.warning { color: #a55e16; }.scenario-summary { margin-top: 16px; padding: 12px; background: #f1f5f2; border-left: 3px solid #7aa993; }.scenario-summary dl, .run-meta dl { display: grid; gap: 6px; margin: 0; }.scenario-summary dl div, .run-meta dl div { display: flex; justify-content: space-between; gap: 8px; }.scenario-summary dt, .run-meta dt { color: #77837d; font-size: 11px; }.scenario-summary dd, .run-meta dd { margin: 0; font: 12px "JetBrains Mono", monospace; }.scenario-summary p { margin: 11px 0 0; color: #45534c; font-size: 12px; line-height: 1.6; }.primary-button, .run-button { width: 100%; margin-top: 17px; padding: 11px 13px; color: #fff; border: 0; cursor: pointer; }.primary-button { background: #2d7658; }.run-button { background: #c8751d; }.primary-button:disabled, .run-button:disabled { opacity: .45; cursor: not-allowed; }.run-meta { display: grid; gap: 9px; padding: 11px; margin-bottom: 8px; background: #eef3f0; }.run-meta > span { color: #77837d; font: 10px "JetBrains Mono", monospace; }.run-meta code { overflow-wrap: anywhere; color: #2b5e49; font-size: 11px; }.text-button, .refresh-button { padding: 8px 0; color: #2d7658; background: transparent; border: 0; cursor: pointer; }.text-button { display: block; width: 100%; }.error-box { margin: 20px 24px; padding: 12px; color: #843a35; background: #fff0ee; border: 1px solid #e2b2ac; font-size: 12px; }.error-box p { margin: 6px 0 0; line-height: 1.5; }
.result-panel { min-width: 0; }.panel-toolbar { display: flex; align-items: center; justify-content: space-between; min-height: 57px; padding: 0 22px; border-bottom: 1px solid #d9dfdc; }.tab-list { display: flex; gap: 20px; }.tab-list button { padding: 19px 0 15px; color: #75817b; background: transparent; border: 0; border-bottom: 3px solid transparent; cursor: pointer; }.tab-list button.active { color: #1d2d25; border-bottom-color: #d98924; }.tab-list span { display: inline-block; min-width: 18px; margin-left: 6px; padding: 2px 5px; color: #fff; background: #75817b; border-radius: 10px; font-size: 10px; }.panel-content { padding: 25px 28px; }.progress-block { padding-bottom: 20px; border-bottom: 1px solid #dfe4e1; }.progress-copy { display: flex; justify-content: space-between; align-items: end; gap: 20px; }.progress-copy > strong { color: #2d7658; font: 700 25px "JetBrains Mono", monospace; }.progress-track { height: 8px; margin: 14px 0 10px; overflow: hidden; background: #e0e6e2; }.progress-track span { display: block; height: 100%; background: #d98924; transition: width .3s ease; }.direction-summary { display: flex; gap: 18px; color: #6f7b75; font-size: 12px; }.direction-summary .up { color: #bd554e; }.direction-summary .neutral { color: #a27620; }.direction-summary .down { color: #367c62; }.direction-summary .failed { color: #a3473f; }.section-grid { display: grid; grid-template-columns: minmax(240px, .75fr) minmax(360px, 1.25fr); gap: 18px; margin-top: 22px; }.mapping-card { min-width: 0; padding: 17px; border: 1px solid #d3dbd6; background: #fbfcfb; }.content-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px; }.source-list { display: grid; gap: 9px; }.source-row { display: flex; align-items: center; gap: 10px; padding: 10px; background: #f0f5f2; }.source-id, .round-badge { flex: 0 0 auto; color: #2d7658; font: 700 11px "JetBrains Mono", monospace; }.source-row strong { display: block; font-size: 13px; }.source-row small { color: #738079; font-size: 11px; }.event-list { display: grid; gap: 12px; margin: 0; padding: 0; list-style: none; }.event-list li { display: flex; gap: 12px; }.round-badge { padding-top: 2px; }.event-list strong { display: block; font-size: 12px; }.event-list p { max-width: 680px; margin: 4px 0 0; color: #65716b; font-size: 11px; line-height: 1.55; }.csv-actions { display: flex; align-items: center; gap: 13px; margin: 18px 0; color: #68746e; font-size: 11px; }.csv-actions a { color: #2d7658; font-weight: 700; }.table-wrap { overflow: auto; border: 1px solid #d2dad5; }table { width: 100%; min-width: 800px; border-collapse: collapse; font-size: 12px; }th, td { padding: 10px 9px; text-align: left; border-bottom: 1px solid #e0e5e2; }th { color: #6b7771; background: #f5f7f5; font-size: 11px; white-space: nowrap; }td { color: #34433a; }.mono { font-family: "JetBrains Mono", monospace; }.direction-badge, .record-status { display: inline-block; padding: 3px 7px; font-size: 11px; }.direction-badge.up { color: #a44540; background: #fbe9e6; }.direction-badge.neutral { color: #966e1d; background: #fff3d8; }.direction-badge.down { color: #2c7659; background: #e6f4ec; }.record-status.ok { color: #2d7658; }.record-status.error { color: #a44540; }.empty-state { min-height: 240px; display: grid; place-content: center; text-align: center; color: #7b8781; }.empty-state.compact { min-height: 120px; }.empty-state strong { color: #435149; }.empty-state p { margin: 7px 0 0; font-size: 12px; }.log-list { display: grid; gap: 9px; margin: 0; padding: 0; list-style: none; }.log-list li { display: flex; gap: 14px; padding: 10px 0; border-bottom: 1px solid #e0e5e2; font-size: 13px; }.log-list time { color: #87928d; font: 11px "JetBrains Mono", monospace; }.log-list span { color: #3f4c45; }
.project-hint { margin: 7px 0 0; color: #2d7658; overflow-wrap: anywhere; font: 10px "JetBrains Mono", monospace; }
.event-pick-list { display: grid; gap: 6px; max-height: 245px; overflow-y: auto; margin: 11px 0; padding-right: 3px; }
.event-pick-row { display: grid; grid-template-columns: 16px 44px 1fr; align-items: start; gap: 7px; padding: 8px; color: #47564d; background: #fff; border: 1px solid #dce3de; cursor: pointer; }
.event-pick-row:has(input:checked) { background: #eef7f2; border-color: #9ec5ae; }
.event-pick-row input { width: 15px; height: 15px; margin: 2px 0 0; }
.event-pick-round { color: #2d7658; font: 10px "JetBrains Mono", monospace; }
.event-pick-text { display: -webkit-box; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 3; font-size: 11px; line-height: 1.5; }
.build-progress { margin-top: 12px; }.build-progress .progress-track { height: 6px; margin: 0 0 6px; }.build-progress small { color: #68746e; font-size: 10px; line-height: 1.4; }
.graph-button { width: 100%; margin-top: 11px; padding: 10px 12px; color: #9a5a18; background: #fff8ed; border: 1px solid #d9a15d; cursor: pointer; font-weight: 700; }.graph-button:disabled { opacity: .45; cursor: not-allowed; }
.round-config { display: grid; gap: 13px; margin-bottom: 15px; }.round-config input, .round-config select { height: 36px; }.round-config small { display: block; margin-top: 5px; color: #7c8881; font-size: 10px; line-height: 1.45; }
.metric-grid { display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 1px; margin-bottom: 22px; background: #d5ddd8; border: 1px solid #d5ddd8; }.metric-grid div { min-height: 82px; padding: 15px; background: #f8faf8; }.metric-grid span { display: block; color: #6b7771; font-size: 11px; }.metric-grid strong { display: block; margin-top: 8px; color: #244f3d; font: 700 22px "JetBrains Mono", monospace; }.action-table table { min-width: 1050px; }.action-detail { max-width: 480px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.batch-meta { margin-top: 14px; }
.batch-download { color: #2d7658; font-size: 11px; font-weight: 700; }
@keyframes pulse { 50% { opacity: .35; } }
</style>
