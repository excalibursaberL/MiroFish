# A 股金融适配层：C0 与 Reddit S1 原型

这个目录是 MiroFish 的金融适配层。当前实现了实验计划中的独立预测基线 `C0`，以及接入 OASIS Reddit 社交主流程的 `S1` 原型。

## C0：无社会互动基线

`C0` 具备以下能力：

- 支持单场景模式，以及一次运行全部 18 个匿名场景的批量模式；
- 从完整 20 人画像池中固定筛选出的 10 个匿名投资者 Agent；
- 当前子集由 1 个机构投资者、3 个有经验散户、5 个具备基础知识的散户、1 个新手散户组成；
- 原始 ID 固定为 `1, 3, 4, 5, 9, 11, 12, 13, 14, 17`，运行时连续重编号为 `0–9`；
- 完整画像池仍按调查配额与 TwinMarket 实验先验生成，便于审计降采样来源；
- 每个 Agent 读取同一场景的 5 个历史种子和 1 个当前公开事件；
- 每个 Agent 单独调用一次模型；
- 模型返回空文本、截断文本或无效 JSON 时自动重试一次，并记录 `attempt_count`；
- 每条结果同时记录 `finish_reason`、正文字符数和 `reasoning_content_present`，便于区分空响应、token 截断和思考通道误用；
- Agent 不能看到其他 Agent 的帖子、预测、回复或聚合结果；
- C0 固定使用每个场景的 5 条历史种子；
- `stock_factors` 保留在数据集和冻结快照中，但不写入 LLM Prompt；
- 输出方向、三分类概率、预期收益、置信度、证据事件 ID 和理由。
- 批量模式会在后台依次完成 `18 × 10 = 180` 次模型调用，并持续写入 CSV。
- 五日方向采用固定的 `±1.7%` 中性区间：`R5 < -1.7%` 为下跌，`|R5| <= 1.7%` 为中性，`R5 > 1.7%` 为上涨。

`C0` 不创建 OASIS 社交环境，也不写入 Zep 社会互动记录，用来提供没有社会互动的基线。`S1` 复用同一组 Profile 和匿名场景，再将 Agent 接入帖子、评论和点赞链路。

## Agent Profile

当前画像版本为 `survey2019_twinmarket_minimal_v1`。C0 Prompt 只启用 `knowledge_level`、`analysis_style`、`risk_attitude` 和 `investment_horizon`：前三类调查属性来自投保基金 2019 年自然人投资者调查的离散配额，`analysis_style` 借用 TwinMarket 的策略类别和约 `40/60` 合成比例作为实验先验。机构画像全部标注为角色设计，不写成自然人调查结果。

`decision_source` 和 `social_role` 会保存在 `profiles.json` 中，但只供后续 S1 使用；C0 Prompt 明确禁止依据这两个字段假设已经看到社会观点。现金、持仓、历史收益、处置效应、换手率和交易次数不进入 C0。

技术型 Agent 只能使用输入中明确提供的信息；没有 K 线或命名技术指标时，Prompt 会禁止其编造均线、MACD、成交量等数据。个人投资期限也不会改变统一的预测目标，所有 Agent 都必须回答场景指定的五日 horizon。

方向标签中的 `neutral` 只表示 Agent 预计未来五日累计收益位于 `[-1.7%, +1.7%]`，不表示消息相互矛盾或 Agent 无法判断。认知上的不确定性应由较低的 `confidence` 和更分散的三分类概率表达。`expected_return` 使用小数收益率，例如 `0.02` 表示 `+2%`。阈值来源为 Astock 训练集五日绝对收益的第 20 百分位数并四舍五入为 `1.7%`；阈值在 OOD 评测前固定，不根据测试结果调整。

## 输入

默认读取：

```text
Dataset/seed5_small_blind/mirofish_inputs.jsonl
```

加载器会递归拒绝包含 `label`、`CHANGE`、未来价格、未来收益或评测字段的输入，并要求场景、资产、事件使用匿名 ID（`SCN_`、`ASSET_`、`COMPANY_`、`EVT_`）以及 `T-nd`/`T+0d` 相对时间。Agent 运行时只能挂载匿名数据目录，不能挂载 `Dataset/reports/` 或原始 `Dataset/seed5_small/`。

## API

后端注册了 `/api/finance` 蓝图。

前端工作台地址：

```text
http://localhost:3000/finance/c0
```

页面会依次完成运行范围选择、输入冻结、Prompt dry-run、Agent 正式运行和结果展示。选择“全部场景”后，正式运行按钮会一键启动 18 个场景、180 次预测。首页右上角也提供了“A股 C0 实验”入口。

获取可选的匿名场景：

```text
GET /api/finance/c0/scenarios
```

先准备快照、角色、Profile 和 Prompt（不调用 LLM）：

```json
POST /api/finance/c0/prepare
{
  "scenario_ids": ["SCN_001"]
}
```

准备全部场景时不要传 `scenario_ids` 或 `limit`：

```json
POST /api/finance/c0/prepare
{
  "run_mode": "all"
}
```

返回的 `run_id` 对应目录：

```text
MiroFish/backend/uploads/finance/<run_id>/
```

主要文件：

- `manifest.json`：运行组别、数据集和文件状态；
- `profiles.json`：10 个 OASIS 兼容 Profile；`user_id` 是 `0–9` 的运行时 ID，`full_population_agent_id` 是其在原 20 人池中的 ID；
- `scenarios.jsonl`：冻结后的安全输入快照；
- `prompts.jsonl`：每个场景/Agent 的独立 Prompt；
- `predictions.jsonl`：真实运行后的结构化预测。
- `predictions.csv`：逐 Agent 预测，不包含真实答案；同时记录知识水平、分析方式、风险态度、投资期限和画像版本，运行中每完成一个 Agent 就会刷新；
- `evaluation.csv`：预测与隐藏真实结果的连接表，只在全部 Agent 完成后生成，供研究者统计准确率和收益误差；
- `llm_responses.jsonl`：每次实际 API 尝试的完整请求和响应，包括重试、响应 ID、choices、content、reasoning_content、finish_reason、usage 和异常信息，不包含 API Key；
- `llm_token_usage.jsonl`：每次模型调用由供应商返回的原始 token 用量；
- `agent_token_usage.csv`：逐 Agent 汇总的输入、输出、总 token 与分阶段用量；
- `token_usage_summary.json`：整次实验及各阶段的 token 汇总。缺失的供应商 usage 会明确计数，不使用文本长度估算。

`llm_responses.jsonl` 用于研究者排查供应商响应问题，其中可能包含完整 Prompt 和模型返回内容，不应提供给 Agent，也不应公开发布。

先只检查 Prompt，不调用模型：

```json
POST /api/finance/c0/run
{
  "run_id": "c0_xxxxxxxxxxxx",
  "dry_run": true
}
```

确认后运行 C0：

```json
POST /api/finance/c0/run
{
  "run_id": "c0_xxxxxxxxxxxx"
}
```

全部场景必须使用后台模式，接口会立即返回 `202`，随后由页面轮询进度：

```json
POST /api/finance/c0/run
{
  "run_id": "c0_xxxxxxxxxxxx",
  "background": true
}
```

批量运行期间可以关闭前端页面。后端进程停止或重启时，正在执行的任务会先显示为“中断”；重新打开页面后可点击“继续剩余预测”，已经写入 `predictions.jsonl` 的结果不会重复调用。180 次调用的时间与费用取决于所选模型；正式启动前应先核对模型价格、API 额度和 Prompt dry-run。

查询状态：

```text
GET /api/finance/c0/<run_id>
```

查看 Prompt 预览和逐 Agent 预测结果：

```text
GET /api/finance/c0/<run_id>/preview
GET /api/finance/c0/<run_id>/predictions
```

下载 CSV：

```text
GET /api/finance/c0/<run_id>/csv/predictions
GET /api/finance/c0/<run_id>/csv/evaluation
```

第二个地址在整批完成前返回 `404`，用于避免评测真值过早进入实验流程。

实验完成后，研究者可读取隐藏的真实结果：

```text
GET /api/finance/c0/<run_id>/outcome
```

该接口同时返回 Astock 的原始 `label/CHANGE` 口径和未来 5 个交易日的端到端收盘收益口径。后者通过 `R5 = close5 / original_price - 1` 计算，并使用固定 `±1.7%` 中性区间。`manifest.json`、outcome 和 `evaluation.csv` 都会记录本次使用的阈值与定义。评测文件由独立 evaluator 读取，不会写入 Prompt、Profile、Zep 或 Agent 场景快照。

单场景 API 仍采用同步执行；全部场景使用当前后端进程内的后台线程顺序执行。预测 JSONL 使用单写入者追加模式，manifest 和 CSV 使用唯一临时文件及 Windows 文件占用重试，避免页面轮询或编辑器读取时触发 `WinError 5`。后端重启会将未完成任务标记为中断，研究者可手动继续已冻结的批量运行；当前仍不是自动任务队列。C0 的 JSON 协议调用会关闭 DeepSeek V4 思考模式，并在空响应或解析失败时重试一次。

## S1：Reddit 社会互动原型

S1 固定运行一个匿名场景，包含同一组 10 个参与预测的投资者 Agent，以及从当前场景动态解析出的信息发布账号。OASIS Reddit 会按 Profile 列表位置分配 Agent ID，因此投资者固定为 `0–9`；信息主体根据事件中实际出现的发布者从 `10` 开始连续编号，数量不再固定。

一次实验执行固定顺序：`历史记忆预载 -> 当前事件公开 -> pre-social 信念测量 -> 每轮社会互动及信念快照 -> 最后一轮快照作为 post-social -> 配对比较`。5 条历史种子直接写入每个投资者的只读 Profile 记忆，不再作为帖子逐轮发布，也不占互动轮数；只有当前事件由图谱解析出的信息主体发布为初始公开帖。默认社会互动为 6 轮，可在 `1–12` 轮之间设置。每轮只表示一次互动机会，不映射成现实中的分钟或交易日。这里的“五日”仅指预测目标是未来 5 个交易日累计收盘收益。

pre-social 和每轮快照都对同一套 OASIS Agent 使用逐字相同、阶段中性的私有测量提示词；轮次只写入实验元数据，不进入提示词。每次测量的无效 JSON 会立即重试一次。最后一轮快照直接派生 `post_social_predictions.jsonl`，不会再发起语义重复的 post-social LLM 采访。评测真值仍然只在全部预测完成后读取。

`random_seed` 默认固定为 `4004`。启动 OASIS 子进程前设置 `PYTHONHASHSEED`，子进程内固定 Python、NumPy 和 PyTorch 随机源，并把实际状态写入 `random_seed_state.json`。这可以固定本地 Agent 激活、推荐抽样和调度随机性；DeepSeek API 没有可依赖的确定性 seed 合约，因此相同输入的模型文本仍可能出现小幅差异，不能声称端到端逐 token 可复现。

正式实验推荐传入已经由 MiroFish 种子材料流程建好的 `project_id` 或 `graph_id`。S1 会复用 `ZepEntityReader` 读取图谱实体，把事件中的发布者匹配到实体 UUID，再生成受系统控制、不参与预测的信息账号。同一实体发布多条事件时只生成一个账号；事件中被提及但没有直接发布信息的实体只进入映射记录，不会自动成为发布者。图谱模式下，只要文本能够识别出发布者但图谱里找不到对应实体，准备阶段就会直接报错，避免悄悄退回虚构账号后仍被误认为正式实验。

如果没有提供图谱，`source_mode=auto` 会进入明确标记的 `scenario` 回退模式，根据“某公司公告”“某媒体电”等归因语言解析发布者。无法归因的事件统一交给一个 `PUBLIC_DISCLOSURE_FEED`，不会再人为补齐公司、媒体、交易所和监管机构四类账号。回退模式适合离线调试；论文正式结果应使用图谱模式。

准备一个 S1 场景（不调用 LLM）：

```json
POST /api/finance/s1/reddit/prepare
{
  "scenario_id": "SCN_009",
  "project_id": "proj_xxxxxxxxxxxx",
  "source_mode": "graph",
  "social_rounds": 6
}
```

也可以直接传入该项目对应的 `graph_id`：

```json
POST /api/finance/s1/reddit/prepare
{
  "scenario_id": "SCN_009",
  "graph_id": "mirofish_xxxxxxxxxxxxxxxx",
  "source_mode": "graph"
}
```

只做离线流程测试时可以省略项目/图谱，或者显式指定 `"source_mode": "scenario"`。`auto` 在提供图谱时选择 `graph`，否则选择 `scenario`；最终选择会写入 manifest，避免混淆两类实验。

读取某个匿名场景的安全事件材料（不包含 `stock_factors`、未来价格或评测答案）：

```text
GET /api/finance/s1/reddit/scenarios/<scenario_id>/seed
```

前端 S1 工作台会自动调用该接口展示 5 条历史种子和 1 条当前事件。点击“从当前事件新建图谱”时，页面将选中的事件整理为临时匿名文本，调用 MiroFish 原有的本体生成和 Zep 图谱构建接口；图谱完成后自动把新项目和 graph ID 带入 S1 准备步骤。

使用返回的 `run_id` 启动后台实验：

```json
POST /api/finance/s1/reddit/run
{
  "run_id": "s1_reddit_xxxxxxxxxxxx"
}
```

查询状态、预测和下载 CSV：

```text
GET /api/finance/s1/reddit/<run_id>
GET /api/finance/s1/reddit/<run_id>/predictions
GET /api/finance/s1/reddit/<run_id>/mapping
GET /api/finance/s1/reddit/<run_id>/metrics
GET /api/finance/s1/reddit/<run_id>/actions?limit=100
GET /api/finance/s1/reddit/<run_id>/csv/predictions
GET /api/finance/s1/reddit/<run_id>/csv/evaluation
GET /api/finance/s1/reddit/<run_id>/csv/agent_changes
GET /api/finance/s1/reddit/<run_id>/csv/round_metrics
```

`predictions` 接口默认返回 pre/post 两阶段结果，也可用 `?stage=pre` 或 `?stage=post` 过滤。

已经完成图谱的场景可以通过前端“一键运行全部已构图场景”串行执行。后端读取 `Dataset/seed5_small_blind/zep_graphs_manifest.json`，只接受其中 `status=completed` 的条目；当前清单已通过 Zep Cloud 核验，包含 `SCN_001` 到 `SCN_018` 全部 18 个场景。单个场景失败会被记录，批次继续下一个场景，不会同时启动多套 OASIS 环境。批次目录中的 `scenario_summary.csv` 汇总每个场景的改判率、JS 变化、共识、群体熵、极化程度和互动量。对应 API 为：

```text
POST /api/finance/s1/reddit/batch/prepare
POST /api/finance/s1/reddit/batch/run
GET /api/finance/s1/reddit/batch/<batch_id>
```

每次 S1 运行归档在 `backend/uploads/finance/<run_id>/`，主要增加以下文件：

- `history_memory.jsonl`：预载入每个投资者只读记忆的 5 条历史事件；
- `current_event.json`：作为 Reddit 初始公开帖发布的当前事件及其图谱信息源；
- `entity_agent_mapping.json`：图谱实体、动态 Agent ID、事件发布者和被提及实体之间的可审计映射；
- `social_actions.jsonl`：从 OASIS `trace` 表导出的完整互动动作，包含推断轮次、Agent、动作类型和完整参数；
- `interview_responses.json`：pre-social 采访响应，以及 post-social 从最后快照派生且未额外调用 LLM 的审计记录；
- `pre_social_predictions.jsonl`、`post_social_predictions.jsonl`：两阶段结构化预测；
- `predictions.jsonl`：post-social 兼容副本；`predictions.csv` 同时包含两个阶段；
- `prediction_changes.jsonl`、`agent_changes.csv`：逐 Agent 的改判、概率分布 JS、预期收益和置信度变化；
- `social_metrics.json`：方向分布、平均概率、群体熵、共识率、极化程度、改判率和各项 pre/post 变化；
- `round_metrics.csv`：逐轮动作数、活跃 Agent 数、发帖、评论、点赞、点踩和刷新数；
- `evaluation.csv`：两阶段预测与隐藏五日真实结果的连接表；
- `llm_token_usage.jsonl`：OASIS/CAMEL 每次模型调用的供应商 token 用量、Agent、阶段和轮次；
- `agent_token_usage.csv`：逐投资者 Agent 汇总，并分别统计 pre-social、社会互动、逐轮信念快照和 post-social 用量；
- `token_usage_summary.json`：本场景的总体及分阶段 token 汇总。
- `random_seed_state.json`：本地随机源实际应用状态，以及 DeepSeek 服务端非确定性的说明。

对应 OASIS 数据库位于 `backend/uploads/simulations/<simulation_id>/reddit_simulation.db`，其中 `trace` 表可用于统计真实发帖、评论、点赞、搜索和采访动作。

S1 前端工作台地址：

```text
http://localhost:3000/finance/s1
```

页面支持选择单个匿名场景、复用或新建图谱、调整社会互动轮数、查看历史记忆与当前公开事件、启动后台互动、比较每个 Agent 的 pre/post 预测、查看群体指标与最近互动轨迹，以及一键串行运行图谱清单中的全部场景。页面不再提供“每轮分钟数”，因为 S1 轮次是离散互动步而非现实时间。

## Research artifacts added

- `round_belief_interviews.jsonl`: raw private forecast interviews after each social round.
- `belief_snapshots.jsonl`: parsed round `0..N` belief rows for every investor; missing interviews are retained as `status=missing`.
- `exposure_edges.jsonl`: one row per observed feed exposure or direct interaction, with viewer, content, author, round, first-seen round, and stance annotation provenance.
- Source-account posts are labeled `informational`; investor posts/comments use explicit metadata when supplied by OASIS, otherwise `lexicon_v1` is a transparent heuristic and must not be treated as human ground truth.

### Offline LLM stance annotation

The social simulation does not call a second model to classify content. After
an S1 run is complete, annotate the deduplicated posts/comments offline:

```powershell
cd MiroFish/backend
python scripts/annotate_stances.py --run-id s1_reddit_xxxxxxxx
```

For a genuinely independent coder, set all three variables in `MiroFish/.env`:

```dotenv
STANCE_LLM_API_KEY=...
STANCE_LLM_BASE_URL=https://api.example.com/v1
STANCE_LLM_MODEL_NAME=an-independent-model
```

The API and the CLI require this independent configuration by default. The
CLI's `--allow-primary-fallback` switch is for local debugging only and must
not be used for the final experiment.

The annotator writes `stance_annotations.jsonl` and `stance_annotations.csv`,
plus the derived views `social_actions_annotated.jsonl` and
`exposure_edges_annotated.jsonl`. Raw OASIS artifacts are never overwritten.
The derived rows retain `baseline_content_stance`,
`baseline_stance_score`, and `baseline_stance_source` so the former
`lexicon_v1` result remains an auditable baseline. After annotation, the
actions and exposure-edges API endpoints prefer the derived views; the
dedicated endpoints are:

```text
POST /api/finance/s1/reddit/<run_id>/stance-annotate
GET  /api/finance/s1/reddit/<run_id>/stance-annotations
GET  /api/finance/s1/reddit/<run_id>/exposure-edges-annotated
```

根目录的 `npm run backend` 会优先使用 `backend/.venv`。因此在 Windows 上已有项目虚拟环境时，不要求额外安装 `uv`；如果两者都不存在，脚本才会回退到系统 Python。

## 静态验证

不启动 Flask、OASIS 或真实 LLM，可运行金融适配层的单元测试：

```text
MiroFish/backend/.venv/Scripts/python.exe -m pytest -q MiroFish/backend/tests/test_finance_c0.py
MiroFish/backend/.venv/Scripts/python.exe -m pytest -q MiroFish/backend/tests/test_finance_s1.py
```
