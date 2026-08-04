# A 股金融适配层：C0 原型

这个目录是 MiroFish 的金融适配层。当前只实现实验计划中的 `C0` 组：

- 支持单场景模式，以及一次运行全部 18 个匿名场景的批量模式；
- 20 个匿名投资者 Agent；
- 3 个机构投资者、6 个有经验散户、8 个具备基础知识的散户、3 个新手散户；
- 每个 Agent 读取同一场景的 5 个历史种子和 1 个当前公开事件；
- 每个 Agent 单独调用一次模型；
- 模型返回空文本、截断文本或无效 JSON 时自动重试一次，并记录 `attempt_count`；
- 每条结果同时记录 `finish_reason`、正文字符数和 `reasoning_content_present`，便于区分空响应、token 截断和思考通道误用；
- Agent 不能看到其他 Agent 的帖子、预测、回复或聚合结果；
- C0 固定使用每个场景的 5 条历史种子；
- `stock_factors` 保留在数据集和冻结快照中，但不写入 LLM Prompt；
- 输出方向、三分类概率、预期收益、置信度、证据事件 ID 和理由。
- 批量模式会在后台依次完成 `18 × 20 = 360` 次模型调用，并持续写入 CSV。

`C0` 暂时不创建 OASIS 社交环境，也不写入 Zep 社会互动记录。这样可以先得到没有社会互动的基线，后续 `S1` 再复用同一组 Profile 和快照，接入 OASIS 的帖子传播链路。

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

页面会依次完成运行范围选择、输入冻结、Prompt dry-run、Agent 正式运行和结果展示。选择“全部场景”后，正式运行按钮会一键启动 18 个场景、360 次预测。首页右上角也提供了“A股 C0 实验”入口。

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
- `profiles.json`：20 个 OASIS 兼容 Profile；
- `scenarios.jsonl`：冻结后的安全输入快照；
- `prompts.jsonl`：每个场景/Agent 的独立 Prompt；
- `predictions.jsonl`：真实运行后的结构化预测。
- `predictions.csv`：逐 Agent 预测，不包含真实答案；运行中每完成一个 Agent 就会刷新，可用于观察部分结果；
- `evaluation.csv`：预测与隐藏真实结果的连接表，只在全部 Agent 完成后生成，供研究者统计准确率和收益误差；
- `llm_responses.jsonl`：每次实际 API 尝试的完整请求和响应，包括重试、响应 ID、choices、content、reasoning_content、finish_reason、usage 和异常信息，不包含 API Key。

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

批量运行期间可以关闭前端页面。后端进程停止或重启时，正在执行的任务会先显示为“中断”；重新打开页面后可点击“继续剩余预测”，已经写入 `predictions.jsonl` 的结果不会重复调用。360 次调用的时间与费用取决于所选模型；正式启动前应先核对模型价格、API 额度和 Prompt dry-run。

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

该接口同时返回 Astock 的原始 `label/CHANGE` 口径和未来 5 个交易日的端到端收盘收益口径。评测文件由独立 evaluator 读取，不会写入 Prompt、Profile、Zep 或 Agent 场景快照。

单场景 API 仍采用同步执行；全部场景使用当前后端进程内的后台线程顺序执行。预测 JSONL 使用单写入者追加模式，manifest 和 CSV 使用唯一临时文件及 Windows 文件占用重试，避免页面轮询或编辑器读取时触发 `WinError 5`。后端重启会将未完成任务标记为中断，研究者可手动继续已冻结的批量运行；当前仍不是自动任务队列。JSON 协议调用会关闭 DeepSeek V4 思考模式，并在空响应或解析失败时重试一次；尚未实现 S1 社会互动。

根目录的 `npm run backend` 会优先使用 `backend/.venv`。因此在 Windows 上已有项目虚拟环境时，不要求额外安装 `uv`；如果两者都不存在，脚本才会回退到系统 Python。

## 静态验证

不启动 Flask、OASIS 或真实 LLM，可运行金融适配层的单元测试：

```text
MiroFish/backend/.venv/Scripts/python.exe -m pytest -q MiroFish/backend/tests/test_finance_c0.py
```
