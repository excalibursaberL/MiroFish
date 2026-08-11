---
name: ashare-institutional-analyst
description: Provide a frozen-evidence institutional A-share research workflow for anonymous T+5 direction forecasts. Use when an institutional finance Agent must analyze supplied historical events through event/news, policy, market-flow, and fundamental-risk lenses without external data, future leakage, extra social nodes, or executable trading advice.
---

# A 股研究分析

把本 Skill 作为投资者的内部研究方法，不要把其中的 Analyst 创建为新的社会 Agent。只让 Agent 对外发帖、评论和提交预测。

## 信息边界

- 只使用当前 Prompt、只读历史记忆和实际曝光内容中明确提供的信息。
- 不调用外部搜索、行情或数据库，不推断匿名公司的真实身份。
- 不使用预测截止点之后的信息、隐藏评价标签或其他 Agent 的私有状态。
- 输入未提供价格、成交量、资金流或财务指标时，明确记为缺失，不得编造。
- 始终预测题目指定的未来 5 个交易日累计收盘收益，不把个人投资期限改写成预测期限。

## 内部研究台

在一次推理中依次执行以下视角。它们是分析职责，不是独立 LLM 调用。

### 1. 事件与新闻 Analyst

- 区分已确认事实、主体表态、市场叙事和推测。
- 判断事件的新信息量、来源可信度、影响对象以及直接或间接传导路径。
- 只引用输入中存在的 `event_id`，并优先使用与 T+5 时间范围相关的证据。

### 2. 政策与监管 Analyst

- 仅在输入明确涉及政策、监管、交易规则、行业准入或处罚时激活。
- 写出“政策事实 -> 公司或行业约束 -> 五日价格可能反应”的最短证据链。
- 区分正式落地、征求意见、传闻和一般性表态，不把长期政策方向直接等同于短期涨跌。

### 3. 市场与资金行为 Analyst

- 仅使用输入明确给出的价格、成交量、热度、资金流、减持、解禁或交易行为。
- 判断短期关注度、供需冲击和兑现压力，但不编造均线、MACD、龙虎榜或北向资金数据。
- 将资金与价格行为视为条件性证据，不让单一短期信号覆盖全部事件证据。

### 4. 基本面与风险复核 Analyst

- 检查事件是否改变盈利、现金流、资产负债、竞争格局、供给或治理风险。
- 至少寻找一项反向证据、兑现条件或信息缺口。
- 证据不足或相互冲突时降低置信度，不能制造虚假确定性。

## 综合流程

1. 建立证据账本：记录每个候选 `event_id` 的事实、时间、来源、方向和缺失项。
2. 分别形成各研究台的暂定方向、影响强度、置信度及引用事件。
3. 检查研究台之间的冲突。不要使用预设固定权重，也不要通过简单多数投票掩盖关键风险。
4. 将剩余不确定性显式反映在三类概率和 `confidence` 中。
5. 输出与实验公共契约完全一致的 JSON object，不添加额外字段或 Markdown。

## 输出契约

```json
{
  "direction": "up|neutral|down",
  "up_probability": 0.0,
  "neutral_probability": 0.0,
  "down_probability": 0.0,
  "expected_return": 0.0,
  "confidence": 0.0,
  "evidence_event_ids": [],
  "reason": "基于可核验事件的简洁综合理由，并说明主要反向风险或缺口"
}
```

- 三项概率必须位于 `[0,1]` 且总和为 1。
- `expected_return` 使用小数单位，例如 `0.03` 表示 3%。
- `evidence_event_ids` 只能包含输入中出现的事件 ID。
- `reason` 给出可审计证据摘要，不输出隐藏推理过程，不给出仓位、买卖点或交易指令。

## 实验一致性

- 不修改 Agent 原有 Profile、风险态度或社会行为规则。
- 同一实验条件下对所有场景使用同一 Skill 文本和版本。
- 将 Skill 名称、内容 SHA-256 和 Agent 分配关系写入运行清单。
- 对照实验关闭本 Skill。
