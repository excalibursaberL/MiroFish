from app.services.report_agent import ReportAgent, ReportOutline, ReportSection


class RecordingLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_report_routes_tools_without_thinking_then_enables_it_for_final_prose():
    llm = RecordingLLM(
        [
            '<tool_call>{"name":"quick_search","parameters":{"query":"a"}}</tool_call>',
            '<tool_call>{"name":"panorama_search","parameters":{}}</tool_call>',
            '<tool_call>{"name":"interview_agents","parameters":{}}</tool_call>',
            "Final Answer: ## 分析结果\n\n结论。",
        ]
    )
    agent = object.__new__(ReportAgent)
    agent.llm = llm
    agent.simulation_requirement = "测试预测"
    agent.report_logger = None
    agent._get_tools_description = lambda: "tools"
    agent._execute_tool = (
        lambda _name, _parameters, report_context=None: "工具返回结果"
    )

    result = agent._generate_section_react(
        ReportSection(title="测试章节", content=""),
        ReportOutline(title="测试报告", summary="摘要", sections=[]),
        previous_sections=[],
        section_index=1,
    )

    assert result == "## 分析结果\n\n结论。"
    assert [call["thinking_mode"] for call in llm.calls] == [
        "disabled",
        "disabled",
        "disabled",
        "enabled",
    ]
    assert llm.calls[-1]["reasoning_effort"] == "low"
    assert llm.calls[-1]["fallback_to_non_thinking"] is True
    assert llm.calls[-1]["max_tokens"] == 8192
