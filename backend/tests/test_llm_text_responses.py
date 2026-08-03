from types import SimpleNamespace

import pytest

from app.utils.llm_client import LLMClient, LLMResponseError


class CompletionSequence:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(content, *, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _client_for(sequence, *, model="deepseek-v4-flash"):
    client = object.__new__(LLMClient)
    client.model = model
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=sequence)
    )
    return client


def test_chat_removes_dsml_markers_from_otherwise_valid_text():
    sequence = CompletionSequence(
        _response("分析\n<｜｜DSML｜｜>\n结论")
    )

    result = _client_for(sequence).chat(
        messages=[{"role": "user", "content": "Analyze"}],
        thinking_mode="enabled",
    )

    assert result == "分析\n\n结论"
    assert "DSML" not in result


def test_chat_rejects_response_containing_only_dsml_markers():
    sequence = CompletionSequence(
        _response("<｜｜DSML｜｜>\n<｜｜DSML｜｜>")
    )

    with pytest.raises(LLMResponseError, match="empty text"):
        _client_for(sequence).chat(
            messages=[{"role": "user", "content": "Analyze"}],
            thinking_mode="enabled",
        )


def test_thinking_text_falls_back_to_non_thinking_after_truncation():
    sequence = CompletionSequence(
        _response("partial", finish_reason="length"),
        _response("完整结果"),
    )

    result = _client_for(sequence).chat(
        messages=[{"role": "user", "content": "Analyze"}],
        max_tokens=8192,
        thinking_mode="enabled",
        reasoning_effort="low",
        fallback_to_non_thinking=True,
    )

    assert result == "完整结果"
    assert sequence.calls[0]["extra_body"] == {
        "thinking": {"type": "enabled"}
    }
    assert sequence.calls[0]["reasoning_effort"] == "low"
    assert sequence.calls[1]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "reasoning_effort" not in sequence.calls[1]
