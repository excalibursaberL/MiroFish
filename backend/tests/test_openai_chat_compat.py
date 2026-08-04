from types import SimpleNamespace

import pytest

from app.utils.openai_chat_compat import (
    create_chat_completion,
    deepseek_v4_request_options,
    extract_chat_completion_finish_reason,
    extract_chat_completion_text,
    has_chat_completion_reasoning_content,
    is_deepseek_v4_family,
    is_gpt5_family,
    strip_internal_protocol_markers,
)


class CompletionRecorder:
    def __init__(self, result=None, error=None):
        self.result = result or object()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def client_for(recorder):
    return SimpleNamespace(chat=SimpleNamespace(completions=recorder))


def test_gpt5_uses_completion_token_limit_without_temperature():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    result = create_chat_completion(
        client_for(recorder),
        model="gpt-5-2025-08-07",
        messages=messages,
        temperature=0.2,
        max_tokens=123,
        response_format={"type": "json_object"},
    )

    assert result is recorder.result
    assert recorder.calls == [
        {
            "model": "gpt-5-2025-08-07",
            "messages": messages,
            "max_completion_tokens": 123,
            "response_format": {"type": "json_object"},
        }
    ]


def test_legacy_model_preserves_original_request_shape():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "hello"}]

    create_chat_completion(
        client_for(recorder),
        model="third-party-chat-model",
        messages=messages,
        temperature=0.7,
        max_tokens=456,
        response_format={"type": "json_object"},
    )

    assert recorder.calls == [
        {
            "model": "third-party-chat-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 456,
            "response_format": {"type": "json_object"},
        }
    ]


def test_provider_error_is_propagated_without_guessing_or_retrying():
    provider_error = RuntimeError("unsupported max_tokens due to a server outage")
    recorder = CompletionRecorder(error=provider_error)

    with pytest.raises(RuntimeError) as captured:
        create_chat_completion(
            client_for(recorder),
            model="legacy-model",
            messages=[],
            max_tokens=10,
        )

    assert captured.value is provider_error
    assert len(recorder.calls) == 1


def test_deepseek_v4_disables_thinking_for_protocol_sensitive_calls():
    recorder = CompletionRecorder()
    messages = [{"role": "user", "content": "Return JSON"}]

    create_chat_completion(
        client_for(recorder),
        model="deepseek-v4-flash",
        messages=messages,
        response_format={"type": "json_object"},
        thinking_mode="disabled",
    )

    assert recorder.calls == [
        {
            "model": "deepseek-v4-flash",
            "messages": messages,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    ]


def test_deepseek_v4_final_writing_enables_low_effort_thinking():
    assert deepseek_v4_request_options(
        "deepseek-v4-pro",
        thinking_mode="enabled",
        reasoning_effort="low",
    ) == {
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "low",
    }


def test_non_deepseek_model_does_not_receive_provider_specific_options():
    assert deepseek_v4_request_options(
        "gpt-4.1",
        thinking_mode="enabled",
        reasoning_effort="low",
    ) == {}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5", True),
        (" GPT-5.1-mini ", True),
        ("gpt-4.1", False),
        ("my-gpt-5-proxy", False),
        (None, False),
    ],
)
def test_gpt5_family_detection(model, expected):
    assert is_gpt5_family(model) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek-v4-flash", True),
        (" DeepSeek-V4-Pro ", True),
        ("deepseek-chat", False),
        ("my-deepseek-v4-proxy", False),
        (None, False),
    ],
)
def test_deepseek_v4_family_detection(model, expected):
    assert is_deepseek_v4_family(model) is expected


def test_extracts_text_from_supported_content_shapes():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"text": {"value": "first"}},
                        {"content": " second"},
                        SimpleNamespace(text=" third"),
                    ]
                )
            )
        ]
    )

    assert extract_chat_completion_text(response) == "first second third"
    assert extract_chat_completion_text(SimpleNamespace(choices=[])) == ""


def test_extracts_finish_reason_from_supported_choice_shapes():
    assert extract_chat_completion_finish_reason(
        SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])
    ) == "length"
    assert extract_chat_completion_finish_reason(
        SimpleNamespace(choices=[{"finish_reason": "stop"}])
    ) == "stop"
    assert extract_chat_completion_finish_reason(SimpleNamespace(choices=[])) is None


def test_detects_reasoning_content_without_exposing_it():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    reasoning_content="internal reasoning",
                )
            )
        ]
    )
    assert has_chat_completion_reasoning_content(response) is True


def test_strips_leaked_deepseek_dsml_markers():
    content = "before\n<｜｜DSML｜｜>\n<<｜｜DSML｜｜>>\nafter"

    assert "DSML" not in strip_internal_protocol_markers(content)
    assert strip_internal_protocol_markers(content).startswith("before")
    assert strip_internal_protocol_markers(content).endswith("after")
