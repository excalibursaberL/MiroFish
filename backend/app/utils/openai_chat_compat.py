"""
OpenAI Chat Completions compatibility helpers.

This module keeps existing behavior for legacy models/providers while
gracefully adapting request parameters for GPT-5 family models.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional


ThinkingMode = Literal["enabled", "disabled"]

_VALID_THINKING_MODES = {"enabled", "disabled"}
_VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_DSML_MARKER_RE = re.compile(
    r"<+[|｜]+\s*DSML\s*[|｜]+>+",
    flags=re.IGNORECASE,
)


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return True when model belongs to GPT-5 family aliases/snapshots."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def is_deepseek_v4_family(model: Optional[str]) -> bool:
    """Return True for the DeepSeek V4 model aliases exposed by its API."""
    if not model:
        return False
    return model.strip().lower().startswith("deepseek-v4-")


def deepseek_v4_request_options(
    model: Optional[str],
    *,
    thinking_mode: Optional[ThinkingMode] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Build DeepSeek-only request options without affecting other providers.

    DeepSeek V4 defaults to thinking mode.  MiroFish disables it for
    protocol-sensitive JSON/tool-routing calls and enables it explicitly for
    final prose generation.  These options can also be passed to CAMEL's
    ``model_config_dict`` because it forwards them to OpenAI-compatible calls.
    """

    if thinking_mode is not None and thinking_mode not in _VALID_THINKING_MODES:
        raise ValueError(f"Unsupported thinking mode: {thinking_mode}")
    if (
        reasoning_effort is not None
        and reasoning_effort not in _VALID_REASONING_EFFORTS
    ):
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
    if not is_deepseek_v4_family(model) or thinking_mode is None:
        return {}

    options: Dict[str, Any] = {
        "extra_body": {"thinking": {"type": thinking_mode}},
    }
    if thinking_mode == "enabled" and reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    return options


def strip_internal_protocol_markers(content: str) -> str:
    """Remove leaked DeepSeek transport markers from user-visible text."""

    if not isinstance(content, str) or "DSML" not in content.upper():
        return content
    return _DSML_MARKER_RE.sub("", content)


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    thinking_mode: Optional[ThinkingMode] = None,
    reasoning_effort: Optional[str] = None,
) -> Any:
    """
    Create a chat completion with model-specific request parameters.

    Compatibility strategy:
    - For GPT-5 family, avoid sending temperature by default.
    - For token limit, use `max_completion_tokens` on GPT-5, `max_tokens` otherwise.
    - Preserve the legacy request shape for every non-GPT-5 model/provider.
    - Propagate provider errors unchanged instead of guessing from message text.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    gpt5_family = is_gpt5_family(model)

    if temperature is not None and not gpt5_family:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        if gpt5_family:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    kwargs.update(
        deepseek_v4_request_options(
            model,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
    )

    return client.chat.completions.create(**kwargs)


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text from chat completion response across SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return strip_internal_protocol_markers(content)

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return strip_internal_protocol_markers("".join(chunks)).strip()

    return strip_internal_protocol_markers(str(content or ""))


def extract_chat_completion_finish_reason(response: Any) -> Optional[str]:
    """Return the provider finish reason for the first choice, when present."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    choice = choices[0]
    if isinstance(choice, dict):
        reason = choice.get("finish_reason")
    else:
        reason = getattr(choice, "finish_reason", None)
    return str(reason) if reason is not None else None


def has_chat_completion_reasoning_content(response: Any) -> bool:
    """Return whether the first response message contains hidden reasoning text."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            return False
        reasoning = message.get("reasoning_content")
    else:
        message = getattr(choice, "message", None)
        reasoning = getattr(message, "reasoning_content", None)
    return bool(reasoning)
