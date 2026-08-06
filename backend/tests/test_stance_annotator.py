import json
from types import SimpleNamespace

from app.finance.stance_annotator import OfflineStanceAnnotator


class CompletionRecorder:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


def fake_client(content):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=CompletionRecorder(content))
    )


def test_stance_payload_accepts_aliases_and_bounds():
    payload = OfflineStanceAnnotator._validate_payload(
        {
            "stance": "bullish",
            "target": "stock",
            "event_valence": "up",
            "stance_score": "0.75",
            "confidence": 0.8,
            "reason": "文本明确表达看好",
        }
    )
    assert payload["stance"] == "positive"
    assert payload["event_valence"] == "positive"
    assert payload["stance_score"] == 0.75


def test_stance_prompt_requires_independent_json_annotation():
    prompt = OfflineStanceAnnotator.build_prompt(
        {
            "content_type": "post",
            "content_id": 1,
            "author_class": "investor",
            "author_agent_id": 0,
            "round": 1,
            "content_text": "这条消息可能带来增长。",
        }
    )
    assert len(prompt) == 2
    assert "不是投资 Agent" in prompt[0]["content"]
    assert "必须返回一个 JSON object" in prompt[0]["content"]
    assert "stance_score" in prompt[1]["content"]


def test_annotate_completed_run_writes_auditable_derived_views(tmp_path):
    run_id = "s1_reddit_annotator_test"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "scenario_id": "SCN_001", "status": "completed"}),
        encoding="utf-8",
    )
    (run_dir / "social_actions.jsonl").write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in [
                {
                    "action_type": "create_post",
                    "post_id": 1,
                    "agent_id": 0,
                    "agent_class": "investor",
                    "round": 1,
                    "action_args": {"content": "我看好这个事件。"},
                    "content_stance": "positive",
                    "stance_score": 1.0,
                    "stance_source": "lexicon_v1",
                },
                {
                    "action_type": "like_post",
                    "post_id": 1,
                    "agent_id": 1,
                    "agent_class": "investor",
                    "round": 1,
                },
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "exposure_edges.jsonl").write_text(
        json.dumps(
            {
                "content_type": "post",
                "content_id": 1,
                "content_text": "我看好这个事件。",
                "content_stance": "positive",
                "stance_score": 1.0,
                "stance_source": "lexicon_v1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    response = json.dumps(
        {
            "stance": "positive",
            "target": "stock",
            "event_valence": "positive",
            "stance_score": 0.9,
            "confidence": 0.88,
            "supports_content_id": None,
            "challenges_content_id": None,
            "reason": "作者明确表示看好。",
        },
        ensure_ascii=False,
    )
    recorder = CompletionRecorder(response)
    annotator = OfflineStanceAnnotator(
        storage_dir=tmp_path,
        api_key="test-key",
        base_url="https://annotator.example/v1",
        model="independent-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=recorder)),
    )

    result = annotator.annotate_run(run_id)

    assert result["content_count"] == 1
    assert result["success_count"] == 1
    assert len(recorder.calls) == 1
    request = recorder.calls[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["model"] == "independent-model"
    rows = annotator.get_annotations(run_id)
    assert rows[0]["stance"] == "positive"
    assert rows[0]["scenario_id"] == "SCN_001"
    enriched = annotator.get_annotated_exposure_edges(run_id)
    assert enriched[0]["stance_source"] == "offline_llm"
    assert enriched[0]["baseline_stance_source"] == "lexicon_v1"
    assert enriched[0]["stance_confidence"] == 0.88

