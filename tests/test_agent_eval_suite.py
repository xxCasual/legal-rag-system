"""Lightweight checks for the Agent Eval suite runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.run_agent_eval_suite as suite  # noqa: E402


def _sample(
    query: str,
    expected_intent,
    expected_tools,
    should_refuse: bool,
    expected_risk_level,
):
    return {
        "query": query,
        "expected_intent": expected_intent,
        "expected_tools": expected_tools,
        "should_refuse": should_refuse,
        "expected_risk_level": expected_risk_level,
    }


def test_agent_eval_suite_splits_samples_by_expected_fields() -> None:
    samples = [
        _sample("试用期最长多久？", "law_qa", ["search_law_articles"], False, None),
        _sample("工资发放日？", "policy_qa", ["search_company_policy"], False, None),
        _sample("讲个笑话", None, [], True, None),
        _sample("合同写放弃社保", None, ["contract_review_rules"], True, "high"),
    ]

    subsets = suite.split_samples(samples)

    assert [sample["query"] for sample in subsets["law_qa"]] == ["试用期最长多久？"]
    assert [sample["query"] for sample in subsets["policy_qa"]] == ["工资发放日？"]
    assert [sample["query"] for sample in subsets["refusal"]] == ["讲个笑话"]
    assert [sample["query"] for sample in subsets["contract_review"]] == ["合同写放弃社保"]
    assert len(subsets["full"]) == 4


def test_agent_eval_suite_selects_expected_subsets() -> None:
    assert suite.selected_subsets("all") == (
        "law_qa",
        "policy_qa",
        "refusal",
        "contract_review",
        "full",
    )
    assert suite.selected_subsets("split") == (
        "law_qa",
        "policy_qa",
        "refusal",
        "contract_review",
    )
    assert suite.selected_subsets("contract") == ("contract_review",)


def test_agent_eval_suite_builds_result_tags_and_latency() -> None:
    assert suite.combined_tag("flash", "law_qa") == "flash_law_qa"
    assert suite.combined_tag(None, "policy_qa") == "policy_qa"
    assert suite.average_latency([{"latency": 1}, {"latency": 2.5}]) == 1.75
    assert suite.average_latency([]) == 0.0


def test_agent_eval_suite_detects_policy_seed_requirement() -> None:
    assert suite.needs_company_policy_seed(("policy_qa",)) is True
    assert suite.needs_company_policy_seed(("full",)) is True
    assert suite.needs_company_policy_seed(("law_qa", "refusal")) is False


def test_agent_eval_suite_seeds_company_policy_index_when_empty() -> None:
    class FakeDocumentService:
        def __init__(self) -> None:
            self.search_calls = []
            self.ingested = []

        def search(self, query: str, k: int = 1):
            self.search_calls.append((query, k))
            return []

        def ingest_upload(self, file_name: str, content: bytes):
            self.ingested.append((file_name, content.decode("utf-8")))
            return {"doc_id": "seed"}

    fake_service = FakeDocumentService()

    seeded = suite.ensure_company_policy_seeded(fake_service)

    assert seeded is True
    assert fake_service.search_calls == [("工资发放日 报销 迟到 加班审批", 1)]
    assert fake_service.ingested
    assert fake_service.ingested[0][0] == "agent_eval_company_policy_seed.txt"
    assert "工资发放日为每月十日" in fake_service.ingested[0][1]


def test_agent_eval_suite_skips_policy_seed_when_index_has_context() -> None:
    class FakeDocumentService:
        def __init__(self) -> None:
            self.ingested = []

        def search(self, query: str, k: int = 1):
            return ["existing context"]

        def ingest_upload(self, file_name: str, content: bytes):
            self.ingested.append((file_name, content))
            return {"doc_id": "seed"}

    fake_service = FakeDocumentService()

    seeded = suite.ensure_company_policy_seeded(fake_service)

    assert seeded is False
    assert fake_service.ingested == []


def test_routing_only_agent_runner_returns_expected_tools_without_rag() -> None:
    law_result = suite.routing_only_agent_runner("试用期最长多久？")
    policy_result = suite.routing_only_agent_runner("公司的工资发放日是什么时候？")
    refusal_result = suite.routing_only_agent_runner("帮我查一下明天台北天气。")

    assert law_result["intent"] == "law_qa"
    assert law_result["tools_used"] == ["search_law_articles"]
    assert policy_result["intent"] == "policy_qa"
    assert policy_result["tools_used"] == ["search_company_policy"]
    assert refusal_result["intent"] == "refusal"
    assert refusal_result["tools_used"] == []
    assert refusal_result["result_type"] == "refusal"


def test_routing_only_agent_runner_reviews_contract_risk_without_evidence() -> None:
    result = suite.routing_only_agent_runner("合同写：员工自愿放弃公司为其缴纳社会保险。")

    assert result["intent"] == "contract_review"
    assert result["tools_used"] == ["contract_review_rules"]
    assert result["risk_level"] == "high"
    assert result["review_status"] == "pending_review"


def test_agent_eval_suite_writes_suite_summary() -> None:
    record = {
        "subset": "law_qa",
        "samples": 1,
        "failed": 0,
        "avg_latency": 0.25,
        "metrics": {
            "intent_accuracy": {"score": 1.0, "correct": 1, "valid_samples": 1},
            "tool_call_accuracy": {"score": 1.0, "correct": 1, "valid_samples": 1},
            "refusal_accuracy": {"score": 1.0, "correct": 1, "valid_samples": 1},
            "risk_accuracy": {"score": None, "correct": 0, "valid_samples": 0},
        },
        "csv_path": "/tmp/law.csv",
        "json_path": "/tmp/law.json",
    }

    with TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        path = suite.write_suite_summary(
            [record],
            testset_path=Path("data/eval/test.json"),
            output_dir=output_dir,
            suite="law",
            tag="unit",
        )

        assert path.exists()
        assert path.name.startswith("agent_eval_unit_suite_")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["suite"] == "law"
        assert payload["tag"] == "unit"
        assert payload["runs"][0]["subset"] == "law_qa"


if __name__ == "__main__":
    test_agent_eval_suite_splits_samples_by_expected_fields()
    test_agent_eval_suite_selects_expected_subsets()
    test_agent_eval_suite_builds_result_tags_and_latency()
    test_agent_eval_suite_detects_policy_seed_requirement()
    test_agent_eval_suite_seeds_company_policy_index_when_empty()
    test_agent_eval_suite_skips_policy_seed_when_index_has_context()
    test_routing_only_agent_runner_returns_expected_tools_without_rag()
    test_routing_only_agent_runner_reviews_contract_risk_without_evidence()
    test_agent_eval_suite_writes_suite_summary()
    print("agent eval suite ok")
