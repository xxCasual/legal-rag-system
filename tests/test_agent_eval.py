"""Lightweight checks for Agent Eval metrics and outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.agent_eval as agent_eval  # noqa: E402


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


def test_agent_eval_calculates_discrete_metrics() -> None:
    samples = [
        _sample("试用期最长多久？", "law_qa", ["search_law_articles"], False, None),
        _sample("公司的工资发放日是什么时候？", "policy_qa", ["search_company_policy"], False, None),
        _sample("合同缺少工资", None, ["contract_review_rules"], False, "medium"),
        _sample("合同写了放弃社保", None, ["contract_review_rules"], True, "high"),
    ]

    def fake_agent_runner(query: str):
        if "工资发放日" in query:
            return {
                "intent": "policy_qa",
                "tools_used": ["search_company_policy"],
                "answer": "制度答案",
            }
        if "合同" in query:
            risk_level = "high" if "放弃社保" in query else "medium"
            return {
                "intent": "contract_review",
                "tools_used": ["contract_review_rules"],
                "answer": "高风险合同审查结果已进入人工复核队列。"
                if risk_level == "high"
                else "合同审查完成。",
                "risk_level": risk_level,
                "review_status": "pending_review"
                if risk_level == "high"
                else "not_required",
            }
        return {
            "intent": "law_qa",
            "tools_used": ["search_law_articles"],
            "answer": "法律答案",
        }

    rows, summary = agent_eval.run_agent_eval(
        samples,
        agent_runner=fake_agent_runner,
    )

    assert rows[0]["task_type"] == "agent_chat"
    assert rows[2]["task_type"] == "agent_chat"
    assert rows[2]["actual_tools"] == ["contract_review_rules"]
    assert rows[2]["latency"] >= 0
    assert summary["failed_samples"] == 0
    assert summary["metrics"]["intent_accuracy"] == {
        "score": 1.0,
        "correct": 2,
        "valid_samples": 2,
    }
    assert summary["metrics"]["tool_call_accuracy"]["score"] == 1.0
    assert summary["metrics"]["refusal_accuracy"]["score"] == 1.0
    assert summary["metrics"]["risk_accuracy"] == {
        "score": 1.0,
        "correct": 2,
        "valid_samples": 2,
    }


def test_agent_eval_saves_csv_and_json_summary() -> None:
    rows = [
        {
            "sample_id": 0,
            "task_type": "agent_chat",
            "query": "试用期最长多久？",
            "expected_intent": "law_qa",
            "actual_intent": "law_qa",
            "expected_tools": ["search_law_articles"],
            "actual_tools": ["search_law_articles"],
            "expected_risk_level": None,
            "actual_risk_level": None,
            "should_refuse": False,
            "actual_refused": False,
            "intent_match": True,
            "tool_call_match": True,
            "refusal_match": True,
            "risk_match": None,
            "latency": 0.01,
            "error": "",
        }
    ]
    summary = agent_eval.build_summary(rows)

    with TemporaryDirectory() as temp_dir:
        csv_path, json_path = agent_eval.save_results(
            rows,
            summary,
            results_dir=Path(temp_dir),
            tag="unit",
        )

        assert csv_path.exists()
        assert json_path.exists()
        assert "agent_eval_unit_" in csv_path.name
        saved_summary = json.loads(json_path.read_text(encoding="utf-8"))
        assert saved_summary["csv_path"] == str(csv_path)
        assert saved_summary["metrics"]["tool_call_accuracy"]["score"] == 1.0


def test_agent_eval_uses_dedicated_results_directory() -> None:
    assert agent_eval.AGENT_RESULTS_DIR == ROOT / "data" / "eval" / "agent_results"
    assert agent_eval.AGENT_RESULTS_DIR.name != "results"


def test_agent_eval_records_sample_errors_without_stopping() -> None:
    samples = [
        _sample("会失败的问题", "law_qa", ["search_law_articles"], False, None)
    ]

    def failing_agent_runner(_: str):
        raise RuntimeError("agent failed")

    rows, summary = agent_eval.run_agent_eval(
        samples,
        agent_runner=failing_agent_runner,
        contract_reviewer=lambda _: {"risk_level": "low"},
    )

    assert rows[0]["error"] == "RuntimeError: agent failed"
    assert summary["failed_samples"] == 1
    assert summary["metrics"]["intent_accuracy"]["score"] == 0.0
    assert summary["metrics"]["tool_call_accuracy"]["score"] == 0.0
    assert summary["metrics"]["refusal_accuracy"]["score"] == 0.0


def test_agent_eval_requires_no_tools_when_expected_tools_is_empty() -> None:
    assert agent_eval._tools_match([], []) is True
    assert agent_eval._tools_match([], ["search_law_articles"]) is False


def test_agent_eval_treats_refusal_route_as_refused() -> None:
    assert agent_eval._extract_refused(
        {
            "route": "refusal",
            "result_type": "refusal",
            "answer": "我只能处理劳动合规相关问题。",
        }
    ) is True


def test_agent_eval_validates_required_fields() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "bad_agent_testset.json"
        path.write_text(
            json.dumps([{"query": "缺少字段"}], ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            agent_eval.load_agent_testset(path)
        except ValueError as exc:
            assert "missing required fields" in str(exc)
        else:
            raise AssertionError("invalid Agent Eval samples should be rejected")


if __name__ == "__main__":
    test_agent_eval_calculates_discrete_metrics()
    test_agent_eval_saves_csv_and_json_summary()
    test_agent_eval_uses_dedicated_results_directory()
    test_agent_eval_records_sample_errors_without_stopping()
    test_agent_eval_requires_no_tools_when_expected_tools_is_empty()
    test_agent_eval_treats_refusal_route_as_refused()
    test_agent_eval_validates_required_fields()
    print("agent eval ok")
