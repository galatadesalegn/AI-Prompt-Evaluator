"""
tests/test_evaluator.py — Unit tests for the AI Prompt Evaluator.
Run with: pytest tests/ -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from evaluator.criteria import CRITERIA, get_quality_tier, CRITERIA_MAP
from evaluator.evaluator import (
    PromptEvaluator, EvalResult, EvalScore,
    _detect_flags, _heuristic_score,
)
from evaluator.storage import save, load_all, export_json, export_csv
from evaluator.report import print_report


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def evaluator():
    return PromptEvaluator(model_name="test-model")


@pytest.fixture
def good_response():
    return {
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris. It is the largest city and the cultural heart of the country.",
    }


@pytest.fixture
def bad_response():
    return {
        "prompt": "Tell me something.",
        "response": "ok",
    }


@pytest.fixture
def toxic_response():
    return {
        "prompt": "What do you think?",
        "response": "You are an idiot for asking that. I hate you.",
    }


@pytest.fixture
def refusal_response():
    return {
        "prompt": "How do I hack an account?",
        "response": "I cannot help with that request as it could be used to harm others.",
    }


# ── Criteria tests ─────────────────────────────────────────────────────────────

class TestCriteria:

    def test_criteria_weights_sum_to_one(self):
        total = sum(c.weight for c in CRITERIA)
        assert abs(total - 1.0) < 0.001

    def test_all_criteria_have_names(self):
        for c in CRITERIA:
            assert c.name and isinstance(c.name, str)

    def test_quality_tier_excellent(self):
        assert get_quality_tier(4.8) == "Excellent"

    def test_quality_tier_good(self):
        assert get_quality_tier(4.0) == "Good"

    def test_quality_tier_fair(self):
        assert get_quality_tier(3.0) == "Fair"

    def test_quality_tier_poor(self):
        assert get_quality_tier(2.0) == "Poor"

    def test_quality_tier_unacceptable(self):
        assert get_quality_tier(0.5) == "Unacceptable"

    def test_criteria_map_has_all_keys(self):
        for c in CRITERIA:
            assert c.name in CRITERIA_MAP


# ── Flag detection tests ───────────────────────────────────────────────────────

class TestFlagDetection:

    def test_detects_toxic_language(self, toxic_response):
        flags = _detect_flags(toxic_response["response"])
        assert "toxic_language" in flags

    def test_detects_refusal(self, refusal_response):
        flags = _detect_flags(refusal_response["response"])
        assert "refusal" in flags

    def test_detects_too_short(self, bad_response):
        flags = _detect_flags(bad_response["response"])
        assert "too_short" in flags

    def test_no_flags_on_clean_response(self, good_response):
        flags = _detect_flags(good_response["response"])
        assert "toxic_language" not in flags
        assert "too_short" not in flags

    def test_detects_hallucination_signals(self):
        response = "I believe this is roughly correct, though I'm not sure."
        flags = _detect_flags(response)
        assert "hallucination_risk" in flags


# ── Heuristic scoring tests ────────────────────────────────────────────────────

class TestHeuristicScoring:

    def test_good_response_scores_high(self, good_response):
        scores = _heuristic_score(**good_response)
        assert scores["helpfulness"] >= 4
        assert scores["safety"] == 5

    def test_toxic_response_scores_low_safety(self, toxic_response):
        scores = _heuristic_score(**toxic_response)
        assert scores["safety"] == 1
        assert scores["tone"] == 1

    def test_refusal_scores_low_helpfulness(self, refusal_response):
        scores = _heuristic_score(**refusal_response)
        assert scores["helpfulness"] == 1

    def test_short_response_scores_low_clarity(self, bad_response):
        scores = _heuristic_score(**bad_response)
        assert scores["clarity"] <= 2


# ── PromptEvaluator tests ──────────────────────────────────────────────────────

class TestPromptEvaluator:

    def test_heuristic_returns_eval_result(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        assert isinstance(result, EvalResult)

    def test_eval_id_is_generated(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        assert result.eval_id and len(result.eval_id) == 8

    def test_overall_score_in_range(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        assert 0.0 <= result.overall_score <= 5.0

    def test_quality_tier_assigned(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        assert result.quality_tier in {"Excellent", "Good", "Fair", "Poor", "Unacceptable"}

    def test_scores_list_has_all_criteria(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        names = {s.criterion for s in result.scores}
        expected = {"helpfulness", "clarity", "accuracy", "safety", "tone"}
        assert names == expected

    def test_model_name_stored(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        assert result.model_name == "test-model"

    def test_toxic_response_flagged(self, evaluator, toxic_response):
        result = evaluator.evaluate_heuristic(**toxic_response)
        assert "toxic_language" in result.flags

    def test_evaluator_type_heuristic(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        assert result.evaluator_type == "heuristic"

    def test_to_dict_has_required_keys(self, evaluator, good_response):
        result = evaluator.evaluate_heuristic(**good_response)
        d = result.to_dict()
        for key in ["eval_id", "prompt", "response", "model_name",
                    "overall_score", "quality_tier", "flags", "timestamp"]:
            assert key in d


# ── Storage tests ──────────────────────────────────────────────────────────────

class TestStorage:

    def test_save_and_load(self, tmp_path, evaluator, good_response):
        path = str(tmp_path / "evals.jsonl")
        result = evaluator.evaluate_heuristic(**good_response)
        save(result, path)
        loaded = load_all(path)
        assert len(loaded) == 1
        assert loaded[0]["eval_id"] == result.eval_id

    def test_load_empty_when_no_file(self):
        entries = load_all("nonexistent/path.jsonl")
        assert entries == []

    def test_export_json(self, tmp_path, evaluator, good_response):
        in_path = str(tmp_path / "evals.jsonl")
        out_path = str(tmp_path / "out.json")
        result = evaluator.evaluate_heuristic(**good_response)
        save(result, in_path)
        export_json(in_path, out_path)
        data = json.loads(Path(out_path).read_text())
        assert len(data) == 1

    def test_export_csv(self, tmp_path, evaluator, good_response):
        in_path = str(tmp_path / "evals.jsonl")
        out_path = str(tmp_path / "out.csv")
        result = evaluator.evaluate_heuristic(**good_response)
        save(result, in_path)
        export_csv(in_path, out_path)
        assert Path(out_path).exists()


# ── Report tests ───────────────────────────────────────────────────────────────

class TestReport:

    def test_report_no_crash_on_empty(self, capsys):
        print_report("nonexistent/path.jsonl")
        out = capsys.readouterr().out
        assert "No evaluation data" in out

    def test_report_prints_summary(self, tmp_path, evaluator, good_response, capsys):
        path = str(tmp_path / "evals.jsonl")
        result = evaluator.evaluate_heuristic(**good_response)
        save(result, path)
        print_report(path)
        out = capsys.readouterr().out
        assert "Total evaluations" in out
        assert "Average score" in out
