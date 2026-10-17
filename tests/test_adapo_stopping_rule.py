"""Unit tests for the AdaPO stopping rule (pure Python, no GPU needed).

    python -m pytest tests/test_adapo_stopping_rule.py
"""

import math
import random

import pytest

from verl.utils.reward_score.ttrl.adapo import (
    AdaPOConfig,
    AdaPOStoppingRule,
    compute_kappa,
    compute_required_confirmations,
    run_sequential,
)


def test_no_decision_before_min_budget():
    rule = AdaPOStoppingRule(n_min=8)
    for _ in range(7):
        assert not rule.update("1")
    assert rule.p0 is None and not rule.stopped


def test_unanimous_stops_at_min_budget():
    rule = run_sequential(["7"] * 64, n_min=8)
    assert rule.stopped and rule.final_answer() == "7" and rule.samples_used() == 8


def test_p0_is_estimated_once_from_first_n_rollouts_and_frozen():
    n_min, eta = 8, 0.6
    # 5/8 votes for "a" in the first N rollouts, then a long run of "a"
    answers = ["a", "b", "a", "c", "a", "b", "a", "a"] + ["a"] * 56
    rule = run_sequential(answers, n_min=n_min, config=AdaPOConfig(eta=eta))
    assert rule.p0 == pytest.approx(eta * 5 / 8)
    assert set(rule.p_history) == {rule.p0}  # never re-estimated after N


def test_p0_ignores_unparsable_answers():
    answers = [None, "a", "a", "b", None, "a", "c", "a"]
    rule = run_sequential(answers, n_min=8, config=AdaPOConfig(eta=0.6))
    assert rule.p0 == pytest.approx(0.6 * 4 / 6)


def test_required_confirmations_formula():
    alpha = beta = 0.05
    gamma = 0.95
    kappa = compute_kappa(0.5, 5)  # = 4
    expected = math.ceil(1 + (math.log(gamma / (1 - gamma)) - math.log((1 - beta) / alpha)) / math.log(kappa))
    assert compute_required_confirmations(kappa, alpha, beta, gamma) == max(1, expected)


def test_uninformative_kappa_never_stops():
    # p_0 = 0.6 * 2/8 = 0.15 < 1/m with m = 5, so kappa <= 1 and the SPRT can never trigger
    answers = ["a", "a", "b", "b", "c", "c", "d", "d"] + ["a"] * 56
    rule = run_sequential(answers, n_min=8)
    assert compute_kappa(rule.p0, 5) <= 1
    assert not rule.stopped and rule.samples_used() == 64 and rule.final_answer() == "a"


def test_confident_leader_stops_early():
    answers = ["a"] * 5 + ["b", "c", "d"] + ["a"] * 56
    rule = run_sequential(answers, n_min=8)
    assert rule.stopped and rule.final_answer() == "a" and rule.samples_used() < 64


@pytest.mark.parametrize("batch_size", [1, 3, 8, 16])
def test_batch_sequential_matches_sequential_stopping_point(batch_size):
    rng = random.Random(0)
    for _ in range(200):
        answers = [rng.choice("aaabbc") for _ in range(64)]
        seq = run_sequential(answers, n_min=16)
        batch = AdaPOStoppingRule(n_min=16)
        for start in range(0, 64, batch_size):
            if batch.update_batch(answers[start : start + batch_size]):
                break
        assert (batch.stopped, batch.samples_used(), batch.final_answer()) == (seq.stopped, seq.samples_used(), seq.final_answer())


def test_frozen_m_uses_first_n_rollouts():
    # 2 candidates in the first N; many new candidates appear afterwards
    answers = ["a"] * 6 + ["b"] * 2 + ["c", "d", "e", "f", "g", "h"] + ["a"] * 50
    dynamic = run_sequential(answers, n_min=8, config=AdaPOConfig(m_mode="dynamic", min_num_candidates=2))
    frozen = run_sequential(answers, n_min=8, config=AdaPOConfig(m_mode="frozen", min_num_candidates=2))
    assert set(frozen.m_history) == {2}
    assert max(dynamic.m_history) > 2
