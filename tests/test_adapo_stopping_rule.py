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
    assert rule.p0 == pytest.approx(eta * 5 / 8)  # never re-estimated after N


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


@pytest.mark.parametrize("kappa", [1.01, 1.5, 4.0, 50.0, 1e6])
def test_default_setting_needs_a_single_confirmation(kappa):
    # gamma / (1 - gamma) = A = 19 at the defaults; float noise must not round Z* up to 2
    assert compute_required_confirmations(kappa, 0.05, 0.05, 0.95) == 1
    assert compute_required_confirmations(0.9, 0.05, 0.05, 0.95) is None  # kappa <= 1: undefined


def test_more_confirmations_for_higher_gamma():
    z = [compute_required_confirmations(4.0, 0.05, 0.05, g) for g in (0.95, 0.99, 0.995, 0.999)]
    assert z == sorted(z) and z[0] == 1 and z[-1] > 1


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
    dynamic = run_sequential(answers[:14], n_min=8, config=AdaPOConfig(m_mode="dynamic", min_num_candidates=2))
    frozen = run_sequential(answers[:14], n_min=8, config=AdaPOConfig(m_mode="frozen", min_num_candidates=2))
    assert frozen.m == 2 and dynamic.m > 2
    assert AdaPOConfig().m_mode == "frozen"


def _reference_rule(answers, n_min, cfg):
    """Literal transcription of the Electing / Confirming automaton (frozen p_0 and m)."""
    from collections import Counter
    A, B = (1 - cfg.beta) / cfg.alpha, cfg.beta / (1 - cfg.alpha)
    votes, phase, cand, z = Counter(), "elc", None, 0
    for n, a in enumerate(answers, 1):
        if a is not None:
            votes[a] += 1
        if n < n_min:
            continue
        if n == n_min:
            p0 = min(cfg.eta * votes.most_common(1)[0][1] / sum(votes.values()), 1 - 1e-6) if votes else 0.0
            kappa = compute_kappa(p0, max(len(votes), cfg.min_num_candidates))
            z_star = compute_required_confirmations(kappa, cfg.alpha, cfg.beta, cfg.gamma)
        if len(votes) == 1:
            return n, next(iter(votes))
        if len(votes) < 2 or kappa <= 1:
            continue
        gap = lambda c: votes[c] - max(v for b, v in votes.items() if b != c)  # noqa: E731
        if phase == "elc":
            leader = max(votes.items(), key=lambda x: x[1])[0]
            if kappa ** gap(leader) >= A:
                phase, cand, z = "cfm", leader, 1
        elif kappa ** gap(cand) < B:
            phase, cand, z = "elc", None, 0
        elif kappa ** gap(cand) >= A:
            z += 1
        if phase == "cfm" and z >= z_star:
            return n, cand
    return len(answers), max(votes.items(), key=lambda x: x[1])[0] if votes else None


@pytest.mark.parametrize("gamma", [0.95, 0.99, 0.995, 0.9999])
def test_confirming_stage_matches_reference_automaton(gamma):
    rng = random.Random(1)
    cfg = AdaPOConfig(gamma=gamma)
    returns = 0
    for _ in range(500):
        pool = rng.choice(["aaabbc", "aabbcd", "abcdef", "aaaab", None])
        answers = [rng.choice(pool) if pool else rng.choice("ab") for _ in range(64)]
        rule = run_sequential(answers, n_min=8, config=cfg)
        assert (rule.samples_used(), rule.final_answer()) == _reference_rule(answers, 8, cfg)
        if rule.stopped and rule.phase == "end" and len(rule.vote_counts) > 1:
            assert rule.confirmations == rule.required_confirmations
        returns += rule.num_returns_to_electing
    if gamma >= 0.995:
        assert returns > 0  # the Confirming -> Electing transition is exercised


def test_return_to_electing_resets_confirmations():
    # p0 = 0.6 * 5/8 -> kappa = 0.375 * 4 / 0.625 = 2.4; Delta_A = 4, Delta_B = -4; Z* at gamma=0.9999 is large
    answers = ["a"] * 5 + ["b", "c", "d"] + ["b"] * 10 + ["a"] * 40
    rule = run_sequential(answers, n_min=8, config=AdaPOConfig(gamma=0.9999))
    assert rule.num_triggers >= 2 and rule.num_returns_to_electing >= 1
