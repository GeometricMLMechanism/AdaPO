# Copyright 2026 The AdaPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AdaPO adaptive stopping rule (Algorithm 1 of the paper).

Rollout answers are fed one at a time into :class:`AdaPOStoppingRule`. The rule
implements the two-stage Bayesian SPRT used by AdaPO:

* **Electing**: once at least ``N`` (``n_min``) rollouts have been observed, the
  leader ``a_L`` is *triggered* whenever its likelihood ratio against the
  runner-up ``a_R``, ``kappa ** (V(a_L) - V(a_R))``, reaches ``A = (1-beta)/alpha``.
* **Confirming**: every rollout at which the trigger condition holds counts as
  one confirmation of the current leader; sampling stops once the leader has
  collected ``Z*(gamma)`` confirmations.

The success probability ``p_0`` is estimated **once**, from the first ``N``
rollouts, as the leader's vote share scaled by the degradation factor ``eta``,
and is then held fixed for the rest of the rollout process (Appendix E).

This module is pure Python so it can be unit-tested without GPUs.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, List, Optional


@dataclass
class AdaPOConfig:
    """Hyper-parameters of the AdaPO stopping rule."""

    alpha: float = 0.05  # type-I error of the SPRT
    beta: float = 0.05  # type-II error of the SPRT
    gamma: float = 0.95  # posterior confidence required by adaptive confirmation
    eta: float = 0.6  # degradation factor applied to the majority-ratio estimate of p_0
    min_num_candidates: int = 5  # lower bound on the number of candidate answers m

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "AdaPOConfig":
        if not cfg:
            return cls()
        known = {k: v for k, v in dict(cfg).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# Upper bound on p_0 so that kappa stays finite when eta * majority_ratio >= 1.
_P0_MAX = 1.0 - 1e-6
# Confirmation rounds returned when kappa <= 1. The SPRT can never trigger in that
# case (kappa ** delta <= 1 < A), so this value only shows up in the logs.
_Z_STAR_UNINFORMATIVE = 10


def compute_kappa(p0: float, m: int) -> float:
    """Likelihood-ratio base ``kappa = p_0 (m - 1) / (1 - p_0)`` (Eq. 3)."""
    return p0 * (m - 1) / (1 - p0)


def compute_required_confirmations(kappa: float, alpha: float, beta: float, gamma: float) -> int:
    """
    Number of confirmation rounds needed to reach posterior confidence ``gamma``:

        Z*(gamma) = ceil(1 + (log(gamma / (1 - gamma)) - log A) / log kappa),  A = (1 - beta) / alpha
    """
    if kappa <= 1.0:
        return _Z_STAR_UNINFORMATIVE
    if gamma <= 0.5 or gamma >= 1.0:
        return 1
    log_A = math.log((1 - beta) / alpha)
    z_star = math.ceil(1 + (math.log(gamma / (1 - gamma)) - log_A) / math.log(kappa))
    return max(1, int(z_star))


class AdaPOStoppingRule:
    """
    Sequential AdaPO stopping rule for a single prompt.

    Call :meth:`update` with each extracted answer (``None`` if no answer could be
    extracted) in generation order. It returns ``True`` as soon as the rule fires;
    further calls are ignored. The sequential variant feeds answers one rollout at
    a time, the batch-sequential variant feeds the answers of each parallel
    generation round in order and stops consuming the round once the rule fires.
    """

    def __init__(self, n_min: int, config: Optional[AdaPOConfig] = None):
        self.n_min = n_min
        self.config = config or AdaPOConfig()
        self.threshold_A = (1 - self.config.beta) / self.config.alpha

        self.num_observed = 0  # rollouts consumed so far (including unparsable ones)
        self.answers: List[Optional[Hashable]] = []  # extracted answers consumed so far
        self.vote_counts: Counter = Counter()
        self.leader_confirmations: Counter = Counter()
        # p_0 estimated from the first n_min rollouts, then frozen
        self.p0: Optional[float] = None

        self.stopped = False
        self.stopping_answer = None
        self.stopping_index: Optional[int] = None  # number of rollouts consumed when the rule fired

        # Diagnostics (logging only, they do not affect the decision)
        self.required_confirmations: Optional[int] = None
        self.z_history: List[int] = []
        self.p_history: List[float] = []
        self.m_history: List[int] = []
        self.kappa_history: List[float] = []
        self.n_min_answer = None  # majority answer after exactly n_min rollouts
        self.confirmation_answers: dict = {}  # {confirmation round: (leader, rollouts consumed)}

    # ------------------------------------------------------------------ #
    def _estimate_p0(self) -> float:
        total = sum(self.vote_counts.values())
        if total == 0:
            return 0.0
        majority_ratio = self.vote_counts.most_common(1)[0][1] / total
        return min(self.config.eta * majority_ratio, _P0_MAX)

    def _majority(self):
        if not self.vote_counts:
            return None
        return max(self.vote_counts.items(), key=lambda x: x[1])[0]

    def _stop(self, answer) -> bool:
        self.stopped = True
        self.stopping_answer = answer
        self.stopping_index = self.num_observed
        return True

    # ------------------------------------------------------------------ #
    def update(self, answer: Optional[Hashable]) -> bool:
        """Consume one rollout answer. Returns ``True`` iff the rule has fired."""
        if self.stopped:
            return True

        self.num_observed += 1
        self.answers.append(answer)
        if answer is not None:
            self.vote_counts[answer] += 1

        n = self.num_observed
        if n < self.n_min:
            return False  # no decision (and no p_0) before the minimum budget N

        if n == self.n_min:
            self.p0 = self._estimate_p0()  # estimated once, frozen afterwards
            self.n_min_answer = self._majority()

        m = max(len(self.vote_counts), self.config.min_num_candidates)
        kappa = compute_kappa(self.p0, m)
        z_star = compute_required_confirmations(kappa, self.config.alpha, self.config.beta, self.config.gamma)
        self.required_confirmations = z_star
        self.z_history.append(z_star)
        self.p_history.append(self.p0)
        self.m_history.append(m)
        self.kappa_history.append(kappa)

        if len(self.vote_counts) == 1:
            # Unanimous after at least N rollouts: no competing answer, stop.
            unanimous = next(iter(self.vote_counts))
            for r in range(1, z_star + 1):
                self.confirmation_answers.setdefault(r, (unanimous, n))
            return self._stop(unanimous)

        if len(self.vote_counts) >= 2:
            (leader, v_leader), (_, v_runner_up) = sorted(self.vote_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            try:
                likelihood_ratio = math.pow(kappa, v_leader - v_runner_up)
            except OverflowError:
                likelihood_ratio = float("inf")

            if likelihood_ratio >= self.threshold_A:
                self.leader_confirmations[leader] += 1
                confirmation_round = self.leader_confirmations[leader]
                if confirmation_round <= z_star:
                    self.confirmation_answers[confirmation_round] = (leader, n)
                if confirmation_round >= z_star:
                    return self._stop(leader)

        return False

    def update_batch(self, answers: List[Optional[Hashable]]) -> bool:
        """Consume answers in order until the rule fires. Returns ``True`` iff it fired."""
        for answer in answers:
            if self.update(answer):
                return True
        return False

    # ------------------------------------------------------------------ #
    def final_answer(self):
        """Pseudo-label: the stopping leader, or the majority vote if the budget ran out."""
        if self.stopped:
            return self.stopping_answer
        return self._majority()

    def samples_used(self) -> int:
        return self.stopping_index if self.stopped else self.num_observed

    def summary(self) -> dict:
        """Diagnostics consumed by the metric / logging code."""
        z_star = self.required_confirmations if self.required_confirmations is not None else 1
        confirmation_answers = dict(self.confirmation_answers)
        if not self.stopped:
            fallback = (self._majority(), self.num_observed)
            for r in range(1, z_star + 1):
                confirmation_answers.setdefault(r, fallback)
        return {
            "n_samples_answer": self.n_min_answer,
            "confirmation_answers": confirmation_answers,
            "required_confirmations": z_star,
            "z_history": list(self.z_history),
            "p_history": list(self.p_history),
            "m_history": list(self.m_history),
            "kappa_history": list(self.kappa_history),
            "p0": self.p0,
        }


def run_sequential(answers: List[Any], n_min: int, config: Optional[AdaPOConfig] = None) -> AdaPOStoppingRule:
    """Replay a list of answers through the stopping rule (sequential variant)."""
    rule = AdaPOStoppingRule(n_min, config)
    rule.update_batch(answers)
    return rule
