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

Rollout answers are fed one at a time into :class:`AdaPOStoppingRule`, which runs
the two-stage Bayesian SPRT of AdaPO once at least ``N`` (``n_min``) rollouts
have been observed:

* **Electing**: the leader ``a_L`` triggers when its likelihood ratio against the
  runner-up, ``kappa ** (V(a_L) - V(a_R))``, reaches ``A = (1-beta)/alpha``. The
  rule then enters Confirming for that candidate ``c = a_L`` with ``Z = 1`` (the
  trigger itself is the first confirmation).
* **Confirming** (confirming-stage theorem, Appendix G): at every later rollout, with
  ``Delta_c = V(c) - max_{a != c} V(a)``,

  - if ``kappa ** Delta_c < B = beta/(1-alpha)``, dominance has collapsed and the
    rule returns to Electing (``Z = 0``);
  - otherwise, if ``kappa ** Delta_c >= A``, the rollout is one more confirmation
    (``Z += 1``).

  Sampling stops as soon as ``Z >= Z*(gamma)``.

``p_0`` (the leader's vote share among the first ``N`` rollouts, scaled by
``eta``) and, by default, the number of candidates ``m`` are estimated **once**
from the first ``N`` rollouts and then held fixed (Appendix E), so ``kappa``,
the SPRT thresholds and ``Z*(gamma)`` are constant afterwards.

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
    # number of candidates m: "frozen" = distinct answers among the first N rollouts (estimated once,
    # like p_0), "dynamic" = distinct answers seen so far (re-estimated at every rollout)
    m_mode: str = "frozen"

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "AdaPOConfig":
        if not cfg:
            return cls()
        known = {k: v for k, v in dict(cfg).items() if k in cls.__dataclass_fields__}
        return cls(**known)


# Upper bound on p_0 so that kappa stays finite when eta * majority_ratio >= 1.
_P0_MAX = 1.0 - 1e-6
# Slack for ceil() in Z*(gamma): with the default alpha = beta = 1 - gamma the argument is exactly 1
# in exact arithmetic, and floating-point noise must not turn it into 2.
_CEIL_EPS = 1e-9

ELECTING, CONFIRMING, END = "electing", "confirming", "end"


def compute_kappa(p0: float, m: int) -> float:
    """Likelihood-ratio base ``kappa = p_0 (m - 1) / (1 - p_0)`` (Eq. 3)."""
    return p0 * (m - 1) / (1 - p0)


def compute_required_confirmations(kappa: float, alpha: float, beta: float, gamma: float) -> Optional[int]:
    """
    Number of confirmation rounds needed to reach posterior confidence ``gamma``:

        Z*(gamma) = ceil(1 + (log(gamma / (1 - gamma)) - log A) / log kappa),  A = (1 - beta) / alpha

    Returns ``None`` when ``kappa <= 1``: rollouts carry no evidence for the leader, so the
    electing stage can never trigger and Z* is undefined.

    Note that ``Z*(gamma) = 1`` whenever ``gamma / (1 - gamma) <= A``, e.g. for the default
    ``alpha = beta = 0.05, gamma = 0.95`` (both sides equal 19): the electing trigger alone already
    certifies posterior confidence ``gamma`` and sampling stops at the first trigger.
    """
    if kappa <= 1.0:
        return None
    if gamma <= 0.5 or gamma >= 1.0:
        return 1
    log_A = math.log((1 - beta) / alpha)
    z_star = math.ceil(1 + (math.log(gamma / (1 - gamma)) - log_A) / math.log(kappa) - _CEIL_EPS)
    return max(1, int(z_star))


def _likelihood_ratio(kappa: float, gap: int) -> float:
    try:
        return math.pow(kappa, gap)
    except OverflowError:
        return float("inf") if gap > 0 else 0.0


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
        assert self.config.m_mode in ("dynamic", "frozen"), self.config.m_mode
        self.threshold_A = (1 - self.config.beta) / self.config.alpha
        self.threshold_B = self.config.beta / (1 - self.config.alpha)

        self.num_observed = 0  # rollouts consumed so far (including unparsable ones)
        self.answers: List[Optional[Hashable]] = []  # extracted answers consumed so far
        self.vote_counts: Counter = Counter()
        # p_0 (and m in "frozen" mode) estimated from the first n_min rollouts, then frozen
        self.p0: Optional[float] = None
        self.m0: Optional[int] = None
        self.m: Optional[int] = None
        self.kappa: Optional[float] = None
        self.required_confirmations: Optional[int] = None  # Z*(gamma); None while kappa <= 1

        # Electing / Confirming state
        self.phase = ELECTING
        self.candidate = None  # answer being confirmed
        self.confirmations = 0  # Z: confirmations of `candidate` since entering Confirming

        self.stopped = False
        self.stopping_answer = None
        self.stopping_index: Optional[int] = None  # number of rollouts consumed when the rule fired

        # Diagnostics (logging only, they do not affect the decision)
        self.n_min_answer = None  # majority answer after exactly n_min rollouts
        self.first_trigger_index: Optional[int] = None  # rollouts consumed at the first Electing -> Confirming
        self.num_triggers = 0  # Electing -> Confirming transitions
        self.num_returns_to_electing = 0  # Confirming -> Electing transitions
        self.confirmation_answers: dict = {}  # {Z: (candidate, rollouts consumed)} of the last Confirming episode

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

    def _gap(self, answer) -> int:
        """V(answer) - max over the other candidates (0 votes if there is none)."""
        best_other = max((v for a, v in self.vote_counts.items() if a != answer), default=0)
        return self.vote_counts[answer] - best_other

    def _stop(self, answer) -> bool:
        self.phase = END
        self.stopped = True
        self.stopping_answer = answer
        self.stopping_index = self.num_observed
        return True

    def _confirm(self, candidate) -> bool:
        self.confirmations += 1
        self.confirmation_answers[self.confirmations] = (candidate, self.num_observed)
        if self.confirmations >= self.required_confirmations:
            return self._stop(candidate)
        return False

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
            self.m0 = len(self.vote_counts)
            self.n_min_answer = self._majority()

        if n == self.n_min or self.config.m_mode == "dynamic":
            num_candidates = len(self.vote_counts) if self.config.m_mode == "dynamic" else self.m0
            self.m = max(num_candidates, self.config.min_num_candidates)
            self.kappa = compute_kappa(self.p0, self.m)
            self.required_confirmations = compute_required_confirmations(self.kappa, self.config.alpha, self.config.beta, self.config.gamma)

        if len(self.vote_counts) == 1:
            # Unanimous after at least N rollouts: no competing answer, stop.
            return self._stop(next(iter(self.vote_counts)))
        if len(self.vote_counts) < 2 or self.kappa <= 1.0:
            return False  # nothing to compare, or no evidence per vote (the SPRT cannot trigger)

        if self.phase == ELECTING:
            leader = self._majority()
            if _likelihood_ratio(self.kappa, self._gap(leader)) >= self.threshold_A:
                self.phase, self.candidate, self.confirmations = CONFIRMING, leader, 0
                self.confirmation_answers = {}
                self.num_triggers += 1
                if self.first_trigger_index is None:
                    self.first_trigger_index = n
                return self._confirm(leader)  # the trigger itself is the first confirmation (Z = 1)
        else:  # CONFIRMING
            likelihood_ratio = _likelihood_ratio(self.kappa, self._gap(self.candidate))
            if likelihood_ratio < self.threshold_B:
                # dominance of the candidate collapsed: back to Electing
                self.phase, self.candidate, self.confirmations = ELECTING, None, 0
                self.num_returns_to_electing += 1
            elif likelihood_ratio >= self.threshold_A:
                return self._confirm(self.candidate)
        return False

    def update_batch(self, answers: List[Optional[Hashable]]) -> bool:
        """Consume answers in order until the rule fires. Returns ``True`` iff it fired."""
        for answer in answers:
            if self.update(answer):
                return True
        return False

    # ------------------------------------------------------------------ #
    def final_answer(self):
        """Pseudo-label: the stopping answer, or the majority vote if the budget ran out."""
        if self.stopped:
            return self.stopping_answer
        return self._majority()

    def samples_used(self) -> int:
        return self.stopping_index if self.stopped else self.num_observed

    def summary(self) -> dict:
        """Diagnostics consumed by the metric / logging code."""
        return {
            "samples_used": self.samples_used(),
            "stopped": self.stopped,
            "phase": self.phase,
            "n_samples_answer": self.n_min_answer,
            "p0": self.p0,
            "m": self.m,
            "kappa": self.kappa,
            "z_star": self.required_confirmations,
            "confirmations": self.confirmations,
            "confirmation_answers": dict(self.confirmation_answers),
            "first_trigger_index": self.first_trigger_index,
            "num_triggers": self.num_triggers,
            "num_returns_to_electing": self.num_returns_to_electing,
        }


def run_sequential(answers: List[Any], n_min: int, config: Optional[AdaPOConfig] = None) -> AdaPOStoppingRule:
    """Replay a list of answers through the stopping rule (sequential variant)."""
    rule = AdaPOStoppingRule(n_min, config)
    rule.update_batch(answers)
    return rule
