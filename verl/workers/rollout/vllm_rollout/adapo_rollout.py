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
Early-stopping rollout scheduler for AdaPO.

Every rollout is an individual request on a continuous-batching engine (vLLM ``LLMEngine``:
``add_request`` / ``step`` / ``abort_request``). Answers are fed to the per-prompt AdaPO
stopping rule strictly in rollout-index order, so the decisions are exactly those of drawing
the rollouts one at a time; parallelism only changes *when* a rollout is generated.

Two policies:

* ``stream`` (sequential variant). For every prompt, rollouts ``0..max(N, lookahead)-1`` are
  submitted immediately and, after that, the next ``lookahead`` rollouts beyond the consumption
  frontier are kept submitted. Each finished rollout is consumed as soon as all lower-index
  rollouts are consumed. When the rule fires, the prompt's remaining requests are aborted.
  ``lookahead <= 0`` (eager) submits all M rollouts up front. Requests are submitted in
  index-major order (rollout 0 of every prompt, then rollout 1, ...), so with the engine's FCFS
  queue the lowest-index rollouts of all prompts are decoded first; requests still waiting in the
  queue when their prompt stops are aborted before costing any compute. Larger ``lookahead``
  lowers wall-clock time and raises speculative waste (tokens decoded for rollouts beyond the
  stopping point).
* ``rounds`` (batch-sequential variant). Each prompt is generated in rounds of
  ``parallel_batch_size`` rollouts; the rule only consumes a round once all of its rollouts
  have finished. There is no barrier across prompts: a prompt starts its next round as soon
  as its previous round is processed.

The scheduler is engine-agnostic (see ``tests/test_adapo_rollout_scheduler.py``).
"""

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from verl.utils.reward_score.ttrl.adapo import AdaPOConfig, AdaPOStoppingRule

_CALL_COUNTER = itertools.count()


@dataclass
class PromptRollouts:
    """Rollout outcome of one prompt, indexed by rollout id ``0..M-1``."""

    token_ids: List[Optional[List[int]]]  # fully generated responses (None: aborted / never launched)
    generated_tokens: List[int]  # tokens actually decoded for each rollout, including aborted partial ones
    num_consumed: int  # rollouts consumed by the stopping rule
    stopped: bool  # True iff the stopping rule fired before the budget M
    wallclock: float  # seconds from the start of the call until the prompt's decision


@dataclass
class _PromptState:
    rule: AdaPOStoppingRule
    num_launched: int = 0
    inflight: Dict[int, str] = field(default_factory=dict)  # rollout id -> request id
    buffers: Dict[int, List[int]] = field(default_factory=dict)
    finished: Dict[int, List[int]] = field(default_factory=dict)
    generated_tokens: List[int] = field(default_factory=list)
    num_consumed: int = 0
    done: bool = False
    wallclock: float = 0.0


def response_valid_length(token_ids: List[int], eos_token_ids, pad_token_id: Optional[int], max_length: int) -> int:
    """
    Number of response tokens the trainer treats as valid, mirroring ``get_response_mask`` applied
    to the right-padded response: everything up to and including the first EOS token. When no EOS
    is generated but the pad token is itself an EOS id, the first padding position counts too.
    """
    eos = set(eos_token_ids) if isinstance(eos_token_ids, (list, tuple, set)) else {eos_token_ids}
    for pos, tok in enumerate(token_ids):
        if tok in eos:
            return pos + 1
    if pad_token_id in eos and len(token_ids) < max_length:
        return len(token_ids) + 1
    return len(token_ids)


class AdaPORolloutScheduler:
    def __init__(
        self,
        engine,
        sampling_params,
        answer_fn: Callable[[List[int]], Any],
        n_max: int,
        n_min: int,
        adapo_config: Optional[AdaPOConfig] = None,
        policy: str = "stream",
        lookahead: int = 16,
        parallel_batch_size: int = 8,
    ):
        """
        Args:
            engine: object exposing ``add_request(request_id, prompt, params)``, ``step()`` (returning
                outputs with ``request_id``, ``finished`` and ``outputs[0].token_ids`` holding the
                *new* tokens of this step) and ``abort_request(request_ids)``.
            sampling_params: per-request sampling parameters (``n=1``, delta outputs).
            answer_fn: maps a finished response (token ids) to its extracted answer (or None).
            n_max / n_min: rollout budgets M and N.
        """
        assert policy in ("stream", "rounds"), policy
        assert 1 <= n_min <= n_max
        self.engine = engine
        self.sampling_params = sampling_params
        self.answer_fn = answer_fn
        self.n_max = n_max
        self.n_min = n_min
        self.adapo_config = adapo_config or AdaPOConfig()
        self.policy = policy
        self.lookahead = n_max if int(lookahead) <= 0 else int(lookahead)

        self.parallel_batch_size = max(1, int(parallel_batch_size))

    # ------------------------------------------------------------------ #
    def _launch(self, p: int, prompt, state: _PromptState, count: int, call_id: int, request_map):
        for _ in range(count):
            i = state.num_launched
            request_id = f"adapo-{call_id}-{p}-{i}"
            self.engine.add_request(request_id, prompt, self.sampling_params)
            state.inflight[i] = request_id
            state.buffers[i] = []
            request_map[request_id] = (p, i)
            state.num_launched += 1

    def _num_to_launch(self, state: _PromptState) -> int:
        if state.done:
            return 0
        if self.policy == "stream":
            # keep `lookahead` rollouts submitted beyond the consumption frontier
            target = min(self.n_max, max(self.n_min, state.num_consumed + self.lookahead))
            count = target - state.num_launched
        else:  # rounds: start the next round once the previous one is fully consumed
            if state.inflight or state.num_consumed < state.num_launched:
                return 0
            if state.num_launched == 0:
                # the first N rollouts are always needed, launch them with the first round
                count = max(self.n_min, min(self.parallel_batch_size, self.n_max))
                count = min(count, self.n_max)
            else:
                count = min(self.parallel_batch_size, self.n_max - state.num_launched)
        return max(0, count)

    def _refill(self, p: int, prompt, state: _PromptState, call_id: int, request_map):
        count = self._num_to_launch(state)
        if count > 0:
            self._launch(p, prompt, state, count, call_id, request_map)

    def _consume(self, state: _PromptState, start_time: float) -> List[str]:
        """Feed finished rollouts to the rule in index order. Returns request ids to abort."""
        if self.policy == "rounds" and state.inflight:
            return []  # wait for the whole round
        while not state.done and state.num_consumed in state.finished:
            i = state.num_consumed
            state.num_consumed += 1
            fired = state.rule.update(self.answer_fn(state.finished[i]))
            if fired or state.num_consumed >= self.n_max:
                state.done = True
                state.wallclock = time.perf_counter() - start_time
        if state.done and state.inflight:
            to_abort = list(state.inflight.values())
            state.inflight.clear()
            return to_abort
        return []

    # ------------------------------------------------------------------ #
    def run(self, prompts: List[Any]) -> List[PromptRollouts]:
        call_id = next(_CALL_COUNTER)
        start = time.perf_counter()
        states = [
            _PromptState(rule=AdaPOStoppingRule(self.n_min, self.adapo_config), generated_tokens=[0] * self.n_max)
            for _ in prompts
        ]
        request_map: Dict[str, Tuple[int, int]] = {}
        # initial submission in index-major order: rollout 0 of every prompt, then rollout 1, ...
        initial = [self._num_to_launch(state) for state in states]
        for _ in range(max(initial, default=0)):
            for p, prompt in enumerate(prompts):
                if states[p].num_launched < initial[p]:
                    self._launch(p, prompt, states[p], 1, call_id, request_map)

        while any(not s.done for s in states):
            outputs = self.engine.step()
            touched = set()
            for out in outputs:
                key = request_map.get(out.request_id)
                if key is None:
                    continue
                p, i = key
                state = states[p]
                if i not in state.inflight:
                    continue  # already aborted
                new_tokens = list(out.outputs[0].token_ids)
                state.buffers[i].extend(new_tokens)
                state.generated_tokens[i] += len(new_tokens)
                if out.finished:
                    state.finished[i] = state.buffers.pop(i)
                    del state.inflight[i]
                    touched.add(p)
            to_abort = []
            for p in touched:
                to_abort.extend(self._consume(states[p], start))
            if to_abort:
                self.engine.abort_request(to_abort)
            for p in touched:
                self._refill(p, prompts[p], states[p], call_id, request_map)

        results = []
        for state in states:
            token_ids = [state.finished.get(i) for i in range(self.n_max)]
            results.append(
                PromptRollouts(
                    token_ids=token_ids,
                    generated_tokens=list(state.generated_tokens),
                    num_consumed=state.num_consumed,
                    stopped=state.rule.stopped,
                    wallclock=state.wallclock,
                )
            )
        return results
