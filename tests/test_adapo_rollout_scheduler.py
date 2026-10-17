"""Tests of the early-stopping rollout scheduler against a fake continuous-batching engine.

    python -m pytest tests/test_adapo_rollout_scheduler.py
"""

import random
from types import SimpleNamespace

import pytest

from verl.utils.reward_score.ttrl.adapo import run_sequential
from verl.workers.rollout.vllm_rollout.adapo_rollout import AdaPORolloutScheduler, response_valid_length


class FakeEngine:
    """Every running request decodes one token per step; rollout (p, i) answers answers[p][i]."""

    def __init__(self, answers, lengths, max_running=None):
        self.answers, self.lengths, self.max_running = answers, lengths, max_running
        self.waiting, self.running, self.progress = [], [], {}
        self.decoded_tokens = 0

    def add_request(self, request_id, prompt, params):
        self.waiting.append(request_id)
        self.progress[request_id] = 0

    def abort_request(self, request_ids):
        for rid in request_ids:
            if rid in self.running:
                self.running.remove(rid)
            if rid in self.waiting:
                self.waiting.remove(rid)

    def _key(self, request_id):
        _, _, p, i = request_id.split("-")
        return int(p), int(i)

    def step(self):
        while self.waiting and (self.max_running is None or len(self.running) < self.max_running):
            self.running.append(self.waiting.pop(0))
        outputs = []
        for rid in list(self.running):
            p, i = self._key(rid)
            self.progress[rid] += 1
            self.decoded_tokens += 1
            finished = self.progress[rid] >= self.lengths[p][i]
            # the first token encodes the answer, the remaining ones are filler
            token = self.answers[p][i] if self.progress[rid] == 1 else -1
            outputs.append(SimpleNamespace(request_id=rid, finished=finished, outputs=[SimpleNamespace(token_ids=[token])]))
            if finished:
                self.running.remove(rid)
        return outputs


def make_problem(seed, num_prompts=6, M=32):
    rng = random.Random(seed)
    answers = [[rng.choice([0, 0, 0, 1, 1, 2, 3]) for _ in range(M)] for _ in range(num_prompts)]
    lengths = [[rng.randint(5, 60) for _ in range(M)] for _ in range(num_prompts)]
    return answers, lengths


def run(policy, answers, lengths, M, N, **kw):
    engine = FakeEngine(answers, lengths, max_running=kw.pop("max_running", None))
    scheduler = AdaPORolloutScheduler(engine, None, answer_fn=lambda toks: toks[0], n_max=M, n_min=N, policy=policy, **kw)
    return scheduler.run(list(range(len(answers)))), engine


@pytest.mark.parametrize("policy,kw", [("stream", {"lookahead": 0}), ("stream", {"lookahead": 0, "max_running": 5}), ("stream", {"lookahead": 1}), ("stream", {"lookahead": 4}), ("stream", {"lookahead": 32}),
                                       ("stream", {"lookahead": 4, "max_running": 7}), ("rounds", {"parallel_batch_size": 4}),
                                       ("rounds", {"parallel_batch_size": 16})])
def test_decisions_match_one_at_a_time_sampling(policy, kw):
    M, N = 32, 8
    for seed in range(20):
        answers, lengths = make_problem(seed, M=M)
        results, _ = run(policy, answers, lengths, M, N, **dict(kw))
        for p, res in enumerate(results):
            ref = run_sequential(answers[p], N)
            assert res.num_consumed == ref.samples_used()
            assert res.stopped == ref.stopped
            # consumed rollouts are fully generated and contiguous from index 0
            assert all(res.token_ids[i] is not None for i in range(res.num_consumed))
            assert all(res.token_ids[i] is None or res.generated_tokens[i] == lengths[p][i] for i in range(M))


def test_stream_aborts_speculative_rollouts_after_stop():
    M, N = 32, 8
    answers = [[0] * M]
    lengths = [[10] * M]
    results, engine = run("stream", answers, lengths, M, N, lookahead=4)
    res = results[0]
    assert res.stopped and res.num_consumed == N  # unanimous at N
    assert sum(t is not None for t in res.token_ids) == N  # nothing beyond the stopping point completed
    assert engine.decoded_tokens == sum(res.generated_tokens)


def test_lookahead_trades_waste_for_steps():
    M, N = 32, 8
    answers, lengths = make_problem(3, num_prompts=8, M=M)
    _, slow = run("stream", answers, lengths, M, N, lookahead=1)
    _, fast = run("stream", answers, lengths, M, N, lookahead=16)
    assert fast.decoded_tokens >= slow.decoded_tokens


def test_response_valid_length_mirrors_response_mask():
    assert response_valid_length([5, 6, 2, 7], eos_token_ids=2, pad_token_id=0, max_length=8) == 3
    assert response_valid_length([5, 6, 7], eos_token_ids=[2, 9], pad_token_id=0, max_length=8) == 3
    # pad token is an EOS id and the response is shorter than the budget -> first pad counts
    assert response_valid_length([5, 6, 7], eos_token_ids=2, pad_token_id=2, max_length=8) == 4
    assert response_valid_length([5, 6, 7], eos_token_ids=2, pad_token_id=2, max_length=3) == 3
