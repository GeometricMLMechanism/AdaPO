import json
import os
from collections import Counter
from typing import List, Optional

from verl.utils.reward_score.ttrl.adapo import AdaPOConfig, AdaPOStoppingRule, run_sequential
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify


class TokenTracker:
    """Process-global bookkeeping of rollout budget and AdaPO statistics (driver process only)."""

    total_tokens = 0  # tokens of rollouts consumed by the stopping rule (the paper's rollout budget)
    total_completions = 0  # rollouts consumed by the stopping rule
    total_generated_tokens = 0  # tokens actually decoded, including speculative / aborted / unused rollouts
    total_generated_completions = 0  # rollouts actually started
    total_question = 0
    current_epoch = 0
    epoch_samples = {}  # {epoch: [rollouts consumed per question]}
    epoch_val_metrics = {}  # {epoch: validation metrics}
    epoch_majority_ratios = {}  # {epoch: [majority ratio per question]}
    epoch_z_star = {}  # {epoch: [Z*(gamma) per question]}
    epoch_z_star_history = {}  # {epoch: [[Z*(gamma) per rollout] per question]}
    epoch_p_final = {}  # {epoch: [p_0 per question]}
    epoch_kappa_final = {}  # {epoch: [kappa per question]}

    _PER_EPOCH_LISTS = (
        "epoch_samples",
        "epoch_majority_ratios",
        "epoch_z_star",
        "epoch_z_star_history",
        "epoch_p_final",
        "epoch_kappa_final",
    )

    @classmethod
    def reset(cls):
        cls.total_tokens = 0
        cls.total_completions = 0
        cls.total_generated_tokens = 0
        cls.total_generated_completions = 0
        cls.total_question = 0
        cls.current_epoch = 0
        cls.epoch_val_metrics = {}
        for name in cls._PER_EPOCH_LISTS:
            setattr(cls, name, {})

    @classmethod
    def set_epoch(cls, epoch):
        cls.current_epoch = epoch
        for name in cls._PER_EPOCH_LISTS:
            getattr(cls, name).setdefault(epoch, [])

    @classmethod
    def _append(cls, name, value):
        getattr(cls, name).setdefault(cls.current_epoch, []).append(value)

    @classmethod
    def record_val_metrics(cls, val_metrics):
        if val_metrics:
            cls.epoch_val_metrics[cls.current_epoch] = val_metrics

    @classmethod
    def record_question(cls, num_tokens, num_completions, generated_tokens, generated_completions, samples_used, majority_ratio, rule: AdaPOStoppingRule):
        cls.total_tokens += num_tokens
        cls.total_completions += num_completions
        cls.total_generated_tokens += generated_tokens
        cls.total_generated_completions += generated_completions
        cls.total_question += 1
        cls._append("epoch_samples", samples_used)
        cls._append("epoch_majority_ratios", majority_ratio)
        cls._append("epoch_z_star", rule.required_confirmations if rule.required_confirmations is not None else 1)
        cls._append("epoch_z_star_history", list(rule.z_history))
        cls._append("epoch_p_final", rule.p0 if rule.p0 is not None else 0.0)
        cls._append("epoch_kappa_final", rule.kappa_history[-1] if rule.kappa_history else 0.0)

    @classmethod
    def save_all(cls, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        files = {
            "epoch_samples.json": cls.epoch_samples,
            "epoch_val_metrics.json": cls.epoch_val_metrics,
            "epoch_majority_ratios.json": cls.epoch_majority_ratios,
            "epoch_z_star.json": cls.epoch_z_star,
            "epoch_z_star_history.json": cls.epoch_z_star_history,
            "epoch_p_final.json": cls.epoch_p_final,
            "epoch_kappa_final.json": cls.epoch_kappa_final,
        }
        for filename, content in files.items():
            with open(os.path.join(output_dir, filename), "w") as f:
                json.dump(content, f, indent=2)
        return list(files)

    @classmethod
    def get_epoch_z_star_stats(cls, epoch=None):
        """Mean / min / max / std of Z*(gamma) over the questions of an epoch."""
        if epoch is None:
            epoch = cls.current_epoch
        z_values = cls.epoch_z_star.get(epoch, [])
        if not z_values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        mean_z = sum(z_values) / len(z_values)
        std_z = (sum((z - mean_z) ** 2 for z in z_values) / len(z_values)) ** 0.5
        return {"mean": mean_z, "min": min(z_values), "max": max(z_values), "std": std_z, "count": len(z_values)}


def _adapo_metrics(rule: AdaPOStoppingRule, solutions, ground_truth, task, extra_info, num_tokens, generated_tokens, generated_completions):
    """
    Pseudo-label rewards and logging metrics.

    ``solutions`` are the responses consumed by the stopping rule; each of them is rewarded against
    the AdaPO pseudo-label (only the first ``N`` are later used for the policy update).
    """
    estimated_label = rule.final_answer()
    samples_used = rule.samples_used()
    consumed_votes = Counter(a for a in rule.answers if a is not None)

    if estimated_label is not None:
        hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0] else 0.0
        rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    else:
        hit_rate = 0.0
        rewards = [0.0] * len(solutions)
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    rewards_hit_rate = sum(1 for r, tr in zip(rewards, true_rewards) if r == tr) / len(rewards)
    majority_count = consumed_votes[estimated_label] if estimated_label is not None else 0
    consumed_valid = sum(consumed_votes.values())

    TokenTracker.record_question(
        num_tokens=num_tokens,
        num_completions=samples_used,
        generated_tokens=generated_tokens,
        generated_completions=generated_completions,
        samples_used=samples_used,
        majority_ratio=majority_count / consumed_valid if consumed_valid else 0.0,
        rule=rule,
    )

    summary = rule.summary()
    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_count / len(solutions),
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        "majority_voting_reward": sum(rewards) / len(rewards),
        "pass@consumed": 1.0 if sum(true_rewards) >= 1 else 0.0,  # any consumed rollout correct
        "adapo/samples_used": samples_used,
        "adapo/early_stopped": 1.0 if rule.stopped else 0.0,
        "adapo/consumed_tokens": num_tokens,
        "adapo/generated_tokens": generated_tokens,
        "adapo/generated_completions": generated_completions,
    }
    epoch_z_stats = TokenTracker.get_epoch_z_star_stats()
    for k in ("mean", "min", "max", "std", "count"):
        ttrl_metrics[f"epoch/z_star_{k}"] = epoch_z_stats[k]
    ttrl_metrics.update(_compute_extra_metrics(summary, ground_truth, task, extra_info))

    # Per-question trace saved to JSON by the trainer (not logged to wandb)
    ttrl_metrics["_intermediate_info"] = {
        "samples_used": samples_used,
        "z_history": summary["z_history"],
        "p_history": summary["p_history"],
        "m_history": summary["m_history"],
        "kappa_history": summary["kappa_history"],
    }
    return rewards, ttrl_metrics


def test_time_train_metrics(
    solutions: List[Optional[str]],
    ground_truth: List[str],
    n_samples_per_prompt,
    response_lengths: List[int],
    task="math",
    extra_info=None,
    generated_tokens: Optional[List[int]] = None,
    adapo_config: Optional[AdaPOConfig] = None,
):
    """
    AdaPO pseudo-labelling of one prompt.

    Args:
        solutions: the M decoded responses in rollout-index order (None for rollouts that were
            never fully generated because the early-stopping rollout aborted / skipped them).
        response_lengths: number of valid tokens of each response (0 if not generated).
        generated_tokens: tokens actually decoded per rollout, including aborted partial ones.
            None means every rollout was fully generated (sequential_replay).

    The rollouts are replayed through the AdaPO stopping rule in index order (for the early-stopping
    rollouts this reproduces the decision taken during generation). Returns one reward per rollout
    (0 for rollouts not consumed by the rule) and the metrics dict.
    """
    assert len(solutions) == len(ground_truth) == len(response_lengths)
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    num_available = next((i for i, sol in enumerate(solutions) if sol is None), len(solutions))
    assert num_available >= n_samples_per_prompt, f"only {num_available} < N={n_samples_per_prompt} rollouts generated"
    model_answers = auto_extract(task, solutions[:num_available], extra_info=extra_info)
    rule = run_sequential(model_answers, n_samples_per_prompt, adapo_config)
    samples_used = rule.samples_used()

    if generated_tokens is None:
        total_generated_tokens = sum(response_lengths)
        generated_completions = len(solutions)
    else:
        total_generated_tokens = int(sum(generated_tokens))
        generated_completions = sum(1 for t in generated_tokens if t > 0)

    consumed_extra_info = extra_info[:samples_used] if isinstance(extra_info, list) else extra_info
    rewards, ttrl_metrics = _adapo_metrics(
        rule,
        solutions[:samples_used],
        ground_truth,
        task,
        consumed_extra_info,
        num_tokens=int(sum(response_lengths[:samples_used])),
        generated_tokens=total_generated_tokens,
        generated_completions=generated_completions,
    )
    rewards = list(rewards) + [0.0] * (len(solutions) - samples_used)
    return rewards, ttrl_metrics


def _compute_extra_metrics(summary, ground_truth, task, extra_info):
    """
    Diagnostics logged under ``extra/``: correctness of the answer after N rollouts and at each
    confirmation round, and statistics of Z*(gamma), p_0, m and kappa during the rollout.
    """
    extra_metrics = {}
    required_confirmations = summary["required_confirmations"]

    def is_correct(answer):
        if answer is None:
            return 0.0
        return 1.0 if auto_verify(task, [answer], [ground_truth], extra_info=extra_info)[0][0] else 0.0

    extra_metrics["extra/n_samples_correct"] = is_correct(summary["n_samples_answer"])
    extra_metrics["extra/required_confirmations"] = float(required_confirmations)

    z_history = summary["z_history"]
    if z_history:
        p_history, m_history, kappa_history = summary["p_history"], summary["m_history"], summary["kappa_history"]
        extra_metrics["extra/z_star_final"] = float(z_history[-1])
        extra_metrics["extra/z_star_mean"] = float(sum(z_history) / len(z_history))
        extra_metrics["extra/z_star_max"] = float(max(z_history))
        extra_metrics["extra/z_star_min"] = float(min(z_history))
        extra_metrics["extra/z_star_initial"] = float(z_history[0])
        extra_metrics["extra/p0"] = float(p_history[-1])
        extra_metrics["extra/m_final"] = float(m_history[-1])
        extra_metrics["extra/m_mean"] = float(sum(m_history) / len(m_history))
        extra_metrics["extra/m_max"] = float(max(m_history))
        extra_metrics["extra/kappa_final"] = float(kappa_history[-1])
        extra_metrics["extra/kappa_mean"] = float(sum(kappa_history) / len(kappa_history))
        extra_metrics["extra/kappa_max"] = float(max(kappa_history))
        extra_metrics["extra/kappa_min"] = float(min(kappa_history))

    confirmation_answers = summary["confirmation_answers"]
    for round_num in range(1, required_confirmations + 1):
        if round_num in confirmation_answers:
            confirm_answer, confirm_samples = confirmation_answers[round_num]
            extra_metrics[f"extra/confirm_{round_num}_correct"] = is_correct(confirm_answer)
            extra_metrics[f"extra/confirm_{round_num}_samples_used"] = float(confirm_samples)
        else:
            extra_metrics[f"extra/confirm_{round_num}_correct"] = 0.0
            extra_metrics[f"extra/confirm_{round_num}_samples_used"] = 0.0

    return extra_metrics


def test_time_train_metrics_eval(solutions: List[str], ground_truth: List[str], task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    counter = Counter(model_answers)
    estimated_label, majority_count = counter.most_common(1)[0]

    hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0] else 0.0
    majority_ratio = majority_count / len(solutions)

    rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    rewards_hit_rate = sum(1 for r, tr in zip(rewards, true_rewards) if r == tr) / len(rewards)
    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        "majority_voting_reward": sum(rewards) / len(rewards),
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
        "pass@1": float(true_rewards[0]) if len(true_rewards) > 0 else 0.0,
    }
    return rewards, ttrl_metrics


def post_test_time_train_metrics(solutions: List[str], ground_truth: List[str], pred_rewards: List, task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    rewards_hit_rate = sum(1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)) / len(pred_rewards)

    return {
        "post_reward_accuracy": rewards_hit_rate,
        "post_ground_truth_ratio": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
        "post_pass@1": float(true_rewards[0]) if len(true_rewards) > 0 else 0.0,
    }
