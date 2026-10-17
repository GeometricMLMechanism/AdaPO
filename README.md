<div align="center">

# AdaPO: Towards Principled Efficient Rollout Allocation in Test-Time Training

**NeurIPS 2026**

Youkang Wang, Jian Wang, Tianyi Zeng, Xiaoyong Wei, Qing Li
<br>The Hong Kong Polytechnic University

</div>

Test-time training methods such as [TTRL](https://github.com/PRIME-RL/TTRL) build label-free rewards by majority voting over a **fixed, large** rollout budget for every query. Many queries reach consensus long before that budget is used up.

**AdaPO** replaces fixed-budget majority voting with an adaptive two-stage procedure built on a Bayesian sequential probability ratio test (SPRT):

<p align="center"><img src="assets/flowchart.png" width="70%" alt="AdaPO state machine: Electing, Confirming, End"></p>

* **Electing.** After a minimum of `N` rollouts, the leader `a_L` is triggered once its likelihood ratio against the runner-up `a_R`, `κ^(V(a_L) − V(a_R))`, reaches `A = (1−β)/α`.
* **Confirming.** Each later rollout at which the trigger condition holds counts as one confirmation. Sampling stops once the leader has `Z*(γ)` confirmations, which bounds the posterior error of accepting it by `1−γ`.
* **Update.** The pseudo-label from the stopping rule rewards the first `N` rollouts with `1[answer = pseudo-label]`, and those rollouts drive any policy-gradient algorithm (GRPO, PPO, REINFORCE++).

Here `κ = p₀(m−1)/(1−p₀)`. The success probability `p₀` is estimated **once**, from the first `N` rollouts, as `η` × (leader's vote share). It is then held fixed for the rest of the rollout process.

AdaPO saves over 40% of rollout tokens on GPQA at competitive accuracy.

## Two variants

| | `variant=sequential` | `variant=batch_sequential` |
|---|---|---|
| Decision granularity | after every rollout | after every parallel generation round |
| Rollout generation | all `M` rollouts per prompt in one vLLM call; they are then fed to the stopping rule one at a time, and only the rollouts consumed before it fires count toward the budget | rounds of `B_par` rollouts per active prompt; prompts are dropped as soon as the rule fires, so later rounds are never generated |
| Token accounting | rollouts consumed up to the stopping point | every rollout in the executed rounds, including those after the stopping point in the last round |
| Multi-GPU | yes | single GPU (one rollout worker) |

Both variants use the same stopping rule (`verl/utils/reward_score/ttrl/adapo.py`). Given the same sequence of answers, they elect the same pseudo-label at the same stopping point.

## Installation

```bash
git clone https://github.com/GeometricMLMechanism/AdaPO.git
cd AdaPO
conda create -n adapo python=3.10 -y && conda activate adapo
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e .
```

Optional: `wandb login` (or pass `--logger "['console']"`), and `huggingface-cli login` for the gated Llama models.

## Data

`data/` contains the four test-time training benchmarks (AIME 2024, AMC 2023, MATH-500, GPQA), plus MATH-500 split by difficulty level (`MATH-L{1..5}-TTT`, JSON only). `run_adapo.sh` converts the JSON files into the parquet files verl reads on first use. To convert them manually:

```bash
cd data && python preprocess.py --data_source AIME-TTT AMC-TTT MATH-TTT GPQA-TTT
```

## Running AdaPO

`run_adapo.sh` is the single entry point. It holds the model and task presets from the paper.

```bash
# Sequential variant, Qwen2.5-Math-1.5B on AMC 2023, GRPO
bash run_adapo.sh --model qwen2.5-math-1.5b --task AMC

# Batch-sequential variant, Llama-3.2-1B-Instruct on AIME 2024
bash run_adapo.sh --model llama3.2-1b --task AIME --variant batch_sequential --parallel-batch-size 16

# PPO / REINFORCE++ instead of GRPO
bash run_adapo.sh --model qwen2.5-math-1.5b --task MATH --adv gae
bash run_adapo.sh --model qwen2.5-7b --task AIME --adv reinforce_plus_plus

# Ablations: degradation factor, confidence level, error rates, budgets
bash run_adapo.sh --model qwen2.5-math-1.5b --task AMC --eta 0.8 --gamma 0.99 --alpha 0.07 --beta 0.07
bash run_adapo.sh --model qwen2.5-math-1.5b --task AMC --min-rollouts 16 --max-rollouts 64

# Any other verl/Hydra option goes after `--`
bash run_adapo.sh --model qwen2.5-math-1.5b --task AMC -- trainer.test_freq=5
```

Run `bash run_adapo.sh --help` for every option.

| Model preset | HF model | GPUs | Rollout temperature |
|---|---|---|---|
| `qwen2.5-math-1.5b` | Qwen/Qwen2.5-Math-1.5B | 1 | 1.0 |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B | 4 | 0.6 |
| `llama3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | 1 | 0.6 |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | 4 | 0.6 |

Defaults: `M=64` maximum rollouts, `N=32` minimum rollouts (also used for the update), `α=β=0.05`, `γ=0.95`, `η=0.6`, 3k-token responses, and 16 validation samples at temperature 0.6 / top-p 0.95. Epochs: AIME 80, AMC 30, MATH 10, GPQA 15.

### AdaPO hyper-parameters

They live under `reward_model.reward_kwargs` in `verl/trainer/config/ppo_trainer.yaml`:

```yaml
reward_model:
  reward_kwargs:
    n_votes_per_prompt: 64        # M, maximum rollout budget
    n_samples_per_prompt: 32      # N, minimum rollout budget / rollouts used for the update
    variant: sequential           # sequential | batch_sequential
    parallel_batch_size: 8        # B_par (batch_sequential only)
    adapo:
      alpha: 0.05
      beta: 0.05
      gamma: 0.95
      eta: 0.6                    # p_0 = eta * leader share of the first N rollouts (then frozen)
      min_num_candidates: 5       # lower bound on the number of candidate answers m
    log_dir: ${trainer.default_local_dir}/adapo_logs
```

## Outputs

Training metrics go to the configured logger (wandb by default):

* `ttrl/cumulative_tokens`, `ttrl/cumulative_completions`: rollout budget consumed so far
* `train/adapo/samples_used`, `train/adapo/early_stopped`: rollouts consumed per prompt, and the fraction of prompts stopped before `M`
* `train/label_accuracy`: accuracy of the AdaPO pseudo-labels (uses ground truth, logging only)
* `train/extra/*`: `Z*(γ)`, `p₀`, `m`, `κ` statistics and per-confirmation-round label accuracy
* `val/...`: mean@16 / pass@16 / pass@1 on the test set

Per-question traces (extracted answers, `Z*`, `p₀`, `κ`, rollouts used) and per-epoch summaries are written as JSON to `<trainer.default_local_dir>/adapo_logs/`.

## Code map

| File | Content |
|---|---|
| `verl/utils/reward_score/ttrl/adapo.py` | AdaPO stopping rule (Algorithm 1), pure Python |
| `verl/utils/reward_score/ttrl/ttt_metrics.py` | pseudo-label rewards, token accounting, metrics |
| `verl/workers/reward_manager/ttrl.py` | reward manager for both variants |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | `generate_sequences_adaptive`: batch-sequential rollout |
| `verl/trainer/ppo/ray_trainer.py` | training loop, plus the variable responses-per-prompt handling for batch-sequential |
| `tests/test_adapo_stopping_rule.py` | unit tests for the stopping rule (`python -m pytest tests/`) |

## Acknowledgements

This code builds on [TTRL](https://github.com/PRIME-RL/TTRL) and [verl](https://github.com/volcengine/verl); the benchmark files come from TTRL. Both projects are Apache-2.0 licensed, and so is this repository (see `LICENSE` and `Notice.txt`).

## Citation

```bibtex
@inproceedings{wang2026adapo,
  title     = {Towards Principled Efficient Rollout Allocation in Test-Time Training},
  author    = {Wang, Youkang and Wang, Jian and Zeng, Tianyi and Wei, Xiaoyong and Li, Qing},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
