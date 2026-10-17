#!/bin/bash
# Single entry point for AdaPO test-time training.
#
#   bash run_adapo.sh --model qwen2.5-math-1.5b --task AMC
#   bash run_adapo.sh --model llama3.2-1b --task AIME --variant batch_sequential --parallel-batch-size 16
#   bash run_adapo.sh --model qwen2.5-math-1.5b --task MATH --adv gae -- reward_model.reward_kwargs.adapo.eta=0.8
#
# Everything after `--` is forwarded verbatim to Hydra (verl.trainer.main_ppo).
# Run `bash run_adapo.sh --help` for all options.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_adapo.sh --model MODEL --task TASK [options] [-- hydra overrides...]

Required:
  --model MODEL               qwen2.5-math-1.5b | qwen2.5-7b | llama3.2-1b | llama3.1-8b
  --task TASK                 AIME | AMC | MATH | GPQA

Options:
  --variant VARIANT           sequential (default) | batch_sequential
  --adv ESTIMATOR             grpo (default) | gae (PPO) | reinforce_plus_plus
  --max-rollouts M            maximum rollout budget per prompt (default: 64)
  --min-rollouts N            minimum rollout budget per prompt, also used for the update (default: 32)
  --parallel-batch-size B     rollouts per prompt per round, batch_sequential only (default: 8)
  --eta ETA                   degradation factor for p_0 (default: 0.6)
  --gamma GAMMA               confidence level of adaptive confirmation (default: 0.95)
  --alpha ALPHA / --beta BETA SPRT error rates (default: 0.05 / 0.05)
  --epochs E                  number of epochs over the test set (default: task preset)
  --gpus G                    GPUs per node (default: model preset)
  --model-path PATH           local path or HF id overriding the model preset
  --project NAME              wandb project (default: AdaPO)
  --logger LIST               Hydra logger list (default: "['console','wandb']")
  -h, --help                  show this message
EOF
}

MODEL="" TASK_NAME="" VARIANT="sequential" ADVANTAGE="grpo"
N_VOTES_PER_PROMPT=64 N_SAMPLES_PER_PROMPT=32 PARALLEL_BATCH_SIZE=8
ETA=0.6 GAMMA=0.95 ALPHA=0.05 BETA=0.05
EPISODE="" N_GPUS="" MODEL_PATH="" PROJECT="AdaPO" LOGGER="['console','wandb']"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --task) TASK_NAME="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --adv) ADVANTAGE="$2"; shift 2 ;;
    --max-rollouts) N_VOTES_PER_PROMPT="$2"; shift 2 ;;
    --min-rollouts) N_SAMPLES_PER_PROMPT="$2"; shift 2 ;;
    --parallel-batch-size) PARALLEL_BATCH_SIZE="$2"; shift 2 ;;
    --eta) ETA="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --epochs) EPISODE="$2"; shift 2 ;;
    --gpus) N_GPUS="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --logger) LOGGER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$TASK_NAME" ]]; then usage; exit 1; fi

# ---------------- model presets ----------------
SUFFIX_PROMPT='"\nPlease reason step by step, and put your final answer within \boxed{}."'
case "$MODEL" in
  qwen2.5-math-1.5b) HF_ID="Qwen/Qwen2.5-Math-1.5B";         TEMPERATURE=1.0; TP=1; GPUS=1; GPU_MEM=0.7; SUFFIX_PROMPT="" ;;
  qwen2.5-7b)        HF_ID="Qwen/Qwen2.5-7B";                TEMPERATURE=0.6; TP=4; GPUS=4; GPU_MEM=0.6 ;;
  llama3.2-1b)       HF_ID="meta-llama/Llama-3.2-1B-Instruct"; TEMPERATURE=0.6; TP=1; GPUS=1; GPU_MEM=0.7 ;;
  llama3.1-8b)       HF_ID="meta-llama/Llama-3.1-8B-Instruct"; TEMPERATURE=0.6; TP=4; GPUS=4; GPU_MEM=0.7 ;;
  *) echo "Unknown model preset: $MODEL" >&2; exit 1 ;;
esac
MODEL_PATH="${MODEL_PATH:-$HF_ID}"
N_GPUS="${N_GPUS:-$GPUS}"
if [[ "$TP" -gt "$N_GPUS" ]]; then TP="$N_GPUS"; fi

# ---------------- task presets ----------------
MAX_PROMPT_LENGTH=1024
case "$TASK_NAME" in
  AIME) DEFAULT_EPISODE=80; TRAIN_BATCH_SIZE=8;  MAX_PROMPT_LENGTH=512 ;;
  AMC)  DEFAULT_EPISODE=30; TRAIN_BATCH_SIZE=8 ;;
  MATH) DEFAULT_EPISODE=10; TRAIN_BATCH_SIZE=32 ;;
  GPQA) DEFAULT_EPISODE=15; TRAIN_BATCH_SIZE=32 ;;
  *) echo "Unknown task: $TASK_NAME" >&2; exit 1 ;;
esac
TASK="${TASK_NAME}-TTT"
EPISODE="${EPISODE:-$DEFAULT_EPISODE}"

case "$VARIANT" in
  sequential|batch_sequential) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
esac
if [[ "$VARIANT" == "batch_sequential" && "$N_GPUS" -ne 1 ]]; then
  echo "The batch_sequential variant currently supports a single GPU (got --gpus $N_GPUS)." >&2
  exit 1
fi

MAX_RESPONSE_LENGTH=3072
VAL_N=16
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/data"
if [[ ! -f "$DATA_DIR/$TASK/train.parquet" ]]; then
  (cd "$DATA_DIR" && python preprocess.py --data_source "$TASK")
fi

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
EXPERIMENT="AdaPO-${VARIANT}-${TASK}-${MODEL}-${ADVANTAGE}"
LOG_NAME="${DATE}-${EXPERIMENT}"
OUTPUT_DIR="$ROOT_DIR/checkpoints/${PROJECT}/${EXPERIMENT}/${DATE}-${TIME_TAG}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

SUFFIX_ARGS=()
if [[ -n "$SUFFIX_PROMPT" ]]; then SUFFIX_ARGS=("data.suffix_prompt=$SUFFIX_PROMPT"); fi

cd "$ROOT_DIR"
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.mode="train" \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.variant=$VARIANT \
  reward_model.reward_kwargs.parallel_batch_size=$PARALLEL_BATCH_SIZE \
  reward_model.reward_kwargs.adapo.alpha=$ALPHA \
  reward_model.reward_kwargs.adapo.beta=$BETA \
  reward_model.reward_kwargs.adapo.gamma=$GAMMA \
  reward_model.reward_kwargs.adapo.eta=$ETA \
  data.train_files=["$DATA_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  ${SUFFIX_ARGS[@]+"${SUFFIX_ARGS[@]}"} \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$MODEL_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger="$LOGGER" \
  trainer.project_name=$PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=2 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
