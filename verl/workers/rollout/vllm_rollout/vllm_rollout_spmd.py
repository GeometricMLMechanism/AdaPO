# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3"):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp
                )
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @torch.no_grad()
    def generate_sequences_adaptive(
        self,
        prompts: DataProto,
        tokenizer,
        n_votes_per_prompt: int,
        n_samples_per_prompt: int,
        parallel_batch_size: int,
        task: str = "math",
        adapo_config: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[DataProto, Dict[str, Any]]:
        """
        AdaPO batch-sequential rollout (Algorithm 1, batch-sequential variant).

        In every round, each still-active prompt receives min(parallel_batch_size, M - n_p)
        generation slots in a single vLLM call. The AdaPO stopping rule then consumes the new
        answers of every prompt in order; prompts whose rule fired, or that reached
        M = `n_votes_per_prompt` responses, are removed from the active set.

        Returns:
            Tuple of (DataProto with all generated responses, adaptive_info dict)
            adaptive_info contains per-prompt metadata: token counts, wallclock times,
            stopping states, and the AdaptiveStoppingState objects.
        """
        from verl.utils.reward_score.ttrl.adapo import AdaPOConfig
        from verl.utils.reward_score.ttrl.ttt_metrics import AdaptiveStoppingState

        adapo_config = AdaPOConfig.from_dict(adapo_config)
        
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        num_prompts = idx.size(0)
        
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(num_prompts)], dtype=object
            )
        
        if "multi_modal_data" in non_tensor_batch:
            all_vllm_inputs = []
            raw_ids_list = list(non_tensor_batch.pop("raw_prompt_ids"))
            mm_list = list(non_tensor_batch.pop("multi_modal_data"))
            for raw_prompt_ids, multi_modal_data in zip(raw_ids_list, mm_list):
                all_vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            raw_ids_list = list(non_tensor_batch.pop("raw_prompt_ids"))
            all_vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in raw_ids_list
            ]
        
        for input_data in all_vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )
        
        stopping_states = [
            AdaptiveStoppingState(n_samples_per_prompt, task, adapo_config) for _ in range(num_prompts)
        ]
        
        all_responses_per_prompt = [[] for _ in range(num_prompts)]
        all_token_counts_per_prompt = [[] for _ in range(num_prompts)]
        prompt_done = [False] * num_prompts
        wallclock_per_prompt = [0.0] * num_prompts
        prompt_start_times = [time.time()] * num_prompts
        
        total_generated_so_far = [0] * num_prompts
        overall_start_time = time.time()
        
        max_batches = (n_votes_per_prompt + parallel_batch_size - 1) // parallel_batch_size
        
        for batch_round in range(max_batches):
            active_prompt_indices = [p for p in range(num_prompts) if not prompt_done[p]]
            if not active_prompt_indices:
                break
            
            remaining = {
                p: min(parallel_batch_size, n_votes_per_prompt - total_generated_so_far[p])
                for p in active_prompt_indices
            }
            
            batch_vllm_inputs = []
            batch_prompt_map = []
            for p in active_prompt_indices:
                n_this = remaining[p]
                if n_this <= 0:
                    prompt_done[p] = True
                    continue
                for _ in range(n_this):
                    batch_vllm_inputs.append(all_vllm_inputs[p])
                    batch_prompt_map.append(p)
            
            if not batch_vllm_inputs:
                break
            
            gen_kwargs = {
                "n": 1,
                "max_tokens": self.config.response_length,
            }
            if vllm_version != "0.3.1":
                gen_kwargs["detokenize"] = True
            for k in self.config.keys():
                if hasattr(SamplingParams(), str(k)):
                    gen_kwargs[k] = self.config.get(k)
            gen_kwargs["n"] = 1
            gen_kwargs["detokenize"] = True
            batch_sampling_params = SamplingParams(**gen_kwargs)
            
            batch_start = time.time()
            outputs = self.inference_engine.generate(
                prompts=batch_vllm_inputs,
                sampling_params=batch_sampling_params,
                use_tqdm=False,
            )
            batch_elapsed = time.time() - batch_start
            
            per_prompt_responses = {p: [] for p in active_prompt_indices}
            per_prompt_token_ids = {p: [] for p in active_prompt_indices}
            per_prompt_token_counts = {p: [] for p in active_prompt_indices}
            per_prompt_texts = {p: [] for p in active_prompt_indices}
            
            for out_idx, output in enumerate(outputs):
                p = batch_prompt_map[out_idx]
                resp_out = output.outputs[0]
                token_ids = resp_out.token_ids
                text = resp_out.text if resp_out.text else tokenizer.decode(list(token_ids), skip_special_tokens=False)
                real_token_count = len(token_ids)
                
                per_prompt_responses[p].append(list(token_ids))
                per_prompt_token_ids[p].append(token_ids)
                per_prompt_token_counts[p].append(real_token_count)
                per_prompt_texts[p].append(text)
            
            for p in active_prompt_indices:
                if prompt_done[p]:
                    continue
                
                wallclock_per_prompt[p] += batch_elapsed
                
                all_responses_per_prompt[p].extend(per_prompt_responses[p])
                all_token_counts_per_prompt[p].extend(per_prompt_token_counts[p])
                total_generated_so_far[p] += len(per_prompt_responses[p])
                
                should_stop = stopping_states[p].process_batch(
                    per_prompt_texts[p], per_prompt_token_counts[p]
                )
                
                if should_stop:
                    prompt_done[p] = True
                    wallclock_per_prompt[p] = time.time() - prompt_start_times[p]
                elif total_generated_so_far[p] >= n_votes_per_prompt:
                    prompt_done[p] = True
                    wallclock_per_prompt[p] = time.time() - prompt_start_times[p]
        
        overall_wallclock = time.time() - overall_start_time
        
        for p in range(num_prompts):
            if wallclock_per_prompt[p] == 0:
                wallclock_per_prompt[p] = overall_wallclock
        
        max_responses = max(len(r) for r in all_responses_per_prompt)
        
        all_idx = []
        all_response_tensor_list = []
        all_attention_mask_list = []
        all_position_ids_list = []
        prompt_to_response_count = []
        
        for p in range(num_prompts):
            n_resp = len(all_responses_per_prompt[p])
            prompt_to_response_count.append(n_resp)
            for resp_tokens in all_responses_per_prompt[p]:
                all_response_tensor_list.append(resp_tokens)
                all_idx.append(idx[p])
                all_attention_mask_list.append(attention_mask[p])
                all_position_ids_list.append(position_ids[p])
        
        total_rows = sum(prompt_to_response_count)
        
        response_tensor = pad_2d_list_to_length(
            all_response_tensor_list, self.pad_token_id, max_length=self.config.response_length
        ).to(idx.device)
        
        stacked_idx = torch.stack(all_idx, dim=0)
        stacked_attention_mask = torch.stack(all_attention_mask_list, dim=0)
        stacked_position_ids = torch.stack(all_position_ids_list, dim=0)
        
        seq = torch.cat([stacked_idx, response_tensor], dim=-1)
        
        resp_len = response_tensor.size(1)
        delta_position_id = torch.arange(1, resp_len + 1, device=stacked_position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(total_rows, -1)
        if stacked_position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(total_rows, 1, -1).expand(total_rows, 3, -1)
        
        response_position_ids = stacked_position_ids[..., -1:] + delta_position_id
        full_position_ids = torch.cat([stacked_position_ids, response_position_ids], dim=-1)
        
        response_attention_mask = get_response_mask(
            response_id=response_tensor, eos_token=eos_token_id, dtype=stacked_attention_mask.dtype
        )
        full_attention_mask = torch.cat((stacked_attention_mask, response_attention_mask), dim=-1)
        
        out_non_tensor = {}
        if "multi_modal_inputs" in non_tensor_batch:
            expanded_mm = []
            for p in range(num_prompts):
                for _ in range(prompt_to_response_count[p]):
                    expanded_mm.append(non_tensor_batch["multi_modal_inputs"][p])
            out_non_tensor["multi_modal_inputs"] = np.array(expanded_mm, dtype=object)
        
        batch_dict = TensorDict(
            {
                "prompts": stacked_idx,
                "responses": response_tensor,
                "input_ids": seq,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            },
            batch_size=total_rows,
        )
        
        adaptive_info = {
            "stopping_states": stopping_states,
            "prompt_to_response_count": prompt_to_response_count,
            "wallclock_per_prompt": wallclock_per_prompt,
            "overall_wallclock": overall_wallclock,
            "total_generated_per_prompt": total_generated_so_far,
            "all_token_counts_per_prompt": all_token_counts_per_prompt,
        }
        
        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        
        return DataProto(batch=batch_dict, non_tensor_batch=out_non_tensor), adaptive_info
