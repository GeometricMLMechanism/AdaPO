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

import json
import os
from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.ttrl.adapo import AdaPOConfig
from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics,
    test_time_train_metrics,
    test_time_train_metrics_adaptive,
    test_time_train_metrics_eval,
)

VARIANTS = ("sequential", "batch_sequential")


def data_source_to_task(data_source):
    if data_source in ["MATH-TTT", "AIME-TTT", "AMC-TTT"]:
        return "math"
    elif data_source in ["GPQA-TTT"]:
        return "gpqa"
    raise NotImplementedError(f"Data source {data_source} is not supported for TTRLRewardManager")


class TTRLRewardManager:
    """
    Label-free reward manager for TTRL / AdaPO.

    In ``train`` mode, up to ``n_votes_per_prompt`` (M) rollouts are generated per prompt, the
    AdaPO stopping rule decides how many of them are consumed and which pseudo-label they elect,
    and the first ``n_samples_per_prompt`` (N) rollouts are rewarded with
    ``1[answer == pseudo-label]`` for the policy update.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        reward_fn_key="data_source",
        compute_score=None,
        n_votes_per_prompt=1,
        n_samples_per_prompt=1,
        mode="eval",
        eval_n_samples=1,
        variant="sequential",
        parallel_batch_size=1,
        adapo=None,
        log_dir=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}, got {variant}"
        self.variant = variant
        self.parallel_batch_size = parallel_batch_size
        self.adapo_config = AdaPOConfig.from_dict(adapo)
        assert n_votes_per_prompt >= n_samples_per_prompt, f"For TTRL settings, n_votes_per_prompt {n_votes_per_prompt} should be greater than or equal to n_samples_per_prompt {n_samples_per_prompt}"

        self.log_dir = log_dir
        self.all_epoch_answers = {}
        self.all_epoch_confirmation_z = {}
        if self.mode == "train" and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.answers_file = os.path.join(self.log_dir, "epoch_extracted_answer.json")
            self.confirmation_z_file = os.path.join(self.log_dir, "epoch_confirmation_Z.json")
            # Resume from existing traces if the run is restarted in the same directory
            if os.path.exists(self.answers_file):
                with open(self.answers_file) as f:
                    self.all_epoch_answers = json.load(f)
            if os.path.exists(self.confirmation_z_file):
                with open(self.confirmation_z_file) as f:
                    self.all_epoch_confirmation_z = json.load(f)

        print(f"TTRLRewardManager initialized: mode={mode}, variant={variant}, M={n_votes_per_prompt}, N={n_samples_per_prompt}, eval_n_samples={eval_n_samples}, adapo={self.adapo_config}")

    def _data_source_to_task(self, data_source):
        return data_source_to_task(data_source)

    @staticmethod
    def _append_step(store, path, epoch, step, records):
        store.setdefault(f"epoch_{epoch}", {}).setdefault(f"step_{step}", []).extend(records)
        with open(path, "w") as f:
            json.dump(store, f, indent=2)

    def save_answers_to_file(self, epoch, step, answers_data):
        """Append the extracted answers of this step to ``<log_dir>/epoch_extracted_answer.json``."""
        if self.log_dir:
            self._append_step(self.all_epoch_answers, self.answers_file, epoch, step, answers_data)

    def save_confirmation_z_to_file(self, epoch, step, confirmation_z_data):
        """Append the per-question AdaPO traces of this step to ``<log_dir>/epoch_confirmation_Z.json``."""
        if self.log_dir:
            self._append_step(self.all_epoch_confirmation_z, self.confirmation_z_file, epoch, step, confirmation_z_data)

    @staticmethod
    def _confirmation_z_record(question_id, data_source, intermediate_info):
        def last(key):
            history = intermediate_info.get(key, [])
            return history[-1] if history else 0

        return {
            "question_id": question_id,
            "z_star_final": last("z_history"),
            "z_history": intermediate_info.get("z_history", []),
            "p0": last("p_history"),
            "m_final": last("m_history"),
            "m_history": intermediate_info.get("m_history", []),
            "kappa_final": last("kappa_history"),
            "kappa_history": intermediate_info.get("kappa_history", []),
            "samples_used": intermediate_info.get("samples_used", 0),
            "data_source": data_source,
        }

    def compute_post_ttrl_metrics(self, data: DataProto):
        """
        Compute post TTRL metrics for the given data.
        """
        assert len(data) % self.n_samples_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_samples_per_prompt}"
        prompt_num = len(data) // self.n_samples_per_prompt

        post_ttrl_info = {}
        post_ttrl_metrics_list = defaultdict(list)

        for prompt_i in range(prompt_num):
                group_vote_rewards = []
                group_pred_outputs = []
                group_labels = []
                group_extra_info = []
                task = None

                for i in range(self.n_samples_per_prompt):
                    data_item = data[prompt_i * self.n_samples_per_prompt + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]
                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    vote_reward = data_item.batch["acc"]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_vote_rewards.append(vote_reward)
                    group_extra_info.append(extra_info)
                
                post_ttrl_metrics = post_test_time_train_metrics(group_pred_outputs, group_labels, group_vote_rewards, task=task, extra_info=group_extra_info)
                for k, v in post_ttrl_metrics.items():
                    post_ttrl_metrics_list[k].append(v)

        for k, v in post_ttrl_metrics_list.items():
            if isinstance(v, list):
                v = np.mean(v)
                print(f"[{k}]", v)
                post_ttrl_info[k] = v
        return post_ttrl_info

    def _compute_ttrl_reward_adaptive(self, data: DataProto, adaptive_info: dict):
        """
        Batch-sequential variant: rewards from the per-prompt AdaptiveStoppingState objects built
        during generation. Prompts have a variable number of responses (one block of
        ``prompt_to_response_count[p]`` rows per prompt); the returned reward tensor holds the
        first N responses of every prompt.
        """
        reward_extra_info = defaultdict(list)
        ttrl_info = {}

        stopping_states = adaptive_info["stopping_states"]
        prompt_to_response_count = adaptive_info["prompt_to_response_count"]
        wallclock_per_prompt = adaptive_info["wallclock_per_prompt"]

        num_prompts = len(stopping_states)
        total_rows = sum(prompt_to_response_count)
        assert len(data) == total_rows, f"Data length {len(data)} != total_rows {total_rows}"

        reward_tensor = torch.zeros(
            num_prompts * self.n_samples_per_prompt,
            data.batch["responses"].shape[1],
            dtype=torch.float32,
            device=data.batch["responses"].device,
        )

        already_print_data_sources = {}
        all_ttrl_metrics = defaultdict(list)
        scores = [0.0] * total_rows
        all_answers_data = []
        all_confirmation_z_data = []

        row_offset = 0
        for prompt_i in range(num_prompts):
            n_resp = prompt_to_response_count[prompt_i]
            state = stopping_states[prompt_i]
            assert n_resp >= self.n_samples_per_prompt, f"prompt {prompt_i} has {n_resp} < N={self.n_samples_per_prompt} responses"

            data_item = data[row_offset]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            question_id = extra_info.get("index", prompt_i) if isinstance(extra_info, dict) else prompt_i
            task = self._data_source_to_task(data_source)
            group_extra_info = [data[row_offset + i].non_tensor_batch.get("extra_info", {}) for i in range(n_resp)]

            rewards, ttrl_metrics = test_time_train_metrics_adaptive(
                state, [ground_truth] * n_resp, task=task, extra_info=group_extra_info
            )
            ttrl_metrics["adaptive/wallclock_seconds"] = wallclock_per_prompt[prompt_i]
            ttrl_metrics["adaptive/n_responses_generated"] = n_resp
            intermediate_info = ttrl_metrics.pop("_intermediate_info", {})
            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

            all_answers_data.append({
                "question_id": question_id,
                "extracted_answers": state.extracted_answers,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "n_responses_generated": n_resp,
                "early_stopped": state.stopped,
                "wallclock_seconds": wallclock_per_prompt[prompt_i],
                "real_tokens": state.get_total_real_tokens(),
            })
            all_confirmation_z_data.append(self._confirmation_z_record(question_id, data_source, intermediate_info))

            for i in range(n_resp):
                scores[row_offset + i] = rewards[i]
                if i < self.n_samples_per_prompt:
                    di = data[row_offset + i]
                    prompt_length = di.batch["prompts"].shape[-1]
                    valid_response_length = di.batch["attention_mask"][prompt_length:].sum()
                    reward_tensor[prompt_i * self.n_samples_per_prompt + i, valid_response_length - 1] = rewards[i]

            already_print_data_sources.setdefault(data_source, 0)
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[adaptive] prompt_i={prompt_i}, n_resp={n_resp}, early_stopped={state.stopped}, real_tokens={state.get_total_real_tokens()}")

            row_offset += n_resp

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)

        for k, v in all_ttrl_metrics.items():
            ttrl_info[k] = np.mean(v)

        ttrl_info["all_answers_data"] = all_answers_data
        ttrl_info["all_confirmation_z_data"] = all_confirmation_z_data
        ttrl_info["adaptive/overall_wallclock"] = adaptive_info["overall_wallclock"]
        ttrl_info["adaptive/mean_wallclock_per_prompt"] = np.mean(wallclock_per_prompt)
        ttrl_info["adaptive/mean_responses_per_prompt"] = np.mean(prompt_to_response_count)
        ttrl_info["adaptive/total_real_tokens"] = sum(s.get_total_real_tokens() for s in stopping_states)
        ttrl_info["adaptive/early_stop_rate"] = np.mean([1.0 if s.stopped else 0.0 for s in stopping_states])
        return reward_tensor, reward_extra_info, ttrl_info

    def _compute_ttrl_reward(self, data: DataProto):
        """
        Sequential variant: M responses per prompt (contiguous rows), replayed in order through
        the AdaPO stopping rule.
        """
        reward_extra_info = defaultdict(list)
        ttrl_info = {}

        assert len(data) % self.n_votes_per_prompt == 0, f"Length of data {len(data)} should be divisible by n_votes_per_prompt {self.n_votes_per_prompt}"
        prompt_num = len(data) // self.n_votes_per_prompt

        reward_tensor = torch.zeros_like(data.batch["responses"][: prompt_num * self.n_samples_per_prompt], dtype=torch.float32)
        already_print_data_sources = {}
        all_ttrl_metrics = defaultdict(list)
        scores = [0.0 for _ in range(len(data))]
        all_answers_data = []
        all_confirmation_z_data = []

        for prompt_i in range(prompt_num):
            group_pred_outputs = []
            group_prompts = []
            group_labels = []
            group_extra_info = []
            group_response_lengths = []
            task = None

            for i in range(self.n_votes_per_prompt):
                data_item = data[prompt_i * self.n_votes_per_prompt + i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]
                if task is None:
                    task = self._data_source_to_task(data_source)
                    first_ground_truth, first_data_source = ground_truth, data_source
                    question_id = extra_info.get("index", prompt_i) if isinstance(extra_info, dict) else prompt_i
                elif task != self._data_source_to_task(data_source):
                    raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                group_labels.append(ground_truth)
                group_prompts.append(prompt_str)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)
                group_response_lengths.append(valid_response_length)

            rewards, ttrl_metrics = test_time_train_metrics(
                group_pred_outputs,
                group_labels,
                n_samples_per_prompt=self.n_samples_per_prompt,
                task=task,
                extra_info=group_extra_info,
                tokenizer=self.tokenizer,
                adapo_config=self.adapo_config,
            )
            intermediate_info = ttrl_metrics.pop("_intermediate_info", {})
            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)

            all_answers_data.append({
                "question_id": question_id,
                "extracted_answers": auto_extract(task, group_pred_outputs, extra_info=group_extra_info),
                "ground_truth": first_ground_truth,
                "data_source": first_data_source,
            })
            all_confirmation_z_data.append(self._confirmation_z_record(question_id, first_data_source, intermediate_info))

            for i in range(self.n_votes_per_prompt):
                if i < self.n_samples_per_prompt:
                    # reward on the last valid token of *this* response
                    reward_tensor[prompt_i * self.n_samples_per_prompt + i, group_response_lengths[i] - 1] = rewards[i]
                scores[prompt_i * self.n_votes_per_prompt + i] = rewards[i]

                already_print_data_sources.setdefault(first_data_source, 0)
                if already_print_data_sources[first_data_source] < self.num_examine:
                    already_print_data_sources[first_data_source] += 1
                    print("[prompt]", group_prompts[i])
                    print("[response]", group_pred_outputs[i])
                    print("[score]", rewards[i])

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)

        for k, v in all_ttrl_metrics.items():
            ttrl_info[k] = np.mean(v)

        # Per-question traces, saved to JSON by the trainer (not logged to wandb)
        ttrl_info["all_answers_data"] = all_answers_data
        ttrl_info["all_confirmation_z_data"] = all_confirmation_z_data
        return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):

            reward_extra_info = defaultdict(list)
            ttrl_info = {}

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            group_pred_outputs = []
            group_labels = []
            group_extra_info = []
            response_lengths = []
            already_print_data_sources = {}
            task = None
            for i in range(len(data)):
                data_item = data[i]
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]
                
                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)
                response_lengths.append(valid_response_length)

                if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        print("[prompt]", prompt_str)
                        print("[response]", response_str)
                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    if task != self._data_source_to_task(data_source):
                        raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

            rewards, verify_extra_info = auto_verify(task, group_pred_outputs, group_labels, extra_info=group_extra_info)

            for k, v in verify_extra_info.items():
                if isinstance(v, list):
                    reward_extra_info[k] += v

            for i in range(len(data)):
                reward_tensor[i, response_lengths[i] - 1] = rewards[i]

            # Compute TTRL metrics
            all_ttrl_metrics = defaultdict(list)
            prompt_num = len(data) // self.eval_n_samples
            for prompt_i in range(prompt_num):
                group_pred_outputs_ttrl = []
                group_labels_ttrl = []
                group_extra_info_ttrl = []
                task = None

                for i in range(self.eval_n_samples):
                    data_item = data[prompt_i * self.eval_n_samples + i]
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]

                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if task is None:
                        task = self._data_source_to_task(data_source)
                    else:
                        if task != self._data_source_to_task(data_source):
                            raise NotImplementedError(f"Non consistent task {task} and {self._data_source_to_task(data_source)} for TTRLRewardManager")

                    group_labels_ttrl.append(ground_truth)
                    group_pred_outputs_ttrl.append(response_str)
                    group_extra_info_ttrl.append(extra_info)
                
                _, ttrl_metrics = test_time_train_metrics_eval(group_pred_outputs_ttrl, group_labels_ttrl, task=task, extra_info=group_extra_info_ttrl)
                for k, v in ttrl_metrics.items():
                    all_ttrl_metrics[k].append(v)
            
            for k, v in all_ttrl_metrics.items():
                if isinstance(v, list):
                    v = np.mean(v)
                    print(f"[{k}]", v)
                    ttrl_info[k] = v
            
            return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict=False, adaptive_info=None):

        if self.mode == "train":
            if adaptive_info is not None:
                reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward_adaptive(data, adaptive_info)
            else:
                reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported for TTRLRewardManager")

        if return_dict:
            return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                    "ttrl_info": ttrl_info,
                }
        else:
            return reward_tensor