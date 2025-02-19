# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf import DictConfig
from concurrent import futures

import torch
from collections import Counter

from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.utils import masked_mean
from nemo_aligner.experimental.grpo.experience.interfaces import EnvironmentInterface
from nemo_aligner.servers.http_communicator import FlaskCommunicator

class IFEvalEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        self.executor = futures.ThreadPoolExecutor()
        self.communicator = FlaskCommunicator(cfg.servers)
        
        print(f"Started IFEvalEnvironment client with {cfg.servers}")
        
    def start_step(self, interactions, metadata):
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            responses = [''.join(interaction[1:]) for interaction in interactions]
            responses = [r.split("</think>")[-1] for r in responses]
            
            # Prepare test data for each response
            test_data = []
            for i, meta in enumerate(metadata):
                test_config = {
                    "instruction_id_list": meta["instruction_id_list"],
                    "instruction_kwargs": meta["instruction_kwargs"],
                    "prompt":interactions[i][0]
                }

                
                test_data.append(test_config)
            
            data = {
                "pred_responses": responses,
                "test_data": test_data,
            }
            return self.communicator.send_data_to_server("ifeval_verifier", data)
        return None

    def finish_step(self, future):
        # gets the future result and also broadcasts within the current MP group
        results = self.communicator.get_result(future, "rewards")

        th_rewards = torch.tensor(results).squeeze(1)
        return None, None, th_rewards, torch.ones(th_rewards.shape[0],)
    
    def global_post_process_and_metrics(self, batch):
        """
        Computes metrics for this environment given a global rollout batch.
        """
        # Example response for debugging
        table = {
            "reward": batch["rewards"][0].item(),
            "prompt": batch["prompt_sentences"][0],
            "response": batch["response_sentences"][0],
            "test_details": batch["extra_verifier_info"][0] if batch["extra_verifier_info"] else None
        }
        
        # Apply rewards only to properly ended sequences
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        
        # Group rewards by prompt for per-prompt metrics
        unique_prompts = {}
        for idx, prompt in enumerate(batch["prompt_sentences"]):
            if prompt not in unique_prompts:
                unique_prompts[prompt] = []
            unique_prompts[prompt].append(batch["rewards"][idx].item())
        
        # Calculate per-prompt metrics
        num_prompts = len(unique_prompts)
        prompts_with_perfect_solution = 0
        prompts_with_reward_signal = 0
        reward_range_per_prompt = []
        
        for prompt_rewards in unique_prompts.values():
            # Check for perfect solutions (reward = 1.0)
            if any(reward == 1.0 for reward in prompt_rewards):
                prompts_with_perfect_solution += 1
                
            # Check for reward signal (variance in rewards)
            if len(prompt_rewards) > 1 and not all(r == prompt_rewards[0] for r in prompt_rewards):
                prompts_with_reward_signal += 1
                
            # Calculate reward statistics per prompt
            if len(prompt_rewards) > 0:
                reward_range_per_prompt.append(max(prompt_rewards) - min(prompt_rewards))
        
        
        # Calculate average generation length for correct solutions
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["response_lengths"] - batch["prompt_lengths"])[batch["rewards"] == 1].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0
        
        
        metrics = {
            #"table": table,  # TODO: Implement table logging if needed
            
            # Overall accuracy metrics
            "ifeval_accuracy": batch["rewards"].mean().item(),
            
            # RL-specific metrics
            "fraction_prompts_with_perfect_solution": prompts_with_perfect_solution / num_prompts,
            "fraction_prompts_with_reward_range_gt_0": prompts_with_reward_signal / num_prompts,
            "avg_reward_range_per_prompt": sum(reward_range_per_prompt) / len(reward_range_per_prompt) if reward_range_per_prompt else 0,
            
            # Generation statistics
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        
        return batch, metrics