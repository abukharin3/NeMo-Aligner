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
import numpy as np

from nemo_aligner.experimental.grpo.experience.interfaces import EnvironmentInterface
from nemo_aligner.servers.http_communicator import HTTPCommunicator
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.utils import masked_mean
from nemo_aligner.experimental.grpo.experience.environments.metrics import calculate_pass_rate_per_prompt
from nemo_aligner.utils.distributed import run_if_model_parallel_src

def _str_list2numpy(str_list) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")

class RMEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        critic_ip_and_port = (cfg.servers.rm.ip, cfg.servers.rm.port)
        server_dict = {
            cfg.critic.name.train: critic_ip_and_port,
            cfg.critic.name.infer: critic_ip_and_port,
            cfg.critic.name.save: critic_ip_and_port,
        }

        if not cfg.combine_rm_and_critic_server:
            server_dict[cfg.reward_model.name] = (cfg.servers.rm.ip, cfg.servers.rm.port)

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        self.combine_rm_and_critic_server = self.cfg.combine_rm_and_critic_server
        self.pad_to_length = self.cfg.pad_to_length
        
        print(f"Started RMEnvironment client with {cfg.servers}")
        
    def start_step(self, interactions, metadata, is_end):
        """
        Send full text responses to the reward model server.
        """
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            responses = [''.join(interaction[1:]) for interaction in interactions]
            
            send_data = {
                "sentences": _str_list2numpy(responses),
            }
            
            return self.communicator.send_data_to_server(
                server_name=self.cfg.reward_model.name, 
                data=send_data
            )
        return None

    def finish_step(self, future):
        """
        Process the result from the reward model server.
        """
        if future is None:
            # Handle the case where we're not on the source rank
            rewards = torch.zeros(0)
            if parallel_state.model_parallel_is_initialized():
                rewards = torch.zeros(
                    1, device=torch.cuda.current_device()
                )
                torch.distributed.broadcast(
                    rewards, 
                    parallel_state.get_model_parallel_src_rank(),
                    group=parallel_state.get_model_parallel_group()
                )
            return None, None, rewards, torch.ones_like(rewards)
        
        # Get the result from the future
        result = future.result()
        rewards = torch.tensor(result["rewards"]).squeeze(1)
        
        # Broadcast the rewards to all ranks in the model parallel group
        if parallel_state.model_parallel_is_initialized():
            torch.distributed.broadcast(
                rewards, 
                parallel_state.get_model_parallel_src_rank(),
                group=parallel_state.get_model_parallel_group()
            )
        
        print('rewards shape', rewards.shape)
        return None, None, rewards, torch.ones(rewards.shape[0],)
    
    def global_post_process_and_metrics(self, batch):
        """
        Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed 
        calculations if you'd prefer for heavy metrics. 
        """
        table = {
            "reward": batch["rewards"][0].item(),
            "prompt_sentence": batch["prompt_sentences"][0],
            "response_sentence": batch["response_sentences"][0],
            "expected_answer": batch["extra_verifier_info"][0]["ground_truth"],
        }
        batch["rewards"] = batch["rewards"] * batch["is_end"] # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["response_lengths"] - batch["prompt_lengths"])[batch["rewards"] == 1].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0
        
        metrics = {
            #"table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], batch["rewards"]),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }
        
        return batch, metrics
