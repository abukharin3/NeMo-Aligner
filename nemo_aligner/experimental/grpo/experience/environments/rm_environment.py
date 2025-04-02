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
from nemo_aligner.utils.distributed import run_if_model_parallel_src, broadcast_2d_tensor_within_mp
from nemo_aligner.utils.server_utils import FutureResult

def _str_list2numpy(str_list) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")

def get_future_result(future, *keys):
    """It waits for the result of the future to be ready, gets the value with the given key,
    and broadcasts it to the model parallel group. Then it returns it as output.
    """
    output = None if future is None else future.result()

    results = []

    for key in keys:

        result = None
        if output is not None:
            result = torch.tensor(output[key], device=torch.cuda.current_device())

        ten = broadcast_2d_tensor_within_mp(result)

        results.append(ten)

    if len(results) == 1:
        return results[0]

    return results

class RMFutureResult(FutureResult):
    def __init__(self, rm_future):
        self.rm_future = rm_future

    def result(self):
        rewards = get_future_result(self.rm_future, "rewards")

        self.rm_future = None
        return rewards.flatten()

class RMEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        
        print()
        server_dict = {"rm": (cfg.servers.rm.ip, cfg.servers.rm.port)}

        self.communicator = HTTPCommunicator.create_http_communicator_from_dict(server_dict)
        self.communicator.print_server_dict()
        
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

            rm_future = run_if_model_parallel_src(
                self.communicator.send_data_to_server, server_name="rm", data=send_data,
            )
            
            return RMFutureResult(rm_future)
        return RMFutureResult(None)

    def finish_step(self, future):
        """
        Process the result from the reward model server.
        """
        
        # Get the result from the future
        rewards = future.result()

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
