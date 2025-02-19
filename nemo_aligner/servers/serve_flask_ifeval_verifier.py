from flask import Flask, request, jsonify
import multiprocessing as mp
from enum import Enum
import logging
from typing import Callable, Dict, List, Union
from nemo_aligner.utils.verifiers.instruction_following.instructions_registry import INSTRUCTION_DICT
import numpy as np

app = Flask(__name__)

PROCESS_COUNT = 32

class WorkerSignal(Enum):
    RUN = 0
    QUIT = 1

def instruction_following_rewards(prompt, response, args):
    """Tests response to see if instrutions are followed."""
    try:
        task_args = args
        instruction_list = task_args["instruction_id_list"]
        is_following_list = []

        for index, instruction_id in enumerate(instruction_list):
            try:
                instruction_cls = INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)

                kwargs = (
                    task_args["instruction_kwargs"][index]
                    if task_args["instruction_kwargs"][index] is not None
                    else {}
                )
                instruction.build_description(**kwargs)
                instruction_args = instruction.get_instruction_args()
                if instruction_args and "prompt" in instruction_args:
                    instruction.build_description(prompt=prompt)

                if response.strip() and instruction.check_following(response):
                    is_following_list.append(True)
                else:
                    is_following_list.append(False)
            except Exception as e:
                print(f"Error in instruction_following_rewards: {e}, task: {args}")

        low, high = 0, 1
        correctness = sum(is_following_list) / len(is_following_list)
        score = low + (high - low) * correctness
        return score, True
    except Exception as e:
        print(f"Error in instruction_following_rewards: {e}")
        return 0, False


def verify_ifeval_worker(input_queue: mp.Queue,
                      output_queue: mp.Queue):
    while True:
        signal, idx, args = input_queue.get()
        if signal == WorkerSignal.RUN:
            response, test_data = args
            prompt = test_data["prompt"]

            reward, ran = instruction_following_rewards(prompt, response, args)
            
            # Include detailed results for debugging
            output_queue.put((idx, {
                "reward": reward,
            }))
        else:
            return

# Initialize queues and workers
submit_queue = mp.Queue()
result_queue = mp.Queue()

workers = []
for _ in range(PROCESS_COUNT):
    p = mp.Process(target=verify_ifeval_worker, args=(submit_queue, result_queue))
    p.start()
    workers.append(p)

@app.route('/ifeval_verifier', methods=['POST'])
def verify_code():
    """
    Endpoint to evaluate ifeval submissions.
    Expects a JSON payload with:
    {
        "pred_responses": [str1, str2, ...],  # List of code strings to evaluate
        "test_data": [  # List of test configurations for each code submission
            {
                "prompt": prompt_str
                "instruction_kwargs": ...
                "instruction_id_list": ...
            },
            ...
        ],
    }
    """
    try:
        data = request.get_json()
        responses = data.get("pred_responses")
        test_data = data.get("test_data")

        # Validate inputs
        if not responses or not test_data:
            return jsonify({"error": "Both 'pred_responses' and 'test_data' must be provided."}), 400
        if len(responses) != len(test_data):
            return jsonify({"error": "Number of responses must match number of test configurations."}), 400

        # Validate test types
        for test_config in test_data:
            if test_config.get("prompt") not in [t.value for t in TestType]:
                return jsonify({"error": f"Invalid test type: {test_config.get('prompt')}"}), 400
            if "instruction_kwargs" not in test_config:
                return jsonify({"error": "test_data must include 'instruction_kwargs' field"}), 400

        # Submit jobs to workers
        for idx, (code, test_config) in enumerate(zip(responses, test_data)):
            submit_queue.put((WorkerSignal.RUN, idx, (code, test_config)))

        # Collect results
        rewards = np.zeros(len(responses))
        for _ in range(len(responses)):
            idx, result = result_queue.get()
            rewards[idx] = result["reward"]

        # Match the math grader format
        output_dict = {
            "rewards": rewards.reshape((rewards.shape[0], 1)).tolist(),
        }

        return jsonify(output_dict)

    except Exception as e:
        app.logger.error("An error occurred: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5568, debug=True)