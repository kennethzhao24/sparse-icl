import os
import json
import glob

from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig, TaskHandler
from evals.util.results import SummaryResults, save_summary

# Determine the starting point based on existing .pt files
def get_resume_point(output_path, batch_size):
    # Find all .pt files in the output directory
    pt_files = glob.glob(f'{output_path}/problem_*_token_ids_*.pt')
    if not pt_files:
        return 0  # No files exist, start from the beginning

    # Extract global_idx from filenames
    global_indices = []
    for pt_file in pt_files:
        # Filename format: problem_<global_idx>_token_ids_*.pt
        parts = os.path.basename(pt_file).split('_')
        try:
            global_idx = int(parts[1])  # Extract the global_idx
            global_indices.append(global_idx)
        except (IndexError, ValueError):
            continue

    if not global_indices:
        return 0  # No valid indices found, start from the beginning

    # Find the largest global_idx and calculate the starting batch
    max_global_idx = max(global_indices)
    resume_point = ((max_global_idx + 1) // batch_size) * batch_size
    print(f"Resuming from batch starting at index {resume_point} (max_global_idx={max_global_idx})")
    return resume_point


def score_responses(
    handler: TaskHandler,
    list_of_results: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, List[int]], int]:

    if not list_of_results:
        return 0.0, {}, 0

    total_correct = 0
    total_finish = 0
    id_to_scores = {}

    for result in tqdm(list_of_results, desc="Scoring responses"):
        # Get content from the result
        model_response = result['model_answer']
        problem_id = result['problem_id']
        problem = eval_data[problem_id]
        
        new_response_entry = handler.update_results(
            problem=problem,
            response=model_response,
        )
        
        if problem_id not in id_to_scores:
            id_to_scores[problem_id] = [0]
        id_to_scores[problem_id][0] = new_response_entry["correctness"]
        
        total_correct += new_response_entry["correctness"]
        total_finish += 1

    accuracy = round(total_correct / total_finish, 4) if total_finish else 0
    return accuracy, id_to_scores, total_finish


def eval_responses(args, jsonl_path, logger):
    responses_path = Path(jsonl_path)

    if responses_path.stat().st_size == 0:
        raise ValueError(f"Response file is empty: {responses_path}")
        
    print(f"Valid response file: {responses_path}")
    
    # Read the .jsonl file line by line and parse each line as a JSON object
    with open(responses_path, "r") as f:
        list_of_results = [json.loads(line) for line in f]
    
    # Check if the response file is a list of dictionaries
    if not all(isinstance(result, dict) for result in list_of_results):
        raise ValueError(f"Response file does not contain valid dictionaries on each line: {responses_path}")
    
    # Check if the response file is a list of dictionaries
    if not isinstance(list_of_results, list):
        raise ValueError(f"Response file is not a list of dictionaries: {responses_path}")
    
    # Obtain the correct task handler
    task = args.task
    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )
    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)
    
    raw_dataset = handler.load_and_filter_dataset(0, -1) # start from 0, load all
    eval_data = [
        row.to_dict()
        for _, row in raw_dataset.iterrows()
    ]
    
    accuracy, id_to_scores, total_finish = score_responses(handler, list_of_results, eval_data)
    logger.info(f"Accuracy: {accuracy}")
    
    num_responses_total = len(id_to_scores)

    summary_data = SummaryResults(
        accuracy=accuracy,
    )
    
    # Create outputs directory if it doesn't exist
    acc_path = f'./scoring_results/greedy'
    os.makedirs(acc_path, exist_ok=True)
    
    sanitized_model_name = args.model.replace("/", "_")
    summary_file = Path(acc_path) / f"{sanitized_model_name}_{args.exp_name}_summary.jsonl"
    save_summary(summary_file, summary_data)
    logger.info(f"Summary saved to {summary_file}")