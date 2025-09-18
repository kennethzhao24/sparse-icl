import os
import argparse
import json
import logging

import torch

from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         parse_chat_messages,)

from vllm.inputs import TextPrompt, TokensPrompt
from vllm.utils import is_list_of

from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig

from utils.data_utils import create_few_shot_example
from utils.eval_utils import get_resume_point, eval_responses

logger = logging.getLogger(__name__)


# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run Many-shot ICL')
    parser.add_argument('--model', type=str, default='google/gemma-3-1b-it',
                      help='Model name or path')
    parser.add_argument('--task', type=str, default='math500',
                      help='Task name')
    parser.add_argument('--num-shots', type=int, default=0,
                      help='Number of few-shot examples to include')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                      help='Data type for model (e.g., bfloat16, float16)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=1024,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--max_seq_len', type=int, default=32768,
                      help='Maximum number of tokens for KV cache')
    parser.add_argument('--exp_name', type=str, default='baseline',
                      help='Experiment name')
    return parser.parse_args()
   


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    # Create outputs directory if it doesn't exist
    output_path = f'./outputs/vllm_main/{args.exp_name}_{args.num_shots}_shot/{args.model}'
    os.makedirs(output_path, exist_ok=True)

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[args.task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)
    eval_data = handler.load_and_filter_dataset(0, -1) # start from 0, load all
    remaining_data = handler.process_remaining_data(eval_data, {})

    # Modify each problem to include few-shot examples
    for index in range(len(remaining_data)):
        few_shot_example = create_few_shot_example(args.task, remaining_data, args.num_shots, index)
        remaining_data[index]['problem'] = few_shot_example
    
    few_shot_data = remaining_data

    conversations = handler.make_conversations(
        few_shot_data,
        None, # str(model_config.system_prompt),
        None, # model_config.user_template,
        None, # model_config.assistant_prefill,
    )
    total_samples = len(conversations)
    print(f"Total samples in the dataset: {total_samples}")

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for tensor parallelism")
    
    model = LLM(model=args.model, 
                tensor_parallel_size=num_gpus,
                max_model_len=args.max_seq_len,
                dtype=args.dtype,
                enforce_eager=True)
    # Configure sampling parameters to return logits
    sampling_params = SamplingParams(temperature=0.0, 
                                     logprobs=5, 
                                     max_tokens=args.max_tokens, 
                                     seed=args.seed)

    # Process in batches
    qa_pairs = []
    jsonl_path = f'{output_path}/qa_pairs_{args.dtype}_bs_{args.batch_size}.jsonl'
    
    start_point = get_resume_point(output_path, args.batch_size)
    
    for batch_start in range(start_point, total_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_samples)
        current_batch = conversations[batch_start:batch_end]
        print(f"Processing batch {batch_start//args.batch_size + 1}/{(total_samples + args.batch_size - 1)//args.batch_size}")
        
        tokenizer = model.get_tokenizer()
        model_config = model.llm_engine.get_model_config()
        prompts = []

        for msgs in current_batch:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.
            conversation, mm_data = parse_chat_messages(
                msgs,
                model_config,
                tokenizer,
                content_format='string',
            )

            prompt_data = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                chat_template=None,
                add_generation_prompt=True,
                continue_final_message=False,
                tools=None,
            )

            if is_list_of(prompt_data, int):
                prompt = TokensPrompt(prompt_token_ids=prompt_data)
            else:
                prompt = TextPrompt(prompt=prompt_data)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            prompts.append(prompt)
        
        # Generate with logits for current batch
        response = model.generate(prompts, sampling_params=sampling_params)
        # Extract output text and logits for each sample in the batch
        qa_pairs = []
        for idx, output in enumerate(response):
            global_idx = batch_start + idx
            generated_text = output.outputs[0].text
            token_logprobs = output.outputs[0].logprobs
            # Create tensors from token_logprobs
            num_tokens = len(token_logprobs)
            token_ids = torch.zeros((num_tokens, 5), dtype=torch.long)
            logprobs = torch.zeros((num_tokens, 5), dtype=torch.float32)
            # Save QA pair to JSONL file
            qa_pair = {
                "problem_id": global_idx,
                "question": current_batch[idx],
                "model_answer": generated_text,
            }

            for i, logprobs_dict in enumerate(token_logprobs):
                # Extract token IDs in order of rank
                sorted_items = sorted(logprobs_dict.items(), key=lambda x: x[1].rank)
                for j, (token_id, L) in enumerate(sorted_items):
                    token_ids[i, j] = token_id
                    logprobs[i, j] = L.logprob
            
            torch.save(token_ids, f'{output_path}/problem_{global_idx}_{args.task}_token_ids_bs_{args.batch_size}_{args.dtype}_max_tokens_{args.max_tokens}.pt')
            torch.save(logprobs, f'{output_path}/problem_{global_idx}_{args.task}_logprobs_bs_{args.batch_size}_{args.dtype}_max_tokens_{args.max_tokens}.pt')
            print(f"Saved tensors for problem {global_idx}")
            
            qa_pairs.append(qa_pair)
    
        with open(jsonl_path, 'a') as f:
            for qa_pair in qa_pairs:
                f.write(json.dumps(qa_pair) + '\n')
        print(f"Saved QA pairs to for batch {batch_start//args.batch_size + 1}")


    eval_responses(args, jsonl_path, logger)