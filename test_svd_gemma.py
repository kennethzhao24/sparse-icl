import argparse
import json
import logging

import torch

from transformers import AutoTokenizer, AutoConfig
from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig

from gemma3 import Gemma3ForCausalLM

logger = logging.getLogger(__name__)

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='SVD Rank Evaluation Script')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-it',
                      help='Model name or path')
    parser.add_argument('--task', type=str, default='math500',
                      help='Task name')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                      help='Data type for model (e.g., bfloat16, float16)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
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

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[args.task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)
    eval_data = handler.load_and_filter_dataset(0, -1) # start from 0, load all
    remaining_data = handler.process_remaining_data(eval_data, {})
    conversations = handler.make_conversations(
        remaining_data,
        None, # str(model_config.system_prompt),
        None, # model_config.user_template,
        None, # model_config.assistant_prefill,
    )
    total_samples = len(conversations)
    print(f"Total samples in the dataset: {total_samples}")

    config = AutoConfig.from_pretrained(args.model)

    # print(config)

    config.vocab_size = 256000

    # print(config)
    model = Gemma3ForCausalLM(config).half().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompt = conversations[0][0]['content']

    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, 
                                 max_new_tokens=20, 
                                 use_cache=True, 
                                 return_dict_in_generate=True)
    past_key_values = outputs.past_key_values

    for i in range (len(past_key_values)):
        print(f"Layer {i}:")
        first_layer_cache = past_key_values[i] # pyright: ignore[reportOptionalSubscript]
        key_cache_shape = first_layer_cache[0].shape
        print(f"Shape of Key cache for layer {i}: {key_cache_shape}")