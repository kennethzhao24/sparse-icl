import random
import torch

from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         parse_chat_messages,)

from vllm.inputs import TextPrompt, TokensPrompt
from vllm.utils import is_list_of

from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig



def create_few_shot_example(remaining_data, num_shots, index):
    few_shot_example = ""
    population = [number for number in range(len(remaining_data)) if number != index]
    sampled_indices = random.sample(population, num_shots)
    for i in sampled_indices:
        data_point = remaining_data[i]
        few_shot_example += data_point['problem'] + '\n Answer is' + data_point['answer'] + '\n\n'
    few_shot_example += remaining_data[index]['problem']
    return few_shot_example.strip()



if __name__ == '__main__':
    task = 'math500'  # change to your desired task
    num_shots = 4  # number of few-shot examples to include

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)
    eval_data = handler.load_and_filter_dataset(0, -1) # start from 0, load all
    remaining_data = handler.process_remaining_data(eval_data, {})

    for index in range(len(remaining_data)):
        few_shot_example = create_few_shot_example(remaining_data, num_shots, index)
        remaining_data[index]['problem'] = few_shot_example
    
    few_shot_data = remaining_data

    # print(few_shot_data[0])  # Inspect the first data point

    conversations = handler.make_conversations(
        few_shot_data,
        None, # str(model_config.system_prompt),
        None, # model_config.user_template,
        None, # model_config.assistant_prefill,
    )
    total_samples = len(conversations)
    print(f"Total samples in the dataset: {total_samples}")

    # print(conversations[0])  # Inspect the first conversation

   # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for tensor parallelism")
    
    model = LLM(model='google/gemma-3-1b-it', 
                tensor_parallel_size=num_gpus,
                # max_model_len=length_used,
                dtype='bfloat16',
                enforce_eager=True)

    # Configure sampling parameters to return logits
    sampling_params = SamplingParams(temperature=0.0, logprobs=5, max_tokens=512, seed=1)

    tokenizer = model.get_tokenizer()
    model_config = model.llm_engine.get_model_config()

    current_batch = conversations[:4]

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
    
    response = model.generate(prompts, sampling_params=sampling_params)

