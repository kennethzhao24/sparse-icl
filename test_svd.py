import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def svd_decomposition(tensor):
    """
    Perform SVD on the last two dimensions of the tensor and truncate to the specified rank.
    Args:
        tensor (torch.Tensor): The input tensor of shape (..., M, N).
        rank (int): The rank for truncation.
    Returns:
        U (torch.Tensor): Left singular vectors of shape (..., M, rank).
        S (torch.Tensor): Singular values of shape (..., rank).
        Vh (torch.Tensor): Right singular vectors of shape (..., rank, N).
    """
    # Perform SVD
    U, S, Vh = torch.svd(tensor)
    return S


# Ensure a CUDA-enabled GPU is available
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. This script requires a GPU.")

# Define the device for computation
device = "cuda"

# Load the tokenizer and model, then move the model to the GPU
# Using bfloat16 can save memory and speed up computation
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.float16
).to(device)


from evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig

task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML['math500'])
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


# Input text
# input_text = "Once upon a time, in a land of code and compilers,"

input_text = conversations[0][0]['content']

# Tokenize and move input tensors to the GPU
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Perform a forward pass and generate text on the GPU
# Ensure use_cache is True to get the past_key_values
outputs = model.generate(**inputs, max_new_tokens=256,  use_cache=True, return_dict_in_generate=True)

# The past_key_values are in the 'past_key_values' key of the output dictionary
# These tensors will reside on the GPU
past_key_values = outputs.past_key_values

# Inspect the structure of the KV cache
print(f"Number of layers in KV cache: {len(past_key_values)}")

# Examine the shape of the key and value tensors for the first layer
# The shape can be inspected directly on the GPU
first_layer_cache = past_key_values[0] # type: ignore
k_rank = svd_decomposition(first_layer_cache[0])
v_rank = svd_decomposition(first_layer_cache[1])

print(k_rank)
print(v_rank)



# The shape will be in the format: [batch_size, num_heads, sequence_length, head_dim]