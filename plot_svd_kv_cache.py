import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
# Define the device for computation
device = "cuda"

# Load the tokenizer and model, then move the model to the GPU
# Using bfloat16 can save memory and speed up computation
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16
).to(device)

model.eval()  

pg_19 = load_dataset("emozilla/pg19", split="test")  # Load a small subset for testing


# Collect outputs for all layers in these lists
collected_k_outputs = []
collected_v_outputs = []
def k_proj_hook(module, input, output):
    """
    module: The layer that produced this output (k_proj).
    input:  The input to k_proj.
    output: The output from k_proj (shape [batch_size, seq_len, hidden_dim]).
    """
    # Detach to avoid growing the autograd graph
    collected_k_outputs.append(output.detach().cpu())

def v_proj_hook(module, input, output):
    """
    Same logic as k_proj_hook, but for v_proj.
    """
    collected_v_outputs.append(output.detach().cpu())


num_layers = len(model.model.language_model.layers)

hooks_k = []
hooks_v = []
for layer_idx in range(num_layers):
    # Access the i-th layer
    layer = model.model.language_model.layers[layer_idx].self_attn

    # Register forward hooks
    hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
    hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
    
    hooks_k.append(hook_k)
    hooks_v.append(hook_v)


n = 20000

inputs = tokenizer(
    pg_19[6]['text'][:n],
    return_tensors="pt",
).to(device)

with torch.no_grad():
    model(inputs['input_ids'], attention_mask=inputs['attention_mask'])


for hook in hooks_k:
    hook.remove()
for hook in hooks_v:
    hook.remove()

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
    _, S, _ = torch.svd(tensor)
    return S

import matplotlib.pyplot as plt
import numpy as np

# rank distribution
for i in range(len(collected_k_outputs)):
    # num_key_value_heads = model.config.text_config.num_key_value_heads

    # num_key_value_heads = model.config.num_key_value_heads

    # k_cache_pre_rope = collected_k_outputs[i].float().reshape(-1, collected_k_outputs[i].shape[1] * num_key_value_heads, collected_k_outputs[i].shape[2] // num_key_value_heads)
    # v_cache_pre_rope = collected_v_outputs[i].float().reshape(-1, collected_k_outputs[i].shape[1] * num_key_value_heads, collected_k_outputs[i].shape[2] // num_key_value_heads)

    k_cache_pre_rope = collected_k_outputs[i].float()
    v_cache_pre_rope = collected_v_outputs[i].float()

    k_rank = svd_decomposition(k_cache_pre_rope)
    v_rank = svd_decomposition(v_cache_pre_rope)

    FIGURE_SIZE = (8, 5)   # (width, height) in inches             # Output resolution
    plt.rcParams['font.size'] = 14  # Set default font size
    plt.rcParams['font.family'] = 'serif' # Set font family, e.g., 'serif', 'sans-serif', 'monospace'

    # k_rank and v_rank are torch.Tensors of shape (batch, seq_len, rank)
    # For dot plot, flatten or select a slice (e.g., first batch and first sequence)
    k_vals = k_rank.cpu().numpy().flatten()
    v_vals = v_rank.cpu().numpy().flatten()
    k_vals = np.sort(k_vals)[::-1]
    v_vals = np.sort(v_vals)[::-1]

    k_cutoff, v_cutoff = k_vals.max() * 0.01, v_vals.max() * 0.01

    print(k_vals.max(), v_vals.max())
    print(k_cutoff, v_cutoff)

    # plt.hlines(y=k_cutoff, xmin=0, xmax=len(k_vals+1), color='coral', linestyles='dashed', label='1% cutoff (k)')
    # plt.hlines(y=v_cutoff, xmin=0, xmax=len(k_vals+1), color='royalblue', linestyles='dashed', label='1% cutoff (k)')

    plt.plot(k_vals/k_vals.max(), color='coral', label='k cache')
    plt.plot(v_vals/v_vals.max(), color='royalblue', label='v cache')
    plt.yscale("log")
    plt.xlabel('Rank index')
    plt.ylabel('Relative Singeular value')
    plt.title(f'SVD of KV Cache - Layer {i}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    plt.savefig(f'./figures/gemma3-4b/svd/svd_layer_{i}_pre_rope.png', dpi=800, bbox_inches='tight')
    plt.clf()


# energy distribution
for i in range(len(collected_k_outputs)):
    # num_key_value_heads = model.config.text_config.num_key_value_heads

    # num_key_value_heads = model.config.num_key_value_heads

    # k_cache_pre_rope = collected_k_outputs[i].float().reshape(-1, collected_k_outputs[i].shape[1] * num_key_value_heads, collected_k_outputs[i].shape[2] // num_key_value_heads)
    # v_cache_pre_rope = collected_v_outputs[i].float().reshape(-1, collected_k_outputs[i].shape[1] * num_key_value_heads, collected_k_outputs[i].shape[2] // num_key_value_heads)

    k_cache_pre_rope = collected_k_outputs[i].float()
    v_cache_pre_rope = collected_v_outputs[i].float()

    k_rank = svd_decomposition(k_cache_pre_rope)
    v_rank = svd_decomposition(v_cache_pre_rope)

    FIGURE_SIZE = (8, 5)   # (width, height) in inches             # Output resolution
    plt.rcParams['font.size'] = 14  # Set default font size
    plt.rcParams['font.family'] = 'serif' # Set font family, e.g., 'serif', 'sans-serif', 'monospace'

    # k_rank and v_rank are torch.Tensors of shape (batch, seq_len, rank)
    # For dot plot, flatten or select a slice (e.g., first batch and first sequence)
    k_vals = k_rank.cpu().numpy().flatten()
    v_vals = v_rank.cpu().numpy().flatten()
    k_vals = np.sort(k_vals)[::-1]
    v_vals = np.sort(v_vals)[::-1]

    k_energy = k_vals**2
    v_energy = v_vals**2

    k_total_energy = np.sum(k_energy)
    v_total_energy = np.sum(v_energy)

    # Calculate the cumulative energy and express it as a percentage
    cumulative_energy_ratio_k = np.cumsum(k_energy) / k_total_energy
    cumulative_energy_ratio_v = np.cumsum(v_energy) / v_total_energy

    cumulative_energy_percentage_k = cumulative_energy_ratio_k * 100
    cumulative_energy_percentage_v = cumulative_energy_ratio_v * 100

    # --- 2. Find Key Thresholds ---
    # Find the rank needed to capture 90%, 95%, and 99% of the energy
    k_rank_90 = np.searchsorted(cumulative_energy_ratio_k, 0.90) + 1
    k_rank_95 = np.searchsorted(cumulative_energy_ratio_k, 0.95) + 1
    k_rank_99 = np.searchsorted(cumulative_energy_ratio_k, 0.99) + 1
    v_rank_90 = np.searchsorted(cumulative_energy_ratio_v, 0.90) + 1
    v_rank_95 = np.searchsorted(cumulative_energy_ratio_v, 0.95) + 1
    v_rank_99 = np.searchsorted(cumulative_energy_ratio_v, 0.99) + 1
    print(f"Layer {i} - K Cache:")
    print(f"Rank to capture 90% of energy: {k_rank_90}")
    print(f"Rank to capture 95% of energy: {k_rank_95}")
    print(f"Rank to capture 99% of energy: {k_rank_99}")
    # print(f"Layer {i} - V Cache:")
    # print(f"Rank to capture 90% of energy: {v_rank_90}")
    # print(f"Rank to capture 95% of energy: {v_rank_95}")
    # print(f"Rank to capture 99% of energy: {v_rank_99}")

    # Plot the cumulative energy
    plt.plot(range(1, len(cumulative_energy_percentage_k) + 1), 
             cumulative_energy_percentage_k, color='coral', marker='o', linestyle='-', markersize=2, label='K Cache')
    # plt.plot(range(1, len(cumulative_energy_percentage_v) + 1), cumulative_energy_percentage_v, color='royalblue',      
    #          marker='o', linestyle='-', markersize=4, label='V Cache')
    # Add horizontal lines for the thresholds
    plt.axhline(y=90, color='coral', linestyle='--', label=f'90% Energy (Rank {k_rank_90})')
    plt.axhline(y=95, color='coral', linestyle='-.', label=f'95% Energy (Rank {k_rank_95})')    
    plt.axhline(y=99, color='coral', linestyle=':', label=f'99% Energy (Rank {k_rank_99})')    



    # plt.axhline(y=90, color='royalblue', linestyle='--', label=f'90% Energy (Rank {v_rank_90})')
    # plt.axhline(y=95, color='royalblue', linestyle='-.', label=f'95% Energy (Rank {v_rank_95})')

    plt.xlabel('Rank index')
    plt.ylabel('Energy Percentage (%)')
    plt.title(f'SVD of KV Cache - Layer {i}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    plt.savefig(f'./figures/gemma3-4b/energy/svd_layer_{i}_pre_rope.png', dpi=800, bbox_inches='tight')
    plt.clf()