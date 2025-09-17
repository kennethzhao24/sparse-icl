import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(f'./figures/{args.model}/{args.task}/svd/', exist_ok=True)
    os.makedirs(f'./figures/{args.model}/{args.task}/energy/', exist_ok=True)

    # Define the device for computation
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()  

    if args.task == 'pg19': 
        data = load_dataset("emozilla/pg19", split="test")  # Load a small subset for testing

    # Collect pre_rope outputs for all layers in these lists
    collected_k_outputs, collected_v_outputs = [], []

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
    
    if args.model.startswith('google/gemma'):
        num_layers = len(model.model.language_model.layers)
    else:
        num_layers = len(model.model.layers)

    n = 20000

    num_samples = 20

    sample_indxes = np.random.choice(len(data), num_samples, replace=False)

    svd_lists_k, svd_lists_v = [], []
    energy_lists_k, energy_lists_v = [], []

    for index in sample_indxes:
        print(f'Processing sample index: {index}')
        svd_layer_k, svd_layer_v = [], []
        energy_layer_k, energy_layer_v = [], []
        # Clear collected outputs before each forward pass
        collected_k_outputs.clear(), collected_v_outputs.clear()

        inputs = tokenizer(data[int(index)]['text'][:n], return_tensors="pt").to(device)
    
        hooks_k, hooks_v = [], []

        for layer_idx in range(num_layers):
            # Access the i-th layer
            if args.model.startswith('google/gemma'):
                layer = model.model.language_model.layers[layer_idx].self_attn
            else:
                layer = model.model.layers[layer_idx].self_attn
            # Register forward hooks
            hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
            hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
            hooks_k.append(hook_k)
            hooks_v.append(hook_v)

        with torch.no_grad():
            model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        for hook in hooks_k:
            hook.remove()
        for hook in hooks_v:
            hook.remove()

        for i in range(len(collected_k_outputs)):
            k_rank = torch.svd(collected_k_outputs[i].float())[1].cpu().numpy().flatten()
            v_rank = torch.svd(collected_v_outputs[i].float())[1].cpu().numpy().flatten()

            k_max, v_max = k_rank.max(), v_rank.max()

            k_vals, v_vals = np.sort(k_rank)[::-1], np.sort(v_rank)[::-1]

            k_energy, v_energy = k_vals**2, v_vals**2
            k_total_energy, v_total_energy = np.sum(k_energy), np.sum(v_energy)

            # Calculate the cumulative energy and express it as a percentage
            cumulative_energy_ratio_k = np.cumsum(k_energy) / k_total_energy
            cumulative_energy_ratio_v = np.cumsum(v_energy) / v_total_energy
            cumulative_energy_percentage_k = cumulative_energy_ratio_k * 100
            cumulative_energy_percentage_v = cumulative_energy_ratio_v * 100

            k_vals = k_vals / k_max
            v_vals = v_vals / v_max

            svd_layer_k.append(k_vals)
            svd_layer_v.append(v_vals)
            energy_layer_k.append(cumulative_energy_percentage_k)
            energy_layer_v.append(cumulative_energy_percentage_v)

        svd_lists_k.append(svd_layer_k)
        svd_lists_v.append(svd_layer_v)
        energy_lists_k.append(energy_layer_k)
        energy_lists_v.append(energy_layer_v)

        print(f'Finished sample index: {index}')


    # Plot error plot (mean ± std) for k_vals across samples for each layer
    num_layers = len(svd_lists_k[0])
    for i in range(num_layers):
        # Stack k_vals for this layer across all samples
        k_vals_layer = [svd_lists_k[sample_idx][i] for sample_idx in range(len(svd_lists_k))]
        v_vals_layer = [svd_lists_v[sample_idx][i] for sample_idx in range(len(svd_lists_v))]

        # Pad to the same length if necessary
        k_max_len = max(len(arr) for arr in k_vals_layer)
        v_max_len = max(len(arr) for arr in v_vals_layer)
        k_vals_padded = np.array([np.pad(arr, (0, k_max_len - len(arr)), 'constant', constant_values=np.nan) for arr in k_vals_layer])
        v_vals_padded = np.array([np.pad(arr, (0, v_max_len - len(arr)), 'constant', constant_values=np.nan) for arr in v_vals_layer])

        mean_k = np.nanmean(k_vals_padded, axis=0)
        mean_v = np.nanmean(v_vals_padded, axis=0)
        std_k = np.nanstd(k_vals_padded, axis=0)
        std_v = np.nanstd(v_vals_padded, axis=0)

        FIGURE_SIZE = (8, 5)   # (width, height) in inches             # Output resolution
        plt.rcParams['font.size'] = 14  # Set default font size
        plt.rcParams['font.family'] = 'serif' # Set font family, e.g., 'serif', 'sans-serif', 'monospace'
        plt.plot(mean_k, color='coral', label='K Cache Mean')
        plt.plot(mean_v, color='royalblue', label='V Cache Mean')
        plt.fill_between(np.arange(len(mean_k)), mean_k - std_k, mean_k + std_k, color='coral', alpha=0.3, label='Std Dev (K)')
        plt.fill_between(np.arange(len(mean_v)), mean_v - std_v, mean_v + std_v, color='royalblue', alpha=0.3, label='Std Dev (V)')

        plt.hlines(y=0.01, xmin=0, xmax=len(k_vals+1), color='black', linestyles='dashed')
        
        # plt.hlines(y=0.01, xmin=0, xmax=len(k_vals+1), color='black', linestyles='dashed', label=f'1% K (Rank {k_cutoff})')
        # plt.hlines(y=0.01, xmin=0, xmax=len(k_vals+1), color='black', linestyles='dashed',  label=f'1% V (Rank {v_cutoff})')

        plt.yscale("log")
        plt.xlabel('Rank Index')
        plt.ylabel('Relative Singular Value')
        plt.title(f'KV Cache SVD Error Plot - Layer {i}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'./figures/{args.model}/{args.task}/svd/svd_layer_{i}_kv_error.png', dpi=800, bbox_inches='tight')
        plt.clf()


    num_layers = len(svd_lists_k[0])
    for i in range(num_layers):
        # Stack energy for this layer across all samples
        k_energy_layer = [energy_lists_k[sample_idx][i] for sample_idx in range(len(energy_lists_k))]
        v_energy_layer = [energy_lists_v[sample_idx][i] for sample_idx in range(len(energy_lists_v))]

        # Pad to the same length if necessary
        k_max_len = max(len(arr) for arr in k_energy_layer)
        v_max_len = max(len(arr) for arr in v_energy_layer)
        k_energy_padded = np.array([np.pad(arr, (0, k_max_len - len(arr)), 'constant', constant_values=np.nan) for arr in k_energy_layer])
        v_energy_padded = np.array([np.pad(arr, (0, v_max_len - len(arr)), 'constant', constant_values=np.nan) for arr in v_energy_layer])

        mean_energy_k = np.nanmean(k_energy_padded, axis=0)
        mean_energy_v = np.nanmean(v_energy_padded, axis=0)
        std_energy_k = np.nanstd(k_energy_padded, axis=0)
        std_energy_v = np.nanstd(v_energy_padded, axis=0)

        k_rank_99 = np.searchsorted(mean_energy_k, 0.99) + 1
        v_rank_99 = np.searchsorted(mean_energy_v, 0.99) + 1

        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'serif'
        plt.plot(mean_energy_k, color='coral', label='K Cache Mean')
        plt.plot(mean_energy_v, color='royalblue', label='V Cache Mean')
        plt.fill_between(np.arange(len(mean_energy_k)), mean_energy_k - std_energy_k, mean_energy_k + std_energy_k, color='coral', alpha=0.3, label='Std Dev (K)')
        plt.fill_between(np.arange(len(mean_energy_v)), mean_energy_v - std_energy_v, mean_energy_v + std_energy_v, color='royalblue', alpha=0.3, label='Std Dev (V)')
        
        plt.axhline(y=99, color='black', linestyle=':')

        # plt.axhline(y=99, color='black', linestyle=':', label=f'99% K (Rank {k_rank_99})')
        # plt.axhline(y=99, color='black', linestyle=':', label=f'99% V (Rank {v_rank_99})') 

        plt.xlabel('Rank Index')
        plt.ylabel('Cumulative Energy Percentage (%)')
        plt.title(f'KV Cache Energy Plot - Layer {i}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'./figures/{args.model}/{args.task}/energy/energy_layer_{i}_kv.png', dpi=800, bbox_inches='tight')
        plt.clf()














    # index = 6 if args.task == 'pg19' else 0
    # inputs = tokenizer(data[index]['text'][:n], return_tensors="pt").to(device)

    # with torch.no_grad():
    #     model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # for hook in hooks_k:
    #     hook.remove()
    # for hook in hooks_v:
    #     hook.remove()

    # # rank distribution
    # for i in range(len(collected_k_outputs)):

    #     k_rank = torch.svd(collected_k_outputs[i].float())[1].cpu().numpy().flatten()
    #     v_rank = torch.svd(collected_v_outputs[i].float())[1].cpu().numpy().flatten()

    #     FIGURE_SIZE = (8, 5)   # (width, height) in inches             # Output resolution
    #     plt.rcParams['font.size'] = 14  # Set default font size
    #     plt.rcParams['font.family'] = 'serif' # Set font family, e.g., 'serif', 'sans-serif', 'monospace'

    #     # k_rank and v_rank are torch.Tensors of shape (batch, seq_len, rank)
    #     # For dot plot, flatten or select a slice (e.g., first batch and first sequence)
    #     k_vals = k_rank.cpu().numpy().flatten()
    #     v_vals = v_rank.cpu().numpy().flatten()
    #     k_vals = np.sort(k_vals)[::-1]
    #     v_vals = np.sort(v_vals)[::-1]

    #     k_cutoff, v_cutoff = k_vals.max() * 0.01, v_vals.max() * 0.01

    #     # print(k_vals.max(), v_vals.max())
    #     # print(k_cutoff, v_cutoff)

    #     plt.hlines(y=0.01, xmin=0, xmax=len(k_vals+1), color='black', linestyles='dashed')
    #     # plt.hlines(y=v_cutoff, xmin=0, xmax=len(k_vals+1), color='royalblue', linestyles='dashed', label='1% cutoff (k)')

    #     plt.plot(k_vals/k_vals.max(), color='coral', label='K Cache')
    #     plt.plot(v_vals/v_vals.max(), color='royalblue', label='V Cache')
    #     plt.yscale("log")
    #     plt.xlabel('Rank Index')
    #     plt.ylabel('Relative Singeular Value')
    #     plt.title(f'SVD of KV Cache - Layer {i}')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.savefig(f'./figures/{args.model}/{args.task}/svd/svd_layer_{i}_pre_rope.png', dpi=800, bbox_inches='tight')
    #     plt.clf()


    # # energy distribution
    # for i in range(len(collected_k_outputs)):
    #     _, k_rank, _ = torch.svd(collected_k_outputs[i].float())
    #     _, v_rank, _ = torch.svd(collected_v_outputs[i].float())

    #     FIGURE_SIZE = (8, 5)   # (width, height) in inches             # Output resolution
    #     plt.rcParams['font.size'] = 14  # Set default font size
    #     plt.rcParams['font.family'] = 'serif' # Set font family, e.g., 'serif', 'sans-serif', 'monospace'

    #     # k_rank and v_rank are torch.Tensors of shape (batch, seq_len, rank)
    #     # For dot plot, flatten or select a slice (e.g., first batch and first sequence)
    #     k_vals = k_rank.cpu().numpy().flatten()
    #     v_vals = v_rank.cpu().numpy().flatten()
    #     k_vals = np.sort(k_vals)[::-1]
    #     v_vals = np.sort(v_vals)[::-1]

    #     k_energy = k_vals**2
    #     v_energy = v_vals**2

    #     k_total_energy = np.sum(k_energy)
    #     v_total_energy = np.sum(v_energy)

    #     # Calculate the cumulative energy and express it as a percentage
    #     cumulative_energy_ratio_k = np.cumsum(k_energy) / k_total_energy
    #     cumulative_energy_ratio_v = np.cumsum(v_energy) / v_total_energy

    #     cumulative_energy_percentage_k = cumulative_energy_ratio_k * 100
    #     cumulative_energy_percentage_v = cumulative_energy_ratio_v * 100

    #     # --- 2. Find Key Thresholds ---
    #     # Find the rank needed to capture 90%, 95%, and 99% of the energy
    #     k_rank_99 = np.searchsorted(cumulative_energy_ratio_k, 0.99) + 1
    #     v_rank_99 = np.searchsorted(cumulative_energy_ratio_v, 0.99) + 1

    #     # Plot the cumulative energy
    #     plt.plot(range(1, len(cumulative_energy_percentage_k) + 1), 
    #             cumulative_energy_percentage_k, color='coral', 
    #             marker='o', linestyle='-.', markersize=2, label='K Cache')
    #     plt.plot(range(1, len(cumulative_energy_percentage_v) + 1), 
    #              cumulative_energy_percentage_v, color='royalblue',      
    #              marker='o', linestyle='-.', markersize=2, label='V Cache')
    #     # Add horizontal lines for the thresholds
    #     plt.axhline(y=99, color='black', linestyle=':', label=f'99% K (Rank {k_rank_99})')
    #     plt.axhline(y=99, color='black', linestyle=':', label=f'99% V (Rank {v_rank_99})') 


    #     plt.xlabel('Rank Index')
    #     plt.ylabel('Energy Percentage (%)')
    #     plt.title(f'Energy of KV Cache - Layer {i}')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     # plt.show()
    #     plt.savefig(f'./figures/{args.model}/{args.task}/energy/energy_layer_{i}_pre_rope.png', dpi=800, bbox_inches='tight')
    #     plt.clf()