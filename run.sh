CUDA_VISIBLE_DEVICES=0 python3 eval_vllm.py --model google/gemma-3-1b-it --task math500 --dtype bfloat16 --seed 42 --batch_size 32 --max_tokens 1024 --exp_name a100_math500_bf16_seq_1k
