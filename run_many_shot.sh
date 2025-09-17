CUDA_VISIBLE_DEVICES=0 


python3 eval_vllm_many_shot_icl.py --model google/gemma-3-4b-it --task math500 --num-shots 0 --dtype bfloat16 --seed 42 --batch_size 4 --max_tokens 1024 --exp_name math500_bf16_seq_1k_0_shot
python3 eval_vllm_many_shot_icl.py --model google/gemma-3-4b-it --task math500 --num-shots 1 --dtype bfloat16 --seed 42 --batch_size 4 --max_tokens 1024 --exp_name math500_bf16_seq_1k_1_shot
python3 eval_vllm_many_shot_icl.py --model google/gemma-3-4b-it --task math500 --num-shots 2 --dtype bfloat16 --seed 42 --batch_size 4 --max_tokens 1024 --exp_name math500_bf16_seq_1k_2_shot

