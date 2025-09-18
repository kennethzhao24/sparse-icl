CUDA_VISIBLE_DEVICES=0 


# python3 eval_vllm_many_shot_icl.py \
#    --model google/gemma-3-4b-it \
#    --task aime24 \
#    --num-shots 0 \
#    --dtype bfloat16 \
#    --seed 42 \
#    --batch_size 32 \
#    --max_tokens 1024 \
#    --max_seq_len 4096 \
#    --exp_name aime24_bf16_seq_1k_0_shot_test


python3 eval_vllm_many_shot_icl.py \
   --model google/gemma-3-4b-it \
   --task math500 \
   --num-shots 1 \
   --dtype bfloat16 \
   --seed 42 \
   --batch_size 32 \
   --max_tokens 1024 \
   --max_seq_len 8192 \
   --exp_name math500_bf16_seq_1k_1_shot_test_new
