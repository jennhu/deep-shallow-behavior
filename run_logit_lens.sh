#!/bin/bash

MODEL=$1 # "meta-llama/Llama-2-7b-chat-hf" 
TASK=$2

# Change this variable to wherever you save Huggingface models
CACHE_DIR="/path/to/huggingface_cache"

echo "$MODEL / $TASK"

# NO system prompt
python logit_lens/run_logit_lens.py \
    --model $MODEL \
    --task $TASK \
    --stimuli_dir data/stimuli \
    --prompt_file data/prompt_contrasts.csv \
    --output_dir model_output/logit_lens \
    --cache_dir $CACHE_DIR

# WITH system prompt - uncomment the below if desired
# python logit_lens/run_logit_lens.py \
#     --model $MODEL \
#     --task $TASK \
#     --stimuli_dir data/stimuli \
#     --prompt_file data/prompt_contrasts.csv \
#     --output_dir model_output/logit_lens \
#     --cache_dir $CACHE_DIR \
#     --use_system_prompt
