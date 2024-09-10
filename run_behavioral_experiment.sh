#!/bin/bash

MODEL=$1
TASK=$2

# Change this variable to wherever you save Huggingface models
CACHE_DIR="/path/to/huggingface_cache"

echo "$MODEL / $TASK"

# NO system prompt
python src/run_behavioral_experiment.py \
    --model $MODEL \
    --task $TASK \
    --stimuli_dir data/stimuli \
    --prompt_file data/prompt_contrasts.csv \
    --output_dir model_output/no-system-prompt \
    --cache_dir $CACHE_DIR

# WITH system prompt - uncomment the below if desired
# python src/run_behavioral_experiment.py \
#     --model $MODEL \
#     --task $TASK \
#     --stimuli_dir data/stimuli \
#     --prompt_file data/prompt_contrasts.csv \
#     --output_dir model_output/system-prompt \
#     --cache_dir $CACHE_DIR \
#     --use_system_prompt
