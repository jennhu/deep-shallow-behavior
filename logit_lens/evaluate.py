# File: evaluate.py
# Description: helper functions for performing LM evaluation

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm
import torch

from utils import *


def conditional_score_all_layers(
    nn_model, 
    prefix: str, 
    continuation: str, 
    sep: str = " ", 
    reduction: str = "sum", 
    check_tokenization: bool = True
):
    """
    ADAPTED FROM THE FOLLOWING:
    https://github.com/Butanium/llm-latent-language/blob/7519b4a3b528dacbc4c37aca0f74497bda18e57f/exp_tools.py#L51

    Get scores of the continuation conditioned on the prefix at each layer.
    Scores are defined as a reduction of the raw token-level log probabilities.

    Args:
        nn_model: NNSight LanguageModel object
        prefix: str (what you want to condition on)
        continuation: str (what you want to compute probability of)
    """
    nn_model.eval()
    text = prefix + sep + continuation
    inputs = nn_model.tokenizer(text, return_tensors="pt")

    n_layers = len(nn_model.model.layers)

    with nn_model.trace(text) as tracer:
        # Useful if we want to do multiple sentences at a time
        # inds = torch.arange(len(inputs.input_ids))
        hiddens_l = [
            layer.output[0][0, :].unsqueeze(1)
            for layer in nn_model.model.layers
        ]
        hiddens = torch.cat(hiddens_l, dim=1)
        rms_out = nn_model.model.norm(hiddens)
        
        # SHAPE: n_tokens x n_layers x vocab_size
        logits = nn_model.lm_head(rms_out).cpu().save()
        logprobs = logits.log_softmax(dim=-1).cpu().save()

    # Tokenize the continuation separately so we know which ones to keep.
    continuation_tokens = nn_model.tokenizer(continuation)["input_ids"]

    # Remove BOS token if necessary, since it might have been
    # automatically prepended when tokenizing the continuation separately.
    if nn_model.tokenizer.bos_token:
        continuation_tokens = [
            t for t in continuation_tokens 
            if t != nn_model.tokenizer.bos_token_id
        ]

    # OPTIONAL: Double check that we are "reconstructing" tokens correctly.
    if check_tokenization:
        prefix_tokens = nn_model.tokenizer(prefix)["input_ids"]
        new_text_tokens = prefix_tokens + continuation_tokens
        text_tokens = inputs["input_ids"][0].tolist()
        assert all(n == t for n, t in zip(new_text_tokens, text_tokens))

    # Only keep the logits corresponding to the continuation.
    # We look at the logits at position i-1 for target token i, so we need to
    # shift logprobs to the left by 1.
    n_continuation_tokens = len(continuation_tokens)
    continuation_logprobs = logprobs[-n_continuation_tokens-1:-1, :, :]

    # Get logprobs corresponding to each token in the continuation,
    # for each layer. FINAL SHAPE: n_layers x n_continuation_tokens
    all_layer_logprobs = np.array([
        [
            continuation_logprobs[i][layer_id][token_id].detach().cpu()
            for i, token_id in enumerate(continuation_tokens)
        ]
        for layer_id in range(n_layers)
    ])

    # Perform reduction on token-level logprobs to get a final score.
    if reduction == "sum":
        scores = np.sum(all_layer_logprobs, axis=1)
    elif reduction == "mean":
        scores = np.mean(all_layer_logprobs, axis=1)
    else:
        raise ValueError("`reduction` should be 'sum' or 'mean'")
    return scores

def evaluate(
    stimuli: pd.DataFrame, 
    prompts: pd.DataFrame,
    model, # : LM, 
    task: str = "crt1",
    metric: str = "sum_logp",
    use_system_prompt: bool = False
) -> pd.DataFrame:
    """Function that implements LM evaluation."""
    # The values in this dict correspond to column names in `stimuli`
    # that contain the relevant answer options. We only care about
    # the "correct" and "intuitive" answers.
    task_answer_types = {
        "crt1": ["correct_money_fmt", "intuitive_money_fmt"],
        "crt2": ["correct", "intuitive"],
        "crt3": ["correct", "intuitive"]
    }
    answer_options = task_answer_types[task]

    # Specify meta variables related to STIMULI and PROMPTS that we want to
    # record in the final results.
    stimulus_meta_vars = list(stimuli) # get all column names
    prompt_meta_vars = list(prompts) # get all column names

    all_results = []

    # Iterate over all stimuli and prompt variants.
    for _, stim_row in tqdm(stimuli.iterrows(), total=len(stimuli)):
        for _, prompt_row in prompts.iterrows():
            # Make chat prompt according to model's chat template.
            prompt = make_chat_prompt(
                model, 
                stim_row.question, 
                task,
                prompt_row.trigger if prompt_row.trigger_type != "baseline" else "",
                use_system_prompt=use_system_prompt
            )

            # Get scores for each answer option, conditioned on the same prompt.
            scores = []
            for a in answer_options:
                answer = str(stim_row[a])
                layer_scores = conditional_score_all_layers(
                    model, prompt, answer, sep=" "
                )
                scores.append(layer_scores)
            # SHAPE: n_answer_options x n_layers
            scores = np.array(scores) 
            n_layers = scores.shape[1]

            # Prepare results.
            for layer_idx in range(n_layers):
                # Add metadata about stimulus, prompt trigger, and layer.
                res = {v: stim_row[v] for v in stimulus_meta_vars}
                res["prompt"] = prompt
                res.update({v: prompt_row[v] for v in prompt_meta_vars})
                res["layer_idx"] = layer_idx

                # Record scores in dictionary.
                answer_scores = {}
                for option_idx, a in enumerate(answer_options):
                    res[f"{metric}_{a}"] = scores[option_idx][layer_idx]
                    answer_scores[a] = scores[option_idx][layer_idx]

                # Compute the argmax score to obtain the model's "chosen" answer.
                top_option = answer_options[np.argmax(
                    s[layer_idx] for s in scores
                )]
                
                # Record the model's chosen answer, and whether it corresponds 
                # to the answer associated with
                # "deep"/"slow" thinking (typically, the "correct" answer) or
                # "shallow"/"fast" thinking (typically, the "intuitive" answer).
                res[f"{metric}_response"] = top_option
                res[f"{metric}_response_isSlow"] = top_option.startswith("correct")

                # Normalize scores to get a probability distribution over answers.
                answer_dist = softmax_answer_scores(answer_scores)

                # Record probability distribution.
                res[f"{metric}_dist"] = str(answer_dist)

                all_results.append(res)

    results_df = pd.DataFrame(all_results)
    return results_df
