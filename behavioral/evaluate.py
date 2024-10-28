# File: evaluate.py
# Description: core functions for implementing LM evaluation

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm

from model import LM
from utils import *


def evaluate(
    stimuli: pd.DataFrame, 
    prompts: pd.DataFrame,
    model: LM, 
    task: str = "crt1",
    use_system_prompt: bool = False
) -> pd.DataFrame:
    """Function that implements LM evaluation."""

    # The values in this dict correspond to column names in `stimuli`
    # that contain the relevant answer options.
    task_answer_types = {
        "crt1": ["correct", "intuitive", "total_cost", "more",
                "correct_money_fmt", "intuitive_money_fmt", 
                "total_cost_money_fmt", "more_money_fmt"],
        "crt2": ["correct", "intuitive"],
        "crt3": ["correct", "intuitive", "t"]
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
            # Initialize results for current item based on stimulus meta vars.
            res = {v: stim_row[v] for v in stimulus_meta_vars}

            # Make chat prompt according to model's chat template.
            prompt = make_chat_prompt(
                model, 
                stim_row.question, 
                task,
                prompt_row.trigger if prompt_row.trigger_type != "baseline" else "",
                use_system_prompt=use_system_prompt
            )

            # Record prompt and relevant meta variables.
            res["prompt"] = prompt
            res.update({v: prompt_row[v] for v in prompt_meta_vars})

            # Set up prefixes and continuations for logprob measurement.
            prefixes = [prompt] * len(answer_options)
            continuations = [str(stim_row[a]) for a in answer_options]

            # Compute logprob of each answer option conditioned on the prompt.
            all_scores = model.scorer.conditional_score(
                prefixes,
                continuations,
                separator=" ",
                reduction=get_reduction_fn("mean_and_sum")
            )
            mean_logprobs, sum_logprobs = zip(*all_scores)
            
            for reduction_type in ["mean", "sum"]:
                metric = f"{reduction_type}_logp"

                # Focus on mean logprobs or sum logprobs.
                if reduction_type == "mean":
                    scores = mean_logprobs
                else:
                    scores = sum_logprobs

                # Record scores in dictionary.
                answer_scores = {}
                for option_idx, option in enumerate(answer_options):
                    res[f"{metric}_{option}"] = scores[option_idx]
                    answer_scores[option] = scores[option_idx]

                # Compute the argmax score to obtain the model's "chosen" answer.
                if task == "crt1":
                    # For the CRT1 task, restrict analysis to the answer options
                    # that are properly formatted in US currency convention.
                    money_fmt_answer_options = answer_options[4:]
                    money_fmt_scores = scores[4:]
                    top_option = money_fmt_answer_options[
                        np.argmax(money_fmt_scores)
                    ]
                else:
                    top_option = answer_options[np.argmax(scores)]
                
                # Record the model's chosen answer, whether it is a distractor
                # answer, and whether it corresponds to the answer associated with
                # "slow" thinking (typically, the "correct" answer) or
                # "fast" thinking (typically, the "intuitive" answer).
                res[f"{metric}_response"] = top_option
                res[f"{metric}_response_isDistractor"] = not (
                    top_option.startswith("correct") or 
                    top_option.startswith("intuitive")
                )
                res[f"{metric}_response_isSlow"] = top_option.startswith("correct")
                res[f"{metric}_response_isFast"] = top_option.startswith("intuitive")

                # For CRT1, only look at answers in proper money formatting.
                if task == "crt1":
                    answer_scores = {
                        k: v for k, v in answer_scores.items() 
                        if k.endswith("_money_fmt")
                    }

                # Normalize scores to get a probability distribution over answers.
                answer_dist = softmax_answer_scores(answer_scores)

                # Record probability distribution.
                res[f"{metric}_dist"] = str(answer_dist)

            all_results.append(res)

    results_df = pd.DataFrame(all_results)
    return results_df
