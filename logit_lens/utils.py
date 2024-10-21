# File: utils.py
# Description: helper functions used during LM evaluation

import numpy as np


def flatten(xss: list) -> list:
    """Helper function for flattening a list."""
    return [x for xs in xss for x in xs]

def get_reduction_fn(reduction: str):
    """Returns an anonymous function that applies a reduction to a Tensor."""
    if reduction == "mean":
        reduction_fn = lambda x: x.mean(0).item()
    elif reduction == "sum":
        reduction_fn = lambda x: x.sum(0).item()
    elif reduction == "mean_and_sum":
        reduction_fn = lambda x: (x.mean(0).item(), x.sum(0).item())
    else:
        raise ValueError("`reduction` should be 'mean', 'sum', or 'mean_and_sum")
    return reduction_fn

def softmax_answer_scores(answer_scores: dict[str, float]) -> dict[str, float]:
    """Applies softmax function to all values in `answer_scores` dictionary."""
    exp_scores = {
        answer: np.exp(score) for answer, score in answer_scores.items()
    }
    total = sum(exp_scores.values())
    norm_scores = {
        answer: score / total for answer, score in exp_scores.items()
    }
    return norm_scores

def get_file_safe_model_name(model: str) -> str:
    """
    Returns a file-safe version of a Huggingface model identifier by
    only keeping the model name after a forward slash (/).
    Example: meta-llama/Llama-2-7b-hf --> Llama-2-7b-hf
    """
    safe_model_name = model.split("/")[1] if "/" in model else model
    return safe_model_name

def make_chat_prompt(
    model,
    query: str, 
    task: str,
    prompt: str,
    use_system_prompt: bool = False
) -> str:
    """Converts question into instruction-chat formatted prompt string."""

    # Cognitive reflection tasks (Hagendorff et al. 2023)
    if task.startswith("crt"):
        # user_msg = "Please answer the following question. " + query.strip()
        user_msg = "Your task is to answer the following question. " + query.strip()
    else:
        raise ValueError(f"Unknown task: {task}")

    # Construct chat object (list of dictionaries).
    user_chat = {"role": "user", "content": user_msg}
    
    # BASELINE CONDITION: empty prompt
    if prompt == "":
        # If there's no specific trigger, just use the basic user prompt.
        chat = [user_chat]
    # OTHER CONDITIONS: meaningful prompt
    else:
        # Put the prompt in the system prompt.
        if use_system_prompt:
            system_chat = {"role": "system", "content": prompt}
            chat = [system_chat, user_chat]
        # Otherwise, prepend the prompt to the user prompt.
        else:
            user_chat["content"] = prompt + " " + user_msg
            chat = [user_chat]

    full_chat_prompt = model.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    return full_chat_prompt
