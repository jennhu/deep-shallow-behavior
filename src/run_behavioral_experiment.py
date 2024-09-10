# File: run_behavioral_experiment.py
# Description: main wrapper script that should be called to evaluate LMs

import argparse
import os
import pandas as pd

from model import LM
import evaluate


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for behavioral experiment."""
    parser = argparse.ArgumentParser()

    # File-related parameters
    parser.add_argument("--stimuli_dir", type=str, default="data/stimuli", 
                        help="Path to folder containing stimuli")
    parser.add_argument("-o", "--output_dir", type=str, default="model_output", 
                        help="Path to directory where output files will be written")
    parser.add_argument("--cache_dir", type=str, 
                        help="Path to Huggingface cache")
    parser.add_argument("--prompt_file", default="data/prompt_contrasts.csv", type=str, 
                        help="Path to CSV file containing prompt contrasts")
    parser.add_argument("--hf_token_path", default="src/hf_token.txt", type=str, 
                        help="Path to text file containing Huggingface token")
    
    # Model-related parameters
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Name of Huggingface model identifier")
    parser.add_argument("--tokenizer", default=None, type=str,
                        help="Name of Huggingface tokenizer. Defaults to the "
                        "same name as model.")

    # Experiment-related parameters
    parser.add_argument("--task", type=str, default=None, nargs="+",
                        choices=["crt1", "crt2", "crt3"])
    parser.add_argument("--use_system_prompt", default=False, action="store_true")
    
    args = parser.parse_args()
    return args

def main() -> None:
    """
    Main high-level function for running a specified experiment on a 
    specified Huggingface language model.
    """
    args = parse_args()
    print(args)

    # Initialize model.
    with open(args.hf_token_path, "r") as fp:
        token = fp.read()
    model = LM(
        args.model, 
        tokenizer_name=args.tokenizer,
        token=token,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )

    # Read prompts.
    prompts = pd.read_csv(args.prompt_file)

    # Evaluate model on each task.
    if args.task is None:
        tasks = ["crt1", "crt2", "crt3"]
    else:
        tasks = args.task
    for task in tasks:
        print(f"***** Task = {task.upper()} *****")

        # Create output directory.
        os.makedirs(args.output_dir, exist_ok=True)

        # Get name of output file where results will be written.
        file = f"{task}_{model.safe_model_name}.csv"
        outfile = os.path.join(args.output_dir, file)

        # Read stimuli.
        stimuli = pd.read_csv(os.path.join(args.stimuli_dir, f"{task}.csv"))
            
        # Run the evaluation.
        result = evaluate.evaluate(
            stimuli,
            prompts,
            model, 
            task=task,
            use_system_prompt=args.use_system_prompt
        )
        result.to_csv(outfile, index=False)
        print(f"Wrote results to {outfile}")

        # Log results to stdout.
        print("="*80)
        print("Answer distribution (mean logprob):")
        print(result.mean_logp_response.value_counts(normalize=True))
        print("="*80)


if __name__ == "__main__":
    main()