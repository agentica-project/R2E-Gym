#!/usr/bin/env python3
import argparse
import json
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Reduce a JSONL of trajectories to (instance_id, model_patch, model_name_or_path)"
    )
    parser.add_argument(
        "input",
        help="path to input .jsonl"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="path to write reduced .jsonl (default: all_preds/<input_basename>.jsonl)"
    )
    args = parser.parse_args()

    # derive model_name_or_path from the input filename (without extension)
    model_name = os.path.splitext(os.path.basename(args.input))[0]

    # determine output path
    if args.output is None:
        out_dir = "all_preds"
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"{model_name}.jsonl")
    else:
        # ensure output directory exists
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.input,  'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:

        for line in tqdm(fin, desc="Processing lines", unit="lines"):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            reduced = {
                "instance_id":        rec.get("ds", {}).get("instance_id"),
                "model_patch":        rec.get("output_patch"),
                "r2e_score":          rec.get("reward"),
                "model_name_or_path": model_name
            }
            fout.write(json.dumps(reduced) + "\n")

    print(f"Wrote reduced JSONL to {args.output}")

if __name__ == "__main__":
    main()