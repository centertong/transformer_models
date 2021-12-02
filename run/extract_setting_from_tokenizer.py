import argparse
import json
import os
import sys

def trainer_setting(args):
    if args.trainer in ["BpeTrainer", "WordPieceTrainer"]:
        return {
            "min_frequency": args.min_frequency,
            "limit_alphabet": args.limit_alphabet,
            "initial_alphabet": args.initial_alphabet,
            "continuing_subword_prefix": args.continuing_subword_prefix,
            "end_of_word_suffix": args.end_of_word_suffix,
        }

    if args.trainer == "UnigramTrainer":
        return {
            "shrinking_factor": args.shrinking_factor,
            "unk_token": args.unk_token,
            "max_piece_length": args.max_piece_length,
            "n_sub_iterations": args.n_sub_iterations,
        }
    
    if args.trainer == "WordLevelTrainer":
        return {
            "min_frequency": args.min_frequency,
        }


def extract_setting(tokenizer_data, args):

    normalizer = tokenizer_data["normalizer"]
    pre_tokenizer = tokenizer_data["pre_tokenizer"]
    model = {k: v for k, v in tokenizer_data["model"].items() if k != "vocab"}
    decoder = tokenizer_data["decoder"]
    post_processor = tokenizer_data["post_processor"]
    if post_processor.get("sep"): post_processor["sep"] = tuple(post_processor.get("sep"))
    if post_processor.get("cls"): post_processor["cls"] = tuple(post_processor.get("cls"))
    
    trainer = trainer_setting(args)
    trainer["type"] = args.trainer
    trainer["vocab_size"] = len(tokenizer_data["model"]["vocab"])
    trainer["special_tokens"] = [token_data["context"] for token_data in tokenizer_data["added_tokens"] if token_data["special"]]


    return {
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "model": model,
        "decoder": decoder,
        "trainer": trainer,
    }


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
    return json_data

def write_json(data, file):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent="\t")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_file", type=str, required=True)    
    parser.add_argument("--output_file", type=str, required=True)
    
    parser.add_argument("--trainer", type=str, required=True, choices=["BpeTrainer", "UnigramTrainer", "WordLevelTrainer", "WordPieceTrainer"])
    
    parser.add_argument("--min_frequency", type=int, default=0)
    parser.add_argument("--show_progress", type=bool, default=False)
    parser.add_argument("--min_frequency", type=int, required=False)
    parser.add_argument("--limit_alphabet ", type=int, default=None)
    parser.add_argument("--initial_alphabet", type=str, nargs='*', default=[])
    parser.add_argument("--continuing_subword_prefix", type=str, default=None)
    parser.add_argument("--end_of_word_suffix", type=str, default=None)
    parser.add_argument("--shrinking_factor", type=float, default=0.75)
    parser.add_argument("--unk_token", type=str, default=None)
    parser.add_argument("--max_piece_length", type=int, default=16)
    parser.add_argument("--n_sub_iterations", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(args.tokenizer)

    



if __name__ == '__main__':
    main()