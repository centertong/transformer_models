import argparse
import tokenizers
from tokenizers import Tokenizer
import json
import os
import sys

import importlib

def get_instance(config, pkg_name):
    if config is None: return None
    pkg = importlib.import_module('.'.join(['tokenizers', pkg_name]))
    args = {k:v for k, v in config.items() if k != "type"} if pkg_name != 'processors' else make_args_in_post_process(config)
    cls = getattr(pkg, config["type"])(**args)
    return cls

def make_args_in_post_process(config):
    if config["type"] != "TemplateProcessing": return {k:v for k, v in config.items() if k != "type"}

    def template_parsing(item):
        prefix = "$" if item.get("Sequence") else ""
        token = item["Sequence"] if item.get("Sequence") else item["SpecialToken"]
        
        return prefix + ':'.join([token["id"], str(token["type_id"])])

    args = {}
    if isinstance(config.get("single"), str):
        args["single"] = config.get("single")
    elif isinstance(config.get("single"), list):
        if len(config.get("single")) == 0: sys.exit("Invalid Syntax Single Parameter for TemplateProcessing")
        single = config.get("single")
        if isinstance(single, str): args["single"] = single
        else: args["single"] = [template_parsing(item) for item in single]
    else:
        sys.exit("Invalid Syntax Single Parameter for TemplateProcessing")

    
    if isinstance(config.get("pair"), str):
        args["pair"] = config.get("pair")
    elif isinstance(config.get("pair"), list):
        if len(config.get("pair")) == 0: sys.exit("Invalid Syntax Pair Parameter for TemplateProcessing")
        pair = config.get("pair")
        if isinstance(pair, str): args["pair"] = pair
        else: args["pair"] = [template_parsing(item) for item in pair]
    else:
        sys.exit("Invalid Syntax Pair Parameter for TemplateProcessing")

    if isinstance(config.get("special_tokens"), dict):
        args["special_tokens"] = [v for _, v in config["special_tokens"].items()]
    elif isinstance(config.get("special_tokens"), list):
        args["special_tokens"] = config.get("special_tokens")
    
    return args

def get_sequence(config, pkg_name):
    if config is None: return None
    if isinstance(config, dict): return get_instance(config, pkg_name)
    if isinstance(config, list):
        pkg = importlib.import_module('.'.join(['tokenizers', pkg_name]))
        arg_list = [get_instance(item, pkg_name) for item in config]
        cls = getattr(pkg, 'Sequence')(arg_list)
        return cls    
    return None
    


def convert_tokenizer_to_vocab(src, dst):
    f = open(dst,'w',encoding='utf-8')
    with open(src) as json_file:
        json_data = json.load(json_file)
        for item in json_data["model"]["vocab"].keys():
            f.write(item+'\n')

        f.close()

def load_config(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
    return json_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)    
    parser.add_argument("--train_files", type=str, nargs='+')
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_file)

    tokenizer = Tokenizer(get_instance(config.get("model"), "models"))

    tokenizer.normalizer = get_sequence(config.get("normalizer"), "normalizers")
    tokenizer.pre_tokenizer  = get_sequence(config.get("pre_tokenizer"), "pre_tokenizers")
    tokenizer.post_processor  = get_instance(config.get("post_processor"), "processors")
    tokenizer.decoder = get_instance(config.get("decoder"), "decoders")

    trainer = get_instance(config.get("trainer"), "trainers")

    tokenizer.train(args.train_files, trainer)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    convert_tokenizer_to_vocab(os.path.join(args.output_dir, "tokenizer.json"), os.path.join(args.output_dir, "vocab.txt"))




if __name__ == '__main__':
    main()