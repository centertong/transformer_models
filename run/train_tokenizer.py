import argparse
import tokenizers
from tokenizers import Tokenizer
import json
import os

TOKENIZER_MODEL_LIST = ["BPE", "WP", "UNI", "WORD"]
TOKENIZER_NORMALIZER_LIST = ["BERT", "LOWER", "NFC", "NFD", "ACCENT"]
TOKENIZER_PRE_TOKENIZER_LIST = ["BERT", "BYTE", "UNI", "WHITESPACE", "PUNCT"]
TOKENIZER_POST_PROCESS_LIST = ["BERT", "ROBERTA", "BYTE"]
TOKENIZER_DECODER_LIST = ["BPE", "BYTE", "WP", "CTC"]

help_doc = """
BERT: --normalizers BERT --pre_tokenizers BERT --model WP --post_process BERT --decoder WP --unk [UNK] --cls [CLS] --sep [SEP] --pad [PAD] --maks [Maks]
GPT2 : --pre_tokenizers BYTE --model BPE --post_process BYTE --unk [UNK]
RoBerta, BART:  --pre_tokenizers BYTE --model BPE --post_process ROBERTA --unk [UNK]
"""

def get_model(args):
    if args.model == "BPE": return tokenizers.models.BPE(unk_token=args.unk, dropout=args.dropout)
    if args.model == "WP": return tokenizers.models.WordPiece(unk_token=args.unk)
    if args.model == "UNI": return tokenizers.models.Unigram()
    if args.model == "WORD": return tokenizers.models.WordLevel(unk_token=args.unk)
    return tokenizers.models.WordPiece(unk_token="[UNK]")
    
    
def get_normalizer(item, args):
    if item == "BERT": return tokenizers.normalizers.BertNormalizer(lowercase=args.lowercase)
    if item == "LOWER": return tokenizers.normalizers.Lowercase()
    if item == "NFC": return tokenizers.normalizers.NFC()
    if item == "NFD": return tokenizers.normalizers.NFD()
    if item == "ACCENT": return tokenizers.normalizers.StripAccents()
    return tokenizers.normalizers.BertNormalizer()

def get_pre_tokenizer(item, args):
    if item == "BERT": return tokenizers.pre_tokenizers.BertPreTokenizer()
    if item == "BYTE": return tokenizers.pre_tokenizers.ByteLevel(args.add_prefix_space)
    if item == "UNI": return tokenizers.pre_tokenizers.UnicodeScripts()
    if item == "WHITESPACE": return tokenizers.pre_tokenizers.Whitespace()
    if item == "PUNCT": return tokenizers.pre_tokenizers.Punctuation()
    return tokenizers.pre_tokenizers.BertPreTokenizer()

def get_post_processor(args):
    if args.post_process == "BERT": return tokenizers.processors.BertProcessing(sep=(args.sep, args.sep_id), cls=(args.cls, args.cls_id))
    if args.post_process == "ROBERTA": return tokenizers.processors.RobertaProcessing(sep=(args.sep, args.sep_id), cls=(args.cls, args.cls_id))
    if args.post_process == "BYTE": return tokenizers.processors.ByteLevel()
    return None

def get_decoder(args):
    if args.decoder == "BPE": return tokenizers.decoders.BPE()
    if args.decoder == "WP": return tokenizers.decoders.WordPiece()
    if args.decoder == "BYTE": return tokenizers.decoders.ByteLevel()
    if args.decoder == "CTC": return tokenizers.decoders.CTC()
    return None

def get_trainer(args):
    vargs = vars(args)
    args_list = ["unk", "sep", "cls", "pad", "mask", "bos", "eos"]
    special_tokens = [vargs.get(arg) for arg in args_list if vargs.get(arg)]
    print(special_tokens)
    if args.model == "BPE": return tokenizers.trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)
    if args.model == "WP": return tokenizers.trainers.WordPieceTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)
    if args.model == "UNI": return tokenizers.trainers.UnigramTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens, unk_token=args.unk)
    if args.model == "WORD": return tokenizers.trainers.WordLevelTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)
    return tokenizers.trainers.WordPieceTrainer(vocab_size=args.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    


def tokenizer_setting(args):
    tokenizer = Tokenizer(get_model(args))

    tokenizer.normalizer = tokenizers.normalizers.Sequence([get_normalizer(item, args) for item in args.normalizers])
    tokenizer.pre_tokenizer  = tokenizers.pre_tokenizers.Sequence([get_pre_tokenizer(item, args) for item in args.pre_tokenizers])
    tokenizer.post_processor  = get_post_processor(args)
    tokenizer.decoder = get_decoder(args)

    return tokenizer

def trainer_setting(args):
    return get_trainer(args)


def convert_tokenizer_to_vocab(src, dst):
    f = open(dst,'w',encoding='utf-8')
    with open(src) as json_file:
        json_data = json.load(json_file)
        for item in json_data["model"]["vocab"].keys():
            f.write(item+'\n')

        f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=30000)
    
    
    parser.add_argument("--normalizers", type=str, default=None, choices=TOKENIZER_NORMALIZER_LIST, nargs='*')
    parser.add_argument("--pre_tokenizers", type=str, required=True, choices=TOKENIZER_PRE_TOKENIZER_LIST, nargs='*')
    parser.add_argument("--model", type=str, required=True, choices=TOKENIZER_MODEL_LIST)
    parser.add_argument("--post_process", type=str, default=None, choices=TOKENIZER_POST_PROCESS_LIST)
    parser.add_argument("--decoder", type=str, default=None, choices=TOKENIZER_DECODER_LIST)


    parser.add_argument("--unk", type=str, default=None)
    parser.add_argument("--sep", type=str, default=None)
    parser.add_argument("--cls", type=str, default=None)
    parser.add_argument("--sep_id", type=int, default=2)
    parser.add_argument("--cls_id", type=int, default=1)
    parser.add_argument("--pad", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--bos", type=str, default=None)
    parser.add_argument("--eos", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--lowercase", type=bool, default=True)
    parser.add_argument("--add_prefix_space", type=bool, default=True)
    
    
    parser.add_argument("--train_files", type=str, nargs='+')
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    
    tokenizer = tokenizer_setting(args)
    trainer = trainer_setting(args)

    tokenizer.train(args.train_files, trainer)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    convert_tokenizer_to_vocab(os.path.join(args.output_dir, "tokenizer.json"), os.path.join(args.output_dir, "vocab.txt"))




if __name__ == '__main__':
    main()