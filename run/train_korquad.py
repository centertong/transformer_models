import os
import sys
import json
import logging

from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers.optimization import Adafactor, AdamW
from transformers import PreTrainedTokenizerFast, PretrainedConfig
from tokenizers import Tokenizer
from models import (
    BertForQuestionAnswering,
)

from dataset.kosquad import KorQuADdataset
from utils import metrics, trainer

import argparse


class KorQuADtrainer(trainer.Trainer):
    def __init__(self, model, tokenizer, optimizer, device=None,
                 train_batch_size=12, test_batch_size=None,
                 checkpoint_path=None, model_name=None,
                 log_dir='./logs'):
        super().__init__(model, tokenizer, optimizer, device,
                         train_batch_size, test_batch_size, checkpoint_path, model_name)

        self.log_dir = log_dir
        logging.basicConfig(
            filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    def metric(self, preds=None, labels=None):
        pass

    def print_metric(self,):
        pass



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--train_data", type=str, default="data/korquad/KorQuAD_v1.0_train.json")
    parser.add_argument("--test_data", type=str, default="data/korquad/KorQuAD_v1.0_dev.json")
    return parser.parse_args()


def getModel(config_path, num_labels):
    config = PretrainedConfig().from_json_file(config_path)
    config.num_labels = num_labels

    if config.model_type == "bert":
        return BertForQuestionAnswering(config)


def main():
    args = parse_args()

    # Tokenizer

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(args.tokenizer))
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'unk_token': '[UNK]'})
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    train_dataset = KorQuADdataset(args.train_data, tokenizer, args.max_len)
    eval_dataset = KorQuADdataset(args.test_data, tokenizer, args.max_len)

    num_labels = 2
    model = getModel(args.config, num_labels)

    # optimizer = Adafactor(model.parameters(), lr= 1e-3, relative_step=False)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    trainer = KorQuADtrainer(model, tokenizer, optimizer, model_name=args.name, device=args.device,
                          checkpoint_path=args.save_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size * 2)

    # tokenizer.save("test.json")

    trainer.build_dataloaders(train_dataset, eval_dataset)

    trainer.train(epochs=args.epochs,
                  log_steps=1,
                  ckpt_steps=500,
                  gradient_accumulation_steps=1)

    # print(model.bert.encoder.layer[0].attention.self.repara_w)


if __name__ == '__main__':
    main()
