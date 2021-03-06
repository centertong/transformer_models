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
    BertForMaskedLM,
    PerformerForMaskedLM,
    RfaForMaskedLM,
    LiteForMaskedLM,
    AftForMaskedLM,
    FastformerForMaskedLM,
    LunaForMaskedLM,
    ScatterBrainForMaskedLM,
    AbcForMaskedLM,
    ScalingForMaskedLM,
    RealformerForMaskedLM,
    MemEffForMaskedLM,
)
from transformers.data.processors.squad import SquadResult

from dataset.mlm import DatasetForMLM
from utils.trainer import Trainer, WarmupScheduler

import argparse


class MlmTrainer(Trainer):
    def __init__(self, model, tokenizer, optimizer, scheduler, device=None,
                 train_batch_size=12, test_batch_size=None,
                 checkpoint_path=None, model_name=None,
                 log_dir='./logs'):
        super().__init__(model, tokenizer, optimizer, scheduler, device,
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
    parser.add_argument("--train_file", type=str, required=True)

    return parser.parse_args()


def getModel(config_path):
    config = PretrainedConfig().from_json_file(config_path)

    if config.model_type == "mem_eff":
        return MemEffForMaskedLM(config)
    if config.model_type == "bert":
        return BertForMaskedLM(config)
    if config.model_type == "performer":
        return PerformerForMaskedLM(config)
    if config.model_type == "rfa":
        return RfaForMaskedLM(config)
    if config.model_type == "lite":
        return LiteForMaskedLM(config)
    if config.model_type == "aft":
        return AftForMaskedLM(config)
    if config.model_type == "fastformer":
        return FastformerForMaskedLM(config)
    if config.model_type == "luna":
        return LunaForMaskedLM(config)
    if config.model_type == "scatterbrain":
        return ScatterBrainForMaskedLM(config)
    if config.model_type == "abc":
        return AbcForMaskedLM(config)
    if config.model_type == "scaling":
        return ScalingForMaskedLM(config)
    if config.model_type == "realformer":
        return RealformerForMaskedLM(config)


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

    dataset = DatasetForMLM(tokenizer, args.max_len, path=args.train_file)

    model = getModel(args.config)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # optimizer = Adafactor(model.parameters(), lr= 1e-3, relative_step=False)
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-3)

    scheduler = WarmupScheduler(optimizer)

    trainer = MlmTrainer(model, tokenizer, optimizer, scheduler, model_name=args.name, device=args.device,
                         checkpoint_path=args.save_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size * 2)

    # tokenizer.save("test.json")

    trainer.build_dataloaders(dataset)

    trainer.train(epochs=args.epochs,
                  log_steps=1,
                  ckpt_steps=500,
                  gradient_accumulation_steps=1)


if __name__ == '__main__':
    main()
