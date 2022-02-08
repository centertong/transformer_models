import os
import sys
import json
import logging

from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast, PretrainedConfig
from tokenizers import Tokenizer
from models import (
    CanineForMaskedLM,
)
from transformers.data.processors.squad import SquadResult

from dataset.mlm import DatasetForCanineMLM
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

    if config.model_type == "canine":
        return CanineForMaskedLM(config)


def main():
    args = parse_args()

    dataset = DatasetForCanineMLM(args.max_len, path=args.train_file)

    model = getModel(args.config)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # optimizer = Adafactor(model.parameters(), lr= 1e-3, relative_step=False)
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-3)

    scheduler = WarmupScheduler(optimizer)

    trainer = MlmTrainer(model, None, optimizer, scheduler, model_name=args.name, device=args.device,
                         checkpoint_path=args.save_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size * 2)

    # tokenizer.save("test.json")

    trainer.build_dataloaders(dataset)

    trainer.train(epochs=args.epochs,
                  log_steps=1,
                  ckpt_steps=500,
                  gradient_accumulation_steps=1)


if __name__ == '__main__':
    main()
