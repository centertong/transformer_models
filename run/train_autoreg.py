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
    BertForSequenceGeneration,
    PerformerForSequenceClassification,
    RfaForSequenceClassification,
    LiteForSequenceClassification,
    AftForSequenceClassification,
    FastformerForSequenceClassification,
    LunaForSequenceClassification,
    ScatterBrainForSequenceClassification,
    AbcForSequenceClassification,
    ScalingForSequenceClassification,
)

#from models.bert import BertConfig, BertForMaskedLM
from dataset.autoregressive import DatasetForAutoRegressive
from utils import metrics, trainer

import argparse

class AutoRegressiveTrainer(trainer.Trainer):
    def __init__(self, model, tokenizer, optimizer, device=None,
                    train_batch_size = 12, test_batch_size =None,
                    checkpoint_path =None, model_name=None,
                 log_dir='./logs'):
        super().__init__(model, tokenizer, optimizer, device, train_batch_size, test_batch_size, checkpoint_path, model_name)
        
        self.log_dir = log_dir
        logging.basicConfig(filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()


def getModel(config_path):
    config = PretrainedConfig().from_json_file(config_path)
    
    if config.model_type == "bert":
        return BertForSequenceGeneration(config)
    if config.model_type == "performer":
        return PerformerForSequenceClassification(config)
    if config.model_type == "rfa":
        return RfaForSequenceClassification(config)
    if config.model_type == "lite":
        return LiteForSequenceClassification(config)
    if config.model_type == "aft":
        return AftForSequenceClassification(config)
    if config.model_type == "fastformer":
        return FastformerForSequenceClassification(config)
    if config.model_type == "luna":
        return LunaForSequenceClassification(config)
    if config.model_type == "scatterbrain":
        return ScatterBrainForSequenceClassification(config)
    if config.model_type == "abc":
        return AbcForSequenceClassification(config)
    if config.model_type == "scaling":
        return ScalingForSequenceClassification(config)
    

def main():
    args = parse_args()

    # Tokenizer

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(args.tokenizer))
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    

    dataset = DatasetForAutoRegressive(tokenizer, 512, args.dataset)
    model = getModel(args.config)

    
    # optimizer = Adafactor(model.parameters(), lr= 1e-3, relative_step=False)
    optimizer = AdamW(model.parameters(), lr= 1e-4, weight_decay= 1e-4)
        
    trainer = AutoRegressiveTrainer(model, tokenizer, optimizer,model_name=args.name, device=args.device, checkpoint_path=args.save_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size * 2)

    #tokenizer.save("test.json")
    
    trainer.build_dataloaders(dataset, None)

    trainer.train(epochs=args.epochs,
                  log_steps=1,
                  ckpt_steps=500,
                  gradient_accumulation_steps=1)
    
    # print(model.bert.encoder.layer[0].attention.self.repara_w)


if __name__ == '__main__':
    main()
