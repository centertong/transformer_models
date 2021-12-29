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
    BertForSequenceClassification,
    PerformerForSequenceClassification,
    RfaForSequenceClassification,
    LiteForSequenceClassification,
    AftForSequenceClassification,
    FastformerForSequenceClassification,
    LunaForSequenceClassification,
    Luna2ForSequenceClassification,
    ScatterBrainForSequenceClassification,
    AbcForSequenceClassification,
    ScalingForSequenceClassification,
    KvtForSequenceClassification,
    MemEffForSequenceClassification,
    FfnForSequenceClassification
)

#from models.bert import BertConfig, BertForMaskedLM
from dataset.klue import getKlueDataset
from utils import metrics, trainer

import argparse

class KlueTrainer(trainer.Trainer):
    def __init__(self,type, model, tokenizer, optimizer, device=None,
                    train_batch_size = 12, test_batch_size =None,
                    checkpoint_path =None, model_name=None,
                 log_dir='./logs'):
        super().__init__(model, tokenizer, optimizer, device, train_batch_size, test_batch_size, checkpoint_path, model_name)
        
        self.type = type
        self.log_dir = log_dir
        logging.basicConfig(filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)

    def metric(self, preds=None, labels=None):
        if preds is None:
            if self.type == 'tc':
                self.score = metrics.MacroF1(self.eval_loader.dataset.l2c)
            if self.type == 'sts':
                self.score = metrics.Pearson()
            if self.type == 'nli':
                self.score = metrics.Accuracy()
        
        else:
            self.score.add(preds.cpu(), labels.cpu())
            

    def print_metric(self,):
        print(self.score.calc())
    
    
model_list = ["bert", "performer", "rfa"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--type", type=str, choices=['sts', 'tc', 'nli'])
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--max_len", type=int, default=512)
    return parser.parse_args()

def getPath(type):
    base_path = "data/klue_benchmark/"
    if type == 'sts': 
        type_path = "klue-sts-v1.1"
        return os.path.join(base_path, type_path, "klue-sts-v1.1_train.json"), os.path.join(base_path, type_path, "klue-sts-v1.1_dev.json")
    if type == 'tc': 
        type_path = "ynat-v1.1"
        return os.path.join(base_path, type_path, "ynat-v1.1_train.json"), os.path.join(base_path, type_path, "ynat-v1.1_dev.json")
    if type == 'nli': 
        type_path = "klue-nli-v1.1"
        return os.path.join(base_path, type_path, "klue-nli-v1.1_train.json"), os.path.join(base_path, type_path, "klue-nli-v1.1_dev.json")


def getModel(config_path, num_labels):
    config = PretrainedConfig().from_json_file(config_path)
    config.num_labels = num_labels
    
    if config.model_type == "bert":
        return BertForSequenceClassification(config)
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
    if config.model_type == "luna2":
        return Luna2ForSequenceClassification(config)
    if config.model_type == "scatterbrain":
        return ScatterBrainForSequenceClassification(config)
    if config.model_type == "abc":
        return AbcForSequenceClassification(config)
    if config.model_type == "scaling":
        return ScalingForSequenceClassification(config)
    if config.model_type == "mem_eff":
        return MemEffForSequenceClassification(config)
    if config.model_type == "kvt":
        return KvtForSequenceClassification(config)
    if config.model_type == "ffn":
        return FfnForSequenceClassification(config)
    

def main():
    args = parse_args()

    # Tokenizer

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(args.tokenizer))
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'unk_token': '[UNK]'})
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    train_path, dev_path = getPath(args.type)
    train_dataset = getKlueDataset(args.type, train_path, tokenizer, args.max_len)
    eval_dataset = getKlueDataset(args.type, dev_path, tokenizer, args.max_len)
        
    if args.type in ['sts']:
        num_labels = 1
    else:
        num_labels = len(train_dataset.c2l)

    model = getModel(args.config, num_labels)

    
    # optimizer = Adafactor(model.parameters(), lr= 1e-3, relative_step=False)
    optimizer = AdamW(model.parameters(), lr= 1e-4, weight_decay= 1e-4)
        
    trainer = KlueTrainer(args.type, model, tokenizer, optimizer,model_name=args.name, device=args.device, checkpoint_path=args.save_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size * 2)

    #tokenizer.save("test.json")
    
    trainer.build_dataloaders(train_dataset, eval_dataset)

    trainer.train(epochs=args.epochs,
                  log_steps=1,
                  ckpt_steps=500,
                  gradient_accumulation_steps=1)
    
    # print(model.bert.encoder.layer[0].attention.self.repara_w)


if __name__ == '__main__':
    main()
