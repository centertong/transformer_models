import os
import logging
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset


class DatasetForAutoRegressive(Dataset):
    def __init__(self, tokenizer, max_len, path, bos='[BOS]', eos='[EOS]', pad='[PAD]'):
        logging.info('Start pretraining data load!')

        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.bos_id = tokenizer.encode(bos)
        self.eos_id = tokenizer.encode(eos)
        self.pad_id = tokenizer.encode(pad)

        # 파일 리스트
        file_list = os.listdir(path) if os.path.isdir(path) else [path]

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        for file_name in file_list:
            file = f'{path}/{file_name}' if os.path.isdir(path) else file_name
            data_file =  open(file, 'r',encoding='utf-8')
            for line in tqdm(data_file, desc='Data load for pretraining'):
                if len(line.split(' ')) < 10 or len(line.split(' ')) > 500: continue
                line = line[:-1]
                self.docs.append(line)
        logging.info('Complete data load')

    def _tokenize_input_ids(self, input_ids: list, add_special_tokens:bool = False, pad_to_max_length: bool = True):
        inputs = self.tokenizer.encode(input_ids)
        return inputs

    def _add_special_tokens(self, inputs):
        if len(inputs) + 1 > self.max_len:
            input_ids = self.bos_id + inputs[:self.max_len - 1] 
            label_ids = inputs[:self.max_len]
        else:
            
            input_ids = self.bos_id + inputs
            label_ids = inputs + self.eos_id
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + self.pad_id * pad_len
            label_ids = label_ids + self.pad_id * pad_len
        return input_ids, label_ids
    
    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True, add_special_tokens=True)
        inputs, labels = self._add_special_tokens(inputs)

        inputs, labels = torch.tensor(inputs), torch.tensor(labels)

        inputs= inputs.squeeze()
        labels= labels.squeeze()
        inputs_mask = inputs != 0


        return inputs, inputs_mask, labels
