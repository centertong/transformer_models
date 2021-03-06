import os
import logging
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset


# Will add code (based https://github.com/lucidrains/electra-pytorch)
class ElectraWrapper(torch.nn.Module):
    def __init__(self, model_desc, model_gener):
        self.desc = model_desc
        self.gener = model_gener


class ElectraDataset(Dataset):
    def __init__(self, tokenizer, max_len, path):
        logging.info('Start pretraining data load!')

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = []

        # 파일 리스트
        file_list = os.listdir(path) if os.path.isdir(path) else [path]

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
        for file_name in file_list:
            file = f'{path}/{file_name}'
            data_file = open(file, 'r', encoding='utf-8')
            for line in tqdm(data_file, desc='Data load for pretraining',leave=True):
                if len(line.split(' ')) < 50 or len(line.split(' ')) > 500: continue
                line = line[:-1]
                self.docs.append(line)
        logging.info('Complete data load')

    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=True, max_length=self.max_len, padding=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs

    def _pad_token(self, inputs):
        input_pads = self.max_len - inputs.shape[-1]
        return F.pad(inputs, pad=(0, input_pads), value=self.tokenizer.pad_token_id)

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        inputs = self._pad_token(inputs)

        inputs= inputs.squeeze()

        return inputs
