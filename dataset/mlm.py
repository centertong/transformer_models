import os
import logging
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset


class DatasetForMLM(Dataset):
    def __init__(self, tokenizer, max_len, path):
        logging.info('start wiki data load')

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = []
        # 파일 리스트

        file_list = os.listdir(path) if os.path.isdir(path) else [path]

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        file_progress_bar = tqdm(
            file_list, position=0, bar_format='{l_bar}{bar:10}{r_bar}')
        for file_name in file_list:
            file = f'{path}/{file_name}' if os.path.isdir(path) else file_name
            data_file = open(file, 'r', encoding='utf-8')
            for line in tqdm(data_file, desc='Data load for pretraining'):
                #line_len = len(self.tokenizer.encode(line))
                # if line_len < 50 or line_len > 512: continue
                if len(line.split(' ')) < 10 or len(line.split(' ')) > 100:
                    continue
                line = line[:-1]
                self.docs.append(line)

        logging.info('complete data load')

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # mlm_probability defaults to 0.15 in Bert
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(
            special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # self.tokenizer.mask_token_id#  # We only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        if pad:
            input_pads = self.max_len - inputs.shape[-1]
            label_pads = self.max_len - labels.shape[-1]

            inputs = F.pad(inputs, pad=(0, input_pads),
                           value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, pad=(0, label_pads),
                           value=self.tokenizer.pad_token_id)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        #inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=False, max_length=self.max_len, padding=pad_to_max_length, return_tensors='pt',truncation=True))
        inputs = self.tokenizer.encode(input_ids, add_special_tokens=True, max_length=self.max_len,
                                       padding=pad_to_max_length, return_tensors='pt', truncation=True)
        return inputs

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(
            self.docs[idx], pad_to_max_length=True)
        inputs, labels = self.mask_tokens(inputs, pad=True)

        inputs = inputs.squeeze()
        inputs_mask = (inputs != self.tokenizer.pad_token_id).long()
        labels = labels.squeeze()

        return inputs, inputs_mask, labels


class DatasetForCanineMLM(Dataset):
    def __init__(self, max_len, path, 
            cls_token_id=1, sep_token_id=2,
            pad_token_id=0, mask_token_id=3, boundary = 32):
        logging.info('start wiki data load')

        self.max_len = max_len
        self.docs = []
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.boundary = boundary # space unicode
        self.special_tokens = [self.cls_token_id, self.sep_token_id, self.pad_token_id, self.boundary]
        # 파일 리스트

        file_list = os.listdir(path) if os.path.isdir(path) else [path]

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        file_progress_bar = tqdm(
            file_list, position=0, bar_format='{l_bar}{bar:10}{r_bar}')
        for file_name in file_list:
            file = f'{path}/{file_name}' if os.path.isdir(path) else file_name
            data_file = open(file, 'r', encoding='utf-8')
            for line in tqdm(data_file, desc='Data load for pretraining'):
                #line_len = len(self.tokenizer.encode(line))
                # if line_len < 50 or line_len > 512: continue
                if len(line) < 20 or len(line) > self.max_len - 2:
                    continue
                line = line.strip()
                self.docs.append(line)

        logging.info('complete data load')

    def mask_tokens(self, inputs: torch.Tensor, boundaries: torch.Tensor, mlm_probability=0.15, pad=True):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # mlm_probability defaults to 0.15 in Bert
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = boundaries == -1

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        mask_index = torch.unique(boundaries[masked_indices])
        masked_indices = torch.full(labels.shape, False)
        for ind in mask_index:
            masked_indices[boundaries == ind] = True
        # self.tokenizer.mask_token_id#  # We only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(mask_index.shape, 0.8)).bool()

        for ind in mask_index[indices_replaced]:
            inputs[boundaries == ind] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        mask_index = mask_index[~indices_replaced]        
        indices_random = torch.bernoulli(torch.full(mask_index.shape, 0.5)).bool()
        for ind in mask_index[indices_random]:
            rand_size = (boundaries == ind).sum().item()
            random_words = torch.randint(
                65532, (rand_size,)) + 4
            inputs[boundaries == ind] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _assign_boundary_id(self, input_ids):
        boundary_id = 0
        boundary_id_list = []
        for c in input_ids:
            if c in self.special_tokens:
                boundary_id_list.append(-1)
                boundary_id += 1
            else:
                boundary_id_list.append(boundary_id)

        return torch.tensor(boundary_id_list)


    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        #inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=False, max_length=self.max_len, padding=pad_to_max_length, return_tensors='pt',truncation=True))
        inputs = [1] + [ord(c) for c in input_ids if ord(c) < 65536] + [2]
        if len(inputs) < self.max_len:
            pad_size = self.max_len - len(inputs)
            inputs = inputs + [0] * pad_size
        boundaries = self._assign_boundary_id(inputs)
        inputs = torch.tensor(inputs)

        return inputs, boundaries

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)

    def __getitem__(self, idx):
        inputs, boundaries = self._tokenize_input_ids(
            self.docs[idx], pad_to_max_length=True)
        inputs, labels = self.mask_tokens(inputs, boundaries, pad=True)

        inputs = inputs.squeeze()
        inputs_mask = (inputs != self.pad_token_id).long()
        labels = labels.squeeze()

        return inputs, inputs_mask, labels
