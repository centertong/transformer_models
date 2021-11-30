import json
from torch.utils.data import Dataset
import numpy as np

def getKlueDataset(type, path, tok):
    if type == "nli": return NLIdataset(path, tok)
    if type == "tc": return TCdataset(path, tok)
    if type == "sts": return STSdataset(path, tok)

class NLIdataset(Dataset):
    def __init__(self, path, tok):
        super(NLIdataset, self).__init__()
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self.tok = tok
        self.data, self.l2c, self.c2l = self.parse(data)
        self.max_len = 512
        
    def parse(self, data):
        returns = [{"premise": item['premise'], "hypothesis": item['hypothesis'], 'label': item['gold_label']} for item in data]
        labels = list(set([ item['label'] for item in returns ]))
        id2label = {k:v for k, v in enumerate(labels)}
        label2id = {v:k for k, v in enumerate(labels)}
        return returns, id2label, label2id

    def __len__(self):
        return len(self.data)
    
    def _tokenize_input_ids(self, first, second, pad_to_max_length: bool = True):
        inputs = self.tok.encode(first, second, add_special_tokens=True, max_length=self.max_len, padding='max_length', return_tensors='pt',truncation=True)
        return inputs

    def __getitem__(self, idx):
        item = self.data[idx]

        inputs = self._tokenize_input_ids(item['premise'], item['hypothesis'])
        inputs= inputs.squeeze()
        inputs_mask = (inputs != self.tok.pad_token_id).long()
        labels= self.c2l[item['label']]

        return inputs, inputs_mask, labels


class TCdataset(Dataset):
    def __init__(self, path, tok):
        super(TCdataset, self).__init__()
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self.tok = tok
        self.data, self.c2l, self.l2c = self.parse(data)
        self.max_len = 512
        
    def parse(self, data):
        returns = [{"input": item['title'], "category": item['label']} for item in data]
        categories = list(set([item['label'] for item in data]))
        cate2label = {cat: idx for idx, cat in enumerate(categories) }
        label2cate = {v: k for k, v in cate2label.items()}
        for item in returns:
            item["label"] = cate2label[item["category"]]

        return returns, cate2label, label2cate

    def __len__(self):
        return len(self.data)
    
    def _tokenize_input_ids(self, first, pad_to_max_length: bool = True):
        inputs = self.tok.encode(first, add_special_tokens=True, max_length=self.max_len, padding='max_length', return_tensors='pt',truncation=True)
        return inputs


    def __getitem__(self, idx):
        item = self.data[idx]

        inputs = self._tokenize_input_ids(item['input'])
        inputs= inputs.squeeze()
        inputs_mask = (inputs != self.tok.pad_token_id).long()
        labels= self.c2l[item['category']]

        return inputs, inputs_mask, labels



class STSdataset(Dataset):
    def __init__(self, path, tok):
        super(STSdataset, self).__init__()
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self.tok = tok
        self.data = self.parse(data)
        self.max_len = 512
        
    def parse(self, data):
        returns = [{"sent1": item['sentence1'], "sent2": item['sentence2'], 'logit': item['labels']['label'], 'label': item['labels']['binary-label'] } for item in data]
        
        return returns

    def __len__(self):
        return len(self.data)
    
    def _tokenize_input_ids(self, first, second, pad_to_max_length: bool = True):
        inputs = self.tok.encode(first, second, add_special_tokens=True, max_length=self.max_len, padding='max_length', return_tensors='pt',truncation=True)
        return inputs

    def __getitem__(self, idx):
        item = self.data[idx]

        inputs = self._tokenize_input_ids(item['sent1'], item['sent2'])
        inputs= inputs.squeeze()
        inputs_mask = (inputs != self.tok.pad_token_id).long()
        labels= np.array(item['logit'], dtype=np.float32)

        return inputs, inputs_mask, labels

