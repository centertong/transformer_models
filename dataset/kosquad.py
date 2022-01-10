import json
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch

class KorQuADdataset(Dataset):
    def __init__(self, path, tok, max_len=512):
        super(KorQuADdataset, self).__init__()

        self.max_len = max_len
        self.data = self.processing(path, tok)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return torch.tensor(data['input_ids'][0]), torch.tensor(data['attention_mask'][0]), torch.tensor([data['start_positions'][0], data['end_positions'][0]])


    def processing(self, file_path, tok):
        with open(file_path, 'r') as fh:
            data = json.load(fh)

        examples = []
        for article in tqdm(data['data']):
            for para in article['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    ques = qa['question']
                    examples.append(self.prepare_train_features(tok, {
                        'context': context,
                        'question': qa['question'],
                        'answers': qa['answers']
                    }))
        return examples


    def prepare_train_features(self, tok, examples):
        tokenized_examples = tok(
            examples["question"],
            examples["context"],
            truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
            max_length=self.max_len,
            stride=0,
            return_overflowing_tokens=True, # 길이를 넘어가는 토큰들을 반환할 것인지
            return_offsets_mapping=True,  # 각 토큰에 대해 (char_start, char_end) 정보를 반환한 것인지
            padding="max_length",
        )
        
        # example 하나가 여러 sequence에 대응하는 경우를 위해 매핑이 필요함.
        overflow_to_sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # offset_mappings으로 토큰이 원본 context 내 몇번째 글자부터 몇번째 글자까지 해당하는지 알 수 있음.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 정답지를 만들기 위한 리스트
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tok.cls_token_id)
            
            # 해당 example에 해당하는 sequence를 찾음.
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            # sequence가 속하는 example을 찾는다.
            example_index = overflow_to_sample_mapping[i]
            answers = examples["answers"][example_index]
            
            # 텍스트에서 answer의 시작점, 끝점
            answer_start_offset = answers["answer_start"]
            answer_end_offset = answer_start_offset + len(answers["text"])

            # 텍스트에서 현재 span의 시작 토큰 인덱스
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            # 텍스트에서 현재 span 끝 토큰 인덱스
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # answer가 현재 span을 벗어났는지 체크
            if not (offsets[token_start_index][0] <= answer_start_offset and offsets[token_end_index][1] >= answer_end_offset):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index와 token_end_index를 answer의 시작점과 끝점으로 옮김
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start_offset:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= answer_end_offset:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
