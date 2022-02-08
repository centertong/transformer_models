import math
import torch
import torch.nn as nn
from functools import reduce


class CanineEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hash = config.n_hash
        self.hidden_size = config.hidden_size
        self.n_bucket = config.n_bucket
        self.embs = nn.ModuleList([nn.Embedding(self.n_bucket, config.hidden_size // self.n_hash, padding_idx=config.pad_token_id) for _ in range(self.n_hash)])
        
        self.hashes = torch.randint(self.n_bucket, (self.n_hash,))

        self.position_embeddings = AbsolutePositionalEncoding(config.max_position_embeddings, config.hidden_size)    
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, position_ids=None, **kawargs):
        
        input_embs = []
        for hash, emb in zip(self.hashes, self.embs):
            index = (input_ids * hash).long() % self.n_bucket
            input_embs.append(emb(index))

        input_embs = torch.cat(input_embs, dim=-1)

        seq_length = input_embs.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.position_embeddings(position_ids)
        input_embs = input_embs + position_embeddings

        return input_embs


        

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "absolute":
            self.position_embeddings = AbsolutePositionalEncoding(config.max_position_embeddings, config.hidden_size)    
        if self.position_embedding_type == "fixed":
            self.position_embeddings = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
        if self.position_embedding_type == "axial":
            ahidden = config.hidden_size // 2
            bhidden = config.hidden_size - ahidden
            asize = config.max_position_embeddings / 64
            self.position_embeddings = AxialPositionalEncoding((asize, 64) ,(ahidden, bhidden))
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )


    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type in ["absolute", "fixed"]:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


"""
Positional Encoding Codes
1. AbsolutePositionalEncoding (Normal)
2. FixedPositionalEncoding (BERT)
3. AxialPositionalEncoding (Reformer)
"""


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe(position_ids)    


class FixedPositionalEncoding(nn.Module):

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: Tensor, shape [seq_len, position_id]
        """
        return self.pe[position_ids]


class AxialPositionalEncoding(nn.Module):
    def __init__(self, axial_shape, dims):
        super().__init__()
        assert len(axial_shape) == 2
        assert len(dims) == 2
        
        self.axial_shape = axial_shape
        self.max_len = reduce(lambda x, y: x * y, axial_shape, 1)
        self.dim = reduce(lambda x, y: x+y, dims, 0)
        
        self.ws = nn.ParameterList([nn.Parameter(torch.zeros((1, n, d)).normal_(0,1)) for n, d in zip(axial_shape, dims)])
        
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        embs = []
        for idx, w in enumerate(self.ws):
            x = w.expand((self.axial_shape[1-idx], -1, -1))
            if idx == 1:
                x = x.transpose(1,0)
            x.reshape(self.axial_shape[idx], -1)
            embs.append(x)
        
        emb = torch.cat(embs, dim=-1)
        return emb[position_ids]

        
