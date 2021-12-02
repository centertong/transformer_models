import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from functools import partial

def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2) # + 1e-5
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    # context = torch.einsum('...nd,...ne->...de', k, v)
    # out = torch.einsum('...de,...nd->...ne', context, q)
    return out


def gaussian_random_matrix(nb_rows, nb_columns, num_head, device = "cpu"):
    block_list = [torch.randn((1, 1, nb_columns), device=device) for _ in range(num_head * nb_rows)]
    matrix = torch.cat(block_list).view(num_head, nb_rows, nb_columns)
    return matrix


def random_feature_map(data, projection_matrix):
    b, *_ = data.shape
    projection = repeat(projection_matrix, 'h j d -> b h j d', b = b)
    projection = projection.type_as(data)
    m = projection.size(-2)

    data_dash = torch.einsum('...id,...jd->...ij', data, projection)
    data_norm = (data ** 2).sum(dim=-1, keepdim=True) / 2
    data_dash = data_dash - data_norm

    data_dash = torch.exp(data_dash - torch.amax(data_dash, dim=-2, keepdim=True)) * (m ** -0.5)
    return data_dash + 1e-5


def RowRankAttention(query, key, value, projection_matrix=None, mask=None):
    query = random_feature_map(query, projection_matrix=projection_matrix)
    key = random_feature_map(key, projection_matrix=projection_matrix)
    
    if mask is not None:
        key.masked_fill_(~mask, 0.)
        value.masked_fill_(~mask, 0.)        

    out = linear_attention(query, key, value)
    return out, query, key

def getHash(vecs, rotation_matrix):
    rotated_vecs = torch.einsum('bhlc, nc-> bhln',  vecs, rotation_matrix)
    rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
    # batch, head, length, bucket
    buckets = torch.argmax(rotated_vecs, dim=-1)
    # batch, head, length

    return buckets

def getSparseMatrix(query, key):
    # batch, head, length
    query = repeat(query, 'b h l -> b h l j', j = key.size(-1))
    key = repeat(key, 'b h l -> b h j l', j= query.size(-1))

    sm = query == key
    return sm.detach()

def getSparseAttention(query, key, matrix):
    # query : B H L C
    sparse_query = torch.einsum('bhqd, qk -> bhqkd', query, matrix)
    sparse_key = torch.einsum('bhkd, qk -> bhkqd', key, matrix)
    return torch.einsum('bhqkd, bhkqd-> bhqk', sparse_query, sparse_key)

def SparseAttention(query, key, value, row_q, row_k, n_buckets = 32):
    rotation_matrix = torch.rand(n_buckets // 2, query.size(-1) ).to(query.device)
    query_buckets = getHash(query, rotation_matrix)
    key_buckets = getHash(key, rotation_matrix)
    
    sparse_matrix = getSparseMatrix(query_buckets, key_buckets)

    attention = getSparseAttention(query, key, sparse_matrix)
    row_attention = getSparseAttention(row_q, row_k, sparse_matrix)
    sparse_attention = attention - row_attention

    out = torch.einsum('bhjl, bhlc->bhjc', sparse_attention, value)
    return out
