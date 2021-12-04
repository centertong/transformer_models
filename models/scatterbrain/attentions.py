import math
from os import lseek
import torch
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as F

import sys

from einops import repeat, rearrange
from functools import partial

def linear_attention(q, k, v):
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd->...ne', context, q)

    attn = torch.einsum('...nd,...ld->...nl', q, k)
    return out, attn.sum(-1)


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
        # query.masked_fill_(~mask, 0.)
        key.masked_fill_(~mask, 0.)
        value.masked_fill_(~mask, 0.)        

    out, attn_d = linear_attention(query, key, value)
    return out, query, key, attn_d

def AngularLSH(vecs, rotation_matrix):
    rotated_vecs = torch.einsum('bhlc, nrc-> bhlrn',  vecs, rotation_matrix)
    rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
    # batch, head, length, round, bucket
    buckets = torch.argmax(rotated_vecs, dim=-1).permute(0,1,3,2)
    # batch, head, round, length
    return buckets

MAX_MASK_VALUE = 100000

def ReformerAttentionIndex(qk, n_buckets=16, n_heads=4, chunk_size=16, mask=None):
    b, h, l, d = qk.size()
    rotation_matrix = torch.rand(n_buckets // 2, n_heads, qk.size(-1) ).to(qk.device)
    qk_buckets = AngularLSH(qk, rotation_matrix)
    # batch, head, round, length
    _, _, r, _ = qk_buckets.size()
    if mask is not None:
        mask = mask[:, None, None, :]
        mask = mask.bool()
        qk_buckets.masked_fill_(~mask, MAX_MASK_VALUE)

    qk_buckets = qk_buckets.argsort(-1)
    bucket_reverse = qk_buckets.argsort(-1)
    # batch, head, round, length

    qk_buckets = qk_buckets.reshape(b,h,r,-1, chunk_size)
    bucket_reverse = bucket_reverse.reshape(b,h,r,-1,chunk_size)

    def look_one_back_index(x):
        x_extra = torch.cat([x[:,:,:,-1:, ...], x[:,:,:, :-1, ...]], dim=-2)
        return torch.cat([x, x_extra], dim=-1)
    
    qk_buckets = look_one_back_index(qk_buckets)
    bucket_reverse = look_one_back_index(bucket_reverse)
    # batch, head, round, split, chunk * 2
    
    return qk_buckets.detach(), bucket_reverse.detach()

def getSparseVector(vector, index):
    # vector: (batch, head, length, dim)
    # index: (batch, head, round, split, chunk * 2)
    s_tuple = index.size()
    index = index.reshape(s_tuple[0], s_tuple[1], -1)
    
    vector = torch.gather(vector, dim=-2, index=index.unsqueeze(-1).expand(-1, -1, -1, vector.size(-1)))
    
    return vector.reshape(s_tuple + (vector.size(-1),))


def ReformerAttention(query, value, row_q, row_k, n_buckets = 16, n_heads = 4, mask=None):
    qk_index, reverse_index = ReformerAttentionIndex(query, n_buckets=n_buckets, n_heads=n_heads, mask=mask)
    
    if mask is not None:
        mask = mask[:, None, :, None]
        mask = mask.bool()
        # query.masked_fill_(~mask, 0)
        # value.masked_fill_(~mask, 0)

    sq = getSparseVector(query, qk_index)
    rq = getSparseVector(row_q, qk_index)
    rk = getSparseVector(row_k, qk_index)
    sv = getSparseVector(value, qk_index)
    s_mask = getSparseVector(mask.expand(-1, qk_index.size(1), -1, -1), qk_index)
    # batch, 1, round, split, chunk*2, 1
    
    b, h, r, s, c = qk_index.size()
    reverse_index = reverse_index.reshape(b, h, r, s, 2, -1)
    reverse_index = reverse_index.transpose(-2, -3)
    reverse_index = reverse_index.reshape(b, h, r * 2, -1)
    
    sparse_attn = torch.einsum('bhrscd, bhrsld -> bhrscl', sq, sq)
    row_attn = torch.einsum('bhrscd, bhrsld -> bhrscl', rq, rk)

    s_mask_t = s_mask.transpose(-1,-2)
    sparse_attn = torch.clip(sparse_attn, max=2)
    sparse_attn = sparse_attn + ~s_mask_t * -1e15
    
    attn = torch.exp(sparse_attn) - row_attn
    # attn: (batch, head, round, split, chunk*2, chunk*2)
    
    score = torch.einsum('bhrscl, bhrsld -> bhrscd', attn, sv)

    b, h, r, s, c, d = score.size()
    score = score.reshape(b, h, r, s, 2, -1, d).transpose(-3, -4)
    # score: (batch, head, round, 2, split, chunk, dim)
    score = score.reshape(b, h, r * 2, -1, d)
    # score: (batch, head, round * 2, length, dim)
    score = torch.gather(score, dim=-2, index=reverse_index.unsqueeze(-1).expand(-1, -1, -1, -1, d))

    score = score.mean(dim= -3)

    # score: (batch, head, dim)
    
    attn = attn.reshape(b,h,r,2, -1, c).transpose(-3, -4)
    attn = attn.reshape(b, h, r*2, -1, c)
    
    attn = torch.gather(attn, dim=-2, index=reverse_index.unsqueeze(-1).expand(-1, -1, -1, -1, c))
    
    attn = attn.sum(-1).mean(-2)
    # attn: (batch, head, length)
    return score, attn

def ScatterBrainAttention(query, key, value, nb_features, n_buckets = 16, n_heads=4, mask=None):
    b, h, l, d = query.size()
    projection_matrix = gaussian_random_matrix(nb_rows = nb_features, nb_columns = d,
                                                num_head=h, device=query.device)
    
    if mask is not None:
        mask_tmp = mask[:, None, :, None]
        mask_tmp = mask_tmp.bool()
        
    ro, row_q, row_k, rd = RowRankAttention(query, key, value, projection_matrix, mask_tmp)
    
    so, sd = ReformerAttention(query, value, row_q, row_k, n_buckets, n_heads, mask)
    
    diag_inv = 1 / (rd + sd + 1e-5)
    diag_inv = diag_inv.unsqueeze(-1)
    score = ro + so

    return score * diag_inv
