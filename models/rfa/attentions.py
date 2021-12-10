import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from einops import repeat, rearrange
from functools import partial

def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

def casual_linear_attention(q, k, v):
    kv = torch.einsum('bhlm, bhlc -> bhlmc', k, v)
    tri_mask = torch.ones((q.size(-2), q.size(-2)))
    tri_mask = torch.triu(tri_mask)

    kv_presum = torch.einsum('bhlmc, lj -> bhjmc', kv, tri_mask)
    k_presum = torch.einsum('bhlm, lj -> bhjm', k, tri_mask)

    D_inv = 1. / torch.einsum('...nd, ...nd -> ...n', q, k_presum.type_as(q))

    context = torch.einsum('bhlm, bhlmc -> bhlc', q, kv_presum)
    out = torch.einsum('bhlc, bhl -> bhlc', context, D_inv)

    return out

def gaussian_random_matrix(nb_rows, nb_columns, num_head, device = "cpu"):
    block_list = [torch.randn((1, 1, nb_columns), device=device) for _ in range(num_head * nb_rows)]
    matrix = torch.cat(block_list).view(num_head, nb_rows, nb_columns)
    return matrix


def random_feature_map(data, projection_matrix, mode="arccos"):
    b, *_ = data.shape
    projection = repeat(projection_matrix, 'h j d -> b h j d', b = b)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', F.normalize(data, p=2, dim=-1), projection)
    if mode == "arccos":
        data_dash = F.relu(data_dash) * (data.shape[-1] ** -0.5) + 1e-5
    else:
        data_dash = torch.cat([torch.sin(data_dash), torch.cos(data_dash)], dim=-1) * (data.shape[-1] ** -0.5)
    return data_dash


def RfaAttention(query, key, value, projection_matrix=None, mode="arccos", mask=None, is_casual=False):
    query = random_feature_map(query, projection_matrix=projection_matrix, mode=mode)
    key = random_feature_map(key, projection_matrix=projection_matrix, mode=mode)
    if mask is not None:
        key.masked_fill_(~mask, 0.)
    
    if is_casual:
        out = casual_linear_attention(query, key, value)
    else:
        out = linear_attention(query, key, value)
    return out

def RfaGateAttention(query, key, value, gate, projection_matrix=None, mode="arccos", mask=None, is_casual=False):
    query = random_feature_map(query, projection_matrix=projection_matrix, mode=mode)
    key = random_feature_map(key, projection_matrix=projection_matrix, mode=mode)
    if mask is not None:
        key.masked_fill_(~mask, 0.)
    
    if is_casual:
        out = casual_linear_attention(query, key, value)
    else:
        out = linear_attention(query, key, value)
    return out
