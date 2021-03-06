import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from functools import partial


def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


def causal_linear_attention(q, k, v):
    g = torch.einsum('bhlm, bhlc -> bhlmc', k, v)
    trimask = torch.ones((q.size(-2), q.size(-2)))
    trimask = torch.triu(trimask)
    g_presum = torch.einsum('bhlmc, lj -> bhjmc', g, trimask)
    k_presum = torch.einsum('bhlm, lj -> bhjm', k, trimask)

    D_inv = 1. / torch.einsum('...nd, ...nd -> ...n', q, k_presum.type_as(q))

    context = torch.einsum('bhlmc, bhlm -> bhlc', g_presum, q)
    out = torch.einsum('bhlc, bhl -> bhlc', context, D_inv)

    return out


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij',
                             (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                      torch.amax(data_dash, dim=-1, keepdim=True)) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True)) + eps)

    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij',
                             (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns

    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn(
            (nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * \
            torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def FAVORPlusAttention(query, key, value, projection_matrix, attention_mask=None, mode="softmax", is_causal=False):
    device = query.device
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, :, None]
        attention_mask = attention_mask.bool()
        key.masked_fill_(~attention_mask, 0.)

    if mode == "relu":
        query = generalized_kernel(
            query, projection_matrix=projection_matrix, device=device)
        key = generalized_kernel(
            key, projection_matrix=projection_matrix, device=device)

    elif mode == "softmax":
        query = softmax_kernel(
            query, projection_matrix=projection_matrix, device=device, is_query=True)
        key = softmax_kernel(
            key, projection_matrix=projection_matrix, device=device, is_query=False)

    if is_causal:
        out = causal_linear_attention(query, key, value)
    else:
        out = linear_attention(query, key, value)
    return out
