import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from functools import partial

def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

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

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)



def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def FAVORPlusAttention(query, key, value, projection_matrix, mode="softmax"):
    device = query.device

    if mode == "relu":
        query = generalized_kernel(query, projection_matrix=projection_matrix, device=device)
        key = generalized_kernel(key, projection_matrix=projection_matrix, device=device)

    elif mode == "softmax":
        query = softmax_kernel(query, projection_matrix=projection_matrix, device=device, is_query= True)
        key = softmax_kernel(key, projection_matrix=projection_matrix, device=device, is_query= False)
        
    out = linear_attention(query, key, value)
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
        data_dash = F.relu(data_dash) * (data.shape[-1] ** -0.5)
    else:
        data_dash = torch.cat([torch.sin(data_dash), torch.cos(data_dash)], dim=-1) * (data.shape[-1] ** -0.5)
    return data_dash



def RfaAttention(query, key, value, projection_matrix=None, mode="arccos"):
    query = random_feature_map(query, projection_matrix=projection_matrix, mode=mode)
    key = random_feature_map(key, projection_matrix=projection_matrix, mode=mode)
    out = linear_attention(query, key, value)
    return out


def AftBase(query, key, value, bias):
    query = query.sigmoid()
    key_exp = torch.exp(key)
    bias_exp = torch.exp(bias)

    key_bias = torch.einsum('bld, ll->bld', key_exp, bias_exp)
    sum_inv_key_layer = 1 / key_bias.sum(dim=-2) #B C
    scores = (key_bias * value).sum(dim=-2) # B, C
    out = query * (scores * sum_inv_key_layer).unsqueeze(1)
    return out


def AftSimple(query, key, value):
    query = query.sigmoid()
    key_exp = torch.exp(key)
    
    sum_inv_key_layer = 1 / key_exp.sum(dim=-2)
    scores = (key_exp * value).sum(dim=-2)
    out = query * (scores * sum_inv_key_layer).unsqueeze(1)
    return out

def AftConv(query, key, value, weight):
    # B, L, H, C
    query = query.sigmoid()
    key_exp = torch.exp(key)
    weight = torch.exp(weight) - 1

    sum_key_exp = key_exp.sum(dim=1) # B H C
    key_value = key_exp * value
    sum_key_value = key_value.sum(dim=1) # B H C

    conv_key= F.conv1d(key, weight)
    conv_key_val = F.conv1d(key_value, weight)
    
    numerator = conv_key_val + sum_key_value
    denominator = conv_key + sum_key_exp

    out = query * (numerator / denominator).unsqueeze(2)    
    
    return out