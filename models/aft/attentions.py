import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from functools import partial



def AftBase(query, key, value, bias):
    query = query.sigmoid()
    key_exp = torch.exp(key - torch.amax(key,dim=-2, keepdim=True))
    bias_exp = torch.exp(bias - torch.amax(bias,dim=-2, keepdim=True))


    key_bias = torch.einsum('bld, lj->bjd', key_exp, bias_exp)
    sum_inv_key_layer = 1 / key_bias.sum(dim=-2) #B C
    scores = (key_bias * value).sum(dim=-2) # B, C
    out = query * (scores * sum_inv_key_layer).unsqueeze(1)
    return out


def AftSimple(query, key, value):
    query = query.sigmoid()
    key_exp = torch.exp(key - torch.amax(key,dim=-2, keepdim=True))
    
    sum_inv_key_layer = 1 / key_exp.sum(dim=-2)
    scores = (key_exp * value).sum(dim=-2)
    out = query * (scores * sum_inv_key_layer).unsqueeze(1)
    return out

def AftConv(query, key, value, weight):
    # Q, V : B, H, L, C
    # K : B L H
    # weight: H, 1, S, S

    _, h, *_ = query.size()

    query = query.sigmoid()
    key_exp = torch.exp(F.normalize(key, p=2, dim=-1))
    weight = torch.exp(weight) - 1

    # key_exp = torch.exp(key)
    # weight = torch.exp(weight) - 1

    sum_key_exp = key_exp.sum(dim=-2) # B H C
    key_value = key_exp * value
    sum_key_value = key_value.sum(dim=-2) # B H C
    
    conv_key_val = F.conv2d(key_value, weight, padding="same", groups=h)
    conv_key= F.conv2d(key, weight, padding="same", groups=h)
    
    numerator = conv_key_val + sum_key_value.unsqueeze(-2)
    denominator = conv_key + sum_key_exp.unsqueeze(-2)

    out = query * (numerator / denominator)
    
    return out