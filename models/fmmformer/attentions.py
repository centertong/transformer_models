import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def makeSlice(x, window_size, forward=1, backward=1):
    # q : batch, head, length, dim
    # k,v : batch, head, length2, dim
    b, h, l, d = x.size()
    pad = (0, 0, window_size * forward, window_size * backward)
    x = F.pad(x, pad, 'constant', value=0.)

    chunk_size = window_size * (forward + backward) + 1

    x_list = [x[:, :, ind:ind+chunk_size, :] for ind in range(l)]

    x = torch.stack(x_list, dim=-3)

    return x


def makeSliceMask(mask, window_size, forward=1, backward=1):
    b, l = mask.size()
    pad = (window_size * forward, window_size * backward)
    mask = F.pad(mask, pad, 'constant', value=0)

    chunk_size = window_size * (forward + backward) + 1
    mask_list = [mask[:, ind:ind+chunk_size] for ind in range(l)]

    mask = torch.stack(mask_list, dim=-2)

    return mask
