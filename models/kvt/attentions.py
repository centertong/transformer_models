import torch
import torch.nn.functional as F


def MultiHeadKnnAttention(query, key, value, knn_size=128, mask=None, dropout=None, head_mask=None):
    b, h, l, d = key.size()

    query = query * (d ** -0.5)

    attn_weights = torch.einsum('...ld, ...md -> ...lm', query, key)

    if mask is not None:
        attn_weights = attn_weights + mask

    mask = torch.zeros((b, h, l, l), device=query.device, requires_grad=False)
    _, index = torch.topk(attn_weights, k=knn_size, dim=-1, largest=True)
    mask.scatter_(-1, index, 1.)
    attn_weights = torch.where(
        mask > 0, attn_weights, torch.full_like(attn_weights, -1e15))

    attn_score = F.softmax(attn_weights, dim=-1)

    if dropout is not None:
        attn_score = dropout(attn_score)

    if head_mask is not None:
        attn_score = attn_score * head_mask

    out = torch.einsum('...lm, ...md -> ...ld', attn_score, value)

    return out
