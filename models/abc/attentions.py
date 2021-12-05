import torch
import math
import torch.nn as nn

def AbcMlpAttention(query, key, value, dropout=None, head_mask=None):
    # q : batch, head, length, dim
    # k,v : batch, head, length2, dim
    *_, c = query.size()

    attention_scores = torch.einsum('bhic, bhjc->bhij', query, key)
    # torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(c)
    
    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    # context_layer = torch.matmul(attention_probs, value)
    context_layer = torch.einsum('bhij, bhjc->bhic', attention_probs, value)
    return context_layer