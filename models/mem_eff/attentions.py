import torch
import torch.utils.checkpoint as checkpoint


def MultiHeadChunkAttention(query, key, value, chunk_size=128, mask=None, dropout=None, head_mask=None):
    b, h, l, d = key.size()
    assert l % chunk_size == 0

    key = key.reshape(b, h, l // chunk_size, chunk_size, d)
    value = value.reshape(b, h, l // chunk_size, chunk_size, d)
    query = query * (d ** -0.5)

    def summarize_chunk(query, key, value, mask, dropout, head_mask):
        attn_weights = torch.einsum('...id, ...jcd-> ...ijc', query, key)

        if mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            mb, mh, mq, ml = mask.size()
            mask = mask.view(mb, mh, mq,  ml // chunk_size, chunk_size)
            attn_weights = attn_weights + mask

        max_score = torch.amax(attn_weights, dim=-1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)

        if dropout is not None:
            exp_weights = dropout(exp_weights)

        if head_mask is not None:
            exp_weights = exp_weights * head_mask

        exp_values = torch.einsum(
            '...ncd, ...lnc-> ...lnd', value, exp_weights)
        # exp_weights: batch, head, query_length, key_chunk_num, key_chunk_size
        # exp_values: batch, head, query_length, key_chunk_num, dim

        return exp_values, exp_weights.sum(dim=-1, keepdim=True), max_score.squeeze(-1)

    if query.requires_grad:
        chunk_values, chunk_weights, chunk_max = checkpoint.checkpoint(
            summarize_chunk, query, key, value, mask, dropout, head_mask
        )
    else:
        chunk_values, chunk_weights, chunk_max = summarize_chunk(
            query, key, value, mask, dropout, head_mask
        )

    # chunk_values = exp_values
    # chunk_weights = exp_weights.sum(dim=-1, keepdim=True)
    # chunk_max = max_score.squeeze(-1)

    # chunk_weights: batch, head, query_length, key_chunk_num, 1
    # chunk_values: batch, head, query_length, key_chunk_num, dim
    # chunk_max: batch, head, query_length, key_chunk_num

    global_max = torch.amax(chunk_max, dim=-1, keepdim=True)
    # global_max: batch, head, query_length, 1
    max_diffs = torch.exp(chunk_max - global_max).unsqueeze(-1)
    # max_diffs: batch, head, query_length, key_chunk_num, 1
    chunk_values = chunk_values * max_diffs
    chunk_weights = chunk_weights * max_diffs
    # chunk_values: batch, head, query_length, key_chunk_num, dim
    # chunk_weights: batch, head, query_length, key_chunk_num, 1

    all_values = chunk_values.sum(dim=-2)
    all_weights = chunk_weights.sum(dim=-2) + 1e-10
    # all_values: batch, head, query_length, dim
    # all_weights: batch, head, query_length, 1

    return all_values / all_weights


def MultiHeadResidualChunkAttention(query, key, value, chunk_size=128, prev_attn=None, mask=None, dropout=None, head_mask=None):
    b, h, l, d = key.size()
    assert l % chunk_size == 0

    key = key.reshape(b, h, l // chunk_size, chunk_size, d)
    value = value.reshape(b, h, l // chunk_size, chunk_size, d)
    query = query * (d ** -0.5)
    prev_attn = prev_attn.reshape(b, h, l, l // chunk_size, chunk_size)

    def summarize_chunk(query, key, value, prev_attn, mask, dropout, head_mask):
        attn_weights = torch.einsum('...id, ...jcd-> ...ijc', query, key)
        attn_weights = attn_weights + prev_attn

        if mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            mb, mh, mq, ml = mask.size()
            mask = mask.view(mb, mh, mq,  ml // chunk_size, chunk_size)
            masked_attn_weights = attn_weights + mask
        else:
            masked_attn_weights = attn_weights

        max_score = torch.amax(masked_attn_weights, dim=-1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(masked_attn_weights - max_score)

        if dropout is not None:
            exp_weights = dropout(exp_weights)

        if head_mask is not None:
            exp_weights = exp_weights * head_mask

        exp_values = torch.einsum(
            '...ncd, ...lnc-> ...lnd', value, exp_weights)
        # exp_weights: batch, head, query_length, key_chunk_num, key_chunk_size
        # exp_values: batch, head, query_length, key_chunk_num, dim

        return exp_values, exp_weights.sum(dim=-1, keepdim=True), max_score.squeeze(-1), attn_weights

    if query.requires_grad:
        chunk_values, chunk_weights, chunk_max, attn_weights = checkpoint.checkpoint(
            summarize_chunk, query, key, value, prev_attn, mask, dropout, head_mask
        )
    else:
        chunk_values, chunk_weights, chunk_max, attn_weights = summarize_chunk(
            query, key, value, prev_attn, mask, dropout, head_mask
        )

    attn_weights = attn_weights.reshape(b, h, l, l)
    # chunk_values = exp_values
    # chunk_weights = exp_weights.sum(dim=-1, keepdim=True)
    # chunk_max = max_score.squeeze(-1)

    # chunk_weights: batch, head, query_length, key_chunk_num, 1
    # chunk_values: batch, head, query_length, key_chunk_num, dim
    # chunk_max: batch, head, query_length, key_chunk_num

    global_max = torch.amax(chunk_max, dim=-1, keepdim=True)
    # global_max: batch, head, query_length, 1
    max_diffs = torch.exp(chunk_max - global_max).unsqueeze(-1)
    # max_diffs: batch, head, query_length, key_chunk_num, 1
    chunk_values = chunk_values * max_diffs
    chunk_weights = chunk_weights * max_diffs
    # chunk_values: batch, head, query_length, key_chunk_num, dim
    # chunk_weights: batch, head, query_length, key_chunk_num, 1

    all_values = chunk_values.sum(dim=-2)
    all_weights = chunk_weights.sum(dim=-2) + 1e-10
    # all_values: batch, head, query_length, dim
    # all_weights: batch, head, query_length, 1

    return all_values / all_weights, attn_weights


