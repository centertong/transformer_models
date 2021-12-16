import torch
import torch.nn.functional as F

from einops import repeat


def linear_attention(q, k, v, is_casual, gate):
    kv = torch.einsum('bhlm, bhlc -> bhlmc', k, v)
    mask = torch.ones((q.size(-2), q.size(-2)), device=q.device)
    if is_casual:
        mask = torch.triu(mask)

    if gate is not None:
        gate = makeGateFeature(gate).t()
        # gate: b, l, l
        kv_gate = torch.einsum('blj, bhlmc -> bhljmc', gate, kv)
        k_gate = torch.einsum('blj, bhlm -> bhljm', gate, k)

        kv_presum = kv_gate.sum(dim=-3)
        k_presum = k_gate.sum(dim=-2)
    else:
        kv_presum = torch.einsum('bhlmc, lj -> bhjmc', kv, mask)
        k_presum = torch.einsum('bhlm, lj -> bhjm', k, mask)

    D_inv = 1. / torch.einsum('...lc, ...lc -> ...l', q, k_presum.type_as(q))

    context = torch.einsum('bhlm, bhlmc -> bhlc', q, kv_presum)
    out = torch.einsum('bhlc, bhl -> bhlc', context, D_inv)
    return out


def gaussian_random_matrix(nb_rows, nb_columns, num_head, device="cpu"):
    block_list = [torch.randn((1, 1, nb_columns), device=device)
                  for _ in range(num_head * nb_rows)]
    matrix = torch.cat(block_list).view(num_head, nb_rows, nb_columns)
    return matrix


def random_feature_map(data, projection_matrix, mode="arccos"):
    b, *_ = data.shape
    projection = repeat(projection_matrix, 'h j d -> b h j d', b=b)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij',
                             F.normalize(data, p=2, dim=-1), projection)
    if mode == "arccos":
        data_dash = F.relu(data_dash) * (data.shape[-1] ** -0.5) + 1e-5
    else:
        data_dash = torch.cat([torch.sin(data_dash), torch.cos(
            data_dash)], dim=-1) * (data.shape[-1] ** -0.5)
    return data_dash


def RfaAttention(query, key, value, projection_matrix=None, mode="arccos", mask=None, is_casual=False, gate=None):
    query = random_feature_map(
        query, projection_matrix=projection_matrix, mode=mode)
    key = random_feature_map(
        key, projection_matrix=projection_matrix, mode=mode)
    if mask is not None:
        key.masked_fill_(~mask, 0.)

    out = linear_attention(query, key, value, is_casual, gate)
    return out


def makeGateFeature(gate):
    # gate : b, l
    *_, length = gate.size()

    o_gate = 1 - gate
    gate = gate.log()
    o_gate = o_gate.log()

    tri_one = torch.ones((length, length), device=gate.device).triu(diagonal=1)
    ones = torch.ones((length, length), device=gate.device).triu()

    gate = torch.einsum('bl, lj -> blj', gate, tri_one)

    o_gate = torch.diag(o_gate)
    gate = gate + o_gate

    gate = torch.einsum('lj, bjk -> blk', ones, gate).exp()
    gate = gate * ones

    return gate
