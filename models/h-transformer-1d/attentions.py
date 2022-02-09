import torch

def HT1dAttention(query, key, value, hierachi_level):
    queries, keys, values = [query], [key], [value]
    dim = query.size(-1)
    for _ in range(hierachi_level):
        query, key, value = coarsen(query), coarsen(key), coarsen(value)
        queries.append(query)
        keys.append(key)
        values.append(value)

    attns = [dot_product(q, k, dim) for q, k in zip(queries, keys)]
    



def dot_product(q, k, d):
    score = torch.einsum('bhld, bhjd -> bhlj', q, k)
    score = score * (d ** -0.5)
    return torch.exp(score)


def coarsen(x):
    #query: b h l d
    x1 = x[:,:,0::2,:]
    x2 = x[:,:,1::2,:]

    return (x1 + x2) / 2
