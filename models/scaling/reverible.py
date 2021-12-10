import torch
import torch.nn as nn
from torch.autograd.function import Function


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks):
        for block in blocks:
            x = block(x)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        for block in ctx.blocks:
            y, dy = block.backward_apply(y, dy)

        return dy


class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)

        with torch.no_grad():
            y1 = x1 + self.f(x2)
            y2 = x2 + self.g(y1)

        return torch.cat([y1, y2], dim=-1)

    def backward_apply(self, y, dy):
        y1, y2 = torch.chunk(y, 2, dim=-1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=-1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1)
            torch.autograd(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.required_grad = True
            fx2 = self.f(x2)
            torch.autograd.backward(fx2, dx1)  # , retain_graph=True)

        with torch.no_grad():
            x2 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=-1)
            dx = torch.cat([dx1, dx2], dim=-1)

        return x, dx


class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()

        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])

    def forward(self, x):
        return _ReversibleFunction.apply(x, self.blocks)
