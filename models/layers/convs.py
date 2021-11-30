import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding) -> None:
        super().__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, groups=in_channels)


    def forward(self, x):
        return self.conv(x)

class LightweightConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, shared, kernel, padding) -> None:
        super().__init__()

        assert out_channels % in_channels == 0
        assert in_channels % shared == 0

        self.w = Parameter(torch.rand((out_channels// shared, 1, kernel)))
        self.bias = Parameter(torch.zeros((out_channels//2)))
        self.s = shared
        self.padding = padding
        self.groups = in_channels // shared

    def forward(self, x):
        # x : batch, channel, length
        b, c, l = x.size()
        w = F.softmax(self.w, dim=-1)
        x = x.reshape(b * self.s, c // self.s, l)
        x = F.conv1d(x, w, self.bias, padding=self.padding, groups=self.groups)
        x = x.view(b, c, l)
        return x

class DynamicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, shared, kernel, padding) -> None:
        super().__init__()


        self.bias = Parameter(torch.zeros((out_channels//2)))
        self.s = shared
        self.padding = padding
        self.groups = in_channels // shared
        self.k = kernel
        
        self.conv = nn.Conv1d(in_channels=in_channels // shared, out_channels=out_channels // shared, kernel_size=kernel, padding=padding, groups=in_channels // shared)
        self.fc = nn.Linear(in_channels, kernel)

    def forward(self, x):
        # B, L, C
        # 1, L, C*B
        w = self.fc(x).view(-1, 1, self.k)
        

        # w: C * B, 1, kenel

        # 
        w = self.fc(x)
        self.conv.weight.data = w
        x = self.conv(x)
        return x