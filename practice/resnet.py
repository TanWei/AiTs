import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels);
        self.bn2 = nn.BatchNorm2d(num_channels);
        '''
        公式--输出w：
        w_out = floor(
            (w_in + 2 * padding - kernel_size) / stride + 1
            )
        '''
    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))