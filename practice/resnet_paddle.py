# import paddle.vision.transforms as T

#加载飞桨和相关类库
from turtle import forward
import paddle
from paddle.nn import Linear
import paddle.nn as nn
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.models import resnet50
from paddle_train import train_pm

class Residual(nn.Layer):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2D(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides, bias_attr=False)
        self.conv2 = nn.Conv2D(num_channels, num_channels,
                               kernel_size=3, padding=1, bias_attr=False)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2D(input_channels, num_channels,
                                   kernel_size=1, stride=strides, bias_attr=False)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2D(num_channels);
        self.bn2 = nn.BatchNorm2D(num_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block = False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(nn.Layer):
    def __init__(self):
        super(ResNet, self).__init__()
    def forward(self, input):
        pass

if __name__ == "__main__":
    b1 = nn.Sequential(nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2D(64), nn.ReLU(),
                       nn.MaxPool2D(kernel_size=3, stride=2,padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128,256,2))
    b5 = nn.Sequential(*resnet_block(256,512,2))
    
    net = nn.Sequential(b1,b2,b3,b4,b5,
                        nn.AdaptiveAvgPool2D((1,1)),
                        nn.Flatten(), nn.Linear(521, 10))
    lr, num_epochs, batch_size = 0.05, 10, 256
    
    
    opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=net.parameters(), weight_decay=0.001)
    train_pm(net, opt)