import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
'''
implement resnet 18
'''
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
                                   kernel_size=1, stride=strides)
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
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        print(X.size())
        print(Y.size())
        Y += X
        return F.relu(Y)
def f1():
    blk = Residual(3, 3) # (out = (in +2*1 - 3) / 1 + 1) ==in # out = floor(in/2+1/2)
    X = torch.rand(4,3,6,6)
    Y = blk(X)
    print(Y.size())
    print("==============") 
def f2():
    blk = Residual(3,6, use_1x1conv=True, strides=2)
    X = torch.rand(4,3,6,6) #
    Y = blk(X)
    print(Y.size())
    print("==============") 
    
def resnet_block(input_channels, num_channels, num_residuals,
                first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

if __name__=="__main__":

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # （224 + 2*3 - 7）/2 +1 = 112
    # （112 + 2*1 - 3）/2 +1 = 56
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    # 56x56
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    # 28x28
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    # 14x14
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # 7x7
    
    net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
    
    # X = torch.rand(size=(1, 1, 224, 224))
    
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t', X.shape)
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())