# import paddle.vision.transforms as T

#加载飞桨和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn as nn
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from paddle.vision.models import resnet50

class Residual(nn.Layer):
    def __init__(self):
        pass

if __name__ == "__main__":
    paddleversion = paddle.__version__
    print("paddle version:", paddleversion)