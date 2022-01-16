#加载飞桨、Numpy和相关类库
import paddle
import paddle.nn as nn
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random

def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

class Regressor(nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        #  定义一层全连接层，输入维度为13，输出维度为1
        self.rc = Linear(in_features=13, out_features=1)
    def forward(self, inputs):
        x = self.rc(inputs)
        return x
    
model = Regressor()
model.train()
training_data, test_data = load_data()
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

EPOCH_NUM = 10
BATCH_SIZE = 10
for epoch_id in range(EPOCH_NUM):
    np.random.shuffle(training_data)
    # mini_batches shape = len(training_data) // BATCH_SIZE, 10, 14
    mini_batches = [training_data[k:k+BATCH_SIZE] 
                    for k in range(0, len(training_data), BATCH_SIZE)
                 ]
    print(mini_batches.shape)
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])
        y = np.array(mini_batch[:, -1:])
        
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)
        
        #前向传播
        predicts = model.forward(x)
        
        loss = F.square_error_cost(predicts, label=prices)