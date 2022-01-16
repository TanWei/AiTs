import numpy as np
import json as json
# datafile = 'housing.data'
# data = np.fromfile(datafile,sep=' ')


# featur_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
#                  'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# featrure_num = len(featur_names) #14
# date = data.reshape(featur_names, data.shape[0] // featrure_num, featrure_num)

# x=data[0] #查看数据集

# ratio = 0.8 #80% for training
# offset = int(data.shape[0] * ratio)
# training_data = data[:offset]
# print(training_data.shape)

# maximums, minimums, avgs = \
#     training_data.max(axis=0), \
#     training_data.min(axis=0), \
#     training_data.sum(axis=0) / training_data.shape[0]
# print(maximums, minimums, avgs)

# #每列归一化
# for i in range(featrure_num):
#     data[:,i] = (data[:i] - minimums[i]) / (maximums[i] - minimums[i])

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

training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:,-1:]

w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1])
t=np.dot(x[0], w)
b = -0.2
z = t+b
class Network(object):
    def __init__(self,num_of_weights):
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0
    def forward(self, x):
        z = np.dot(x,self.w) + self.b
        return z
    def loss(self, z , y):
        error = z - y
        cost = np.mean(error*error)
        return cost
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)      
        return gradient_w, gradient_b
    
    
net = Network(13)
z = net.forward(x)
loss = net.loss(z, y)
# 绘图开始
w5 = np.arange(-10.0, 10.0, 1.0)
w9 = np.arange(-10.0, 10.0, 1.0)
# print(w5, w9)
losses  = np.zeros([len(w5),len(w9)])
print(losses)
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]
        net.w[9] = w9[i]
        z = net.forward(x)
        loss = net.loss(z, y)
        losses[i, j] = loss

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

w5, w9 = np.meshgrid(w5, w9)

ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
plt.show()
#绘图结束
gradient_w = (z - y) * x
gradient_w = np.mean(gradient_w, axis=0)
gradient_w = gradient_w[:, np.newaxis]


z = net.forward(x) #前向传播
gradient_w = (z - y) * x #计算梯度
gradient_w = np.mean(gradient_w, axis=0) #计算平均梯度
gradient_w = gradient_w[:, np.newaxis] #将梯度转换为[13, 1]的形状
gradient_w

gradient_b = (z - y)
gradient_b = np.mean(gradient_b)
# 此处b是一个数值，所以可以直接用np.mean得到一个标量
gradient_b

