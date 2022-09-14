import paddle
from custom_ops import tanh_op
import numpy as np
x = np.random.random((4, 10)).astype("float32")
print(x)
custom_ops_x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))
custom_ops_x.stop_gradient = False
custom_ops_y = tanh_op(custom_ops_x)
custom_ops_y.backward()
grad = custom_ops_x.gradient()

print("==========================================================")
print("前向传播：")
print(custom_ops_y)
print("==========================================================")
print("检测是否在GPU上：")
print(custom_ops_y.place)
print("==========================================================")
print("梯度：")
print(grad)
