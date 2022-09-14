import paddle
from paddle.utils.cpp_extension import load
# 出错可以查看C:\Users\lei\.cache\paddle_extensions
paddle.device.set_device('gpu')

custom_ops = load(
    name="custom_jit_ops",
    sources=['tanh.cpp', 'tanh.cu'])
print('dynamic compile success')
x = paddle.randn([4, 10], dtype='float32')
out = custom_ops.tanh_op(x)
print(x)
