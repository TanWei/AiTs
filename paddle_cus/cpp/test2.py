import paddle
from paddle.utils.cpp_extension import load
# 出错可以查看C:\Users\lei\.cache\paddle_extensions
paddle.device.set_device('cpu')

custom_ops = load(
    name="custom_jit_ops",
    sources=[r"/Users/leilei/MyWorkSpace/ml/AiTs/paddle_cus/cpp/relu_cpu_fp32.cc"])

x = paddle.randn([4, 10], dtype='float32')
out = custom_ops.custom_relu(x)
print(x)
