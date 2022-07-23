from turtle import forward
import paddle
from paddle.autograd import PyLayer

class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        # ctx 为PyLayerContext对象，可以把y从forward传递到backward。
        ctx.save_for_backward(y)
        return y
        
    @staticmethod
    def backward(ctx, dy):
        y, = ctx.saved_tensor()
        
        grad = dy * (1 - paddle.square(y))
        return grad
    
    
data = paddle.randn([2,3], dtype="float32")
data.stop_gradient = False

z = cus_tanh.apply(data)
z.mean().backward()

