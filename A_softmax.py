import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

class Asoftmax(nn.Cell):
    '''
    接收扩维后的数据，输出cos(\theta +m)
    '''
    def __init__(self, s=64.0, m=0.5):
        super(Asoftmax, self).__init__()
        self.s = s
        self.shape = ops.Shape()
        self.mul = ops.Mul()
        self.cos = ops.Cos()
        self.acos = ops.ACos()
        self.onehot = ops.OneHot()
        self.on_value = Tensor(m, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)

    def construct(self, cosine, label):
        m_hot = self.onehot(label, self.shape(cosine)[1], self.on_value, self.off_value)

        cosine = self.acos(cosine)
        cosine += m_hot
        cosine = self.cos(cosine)
        cosine = self.mul(cosine, self.s)
        return cosine


if __name__ == '__main__':
    input = Tensor(np.random.random((2,13938)),ms.float32)
    model = Asoftmax()
    output = model(input,Tensor([0,1],ms.int32))
    print(output.shape)


