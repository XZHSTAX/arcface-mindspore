import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

# class Asoftmax(nn.Cell):
#     '''
#     接收扩维后的数据，输出cos(\theta +m)
#     '''
#     def __init__(self, s=64.0, m=0.5):
#         super(Asoftmax, self).__init__()
#         self.s = s
#         self.shape = ops.Shape()
#         self.mul = ops.Mul()
#         self.cos = ops.Cos()
#         self.acos = ops.ACos()
#         self.onehot = ops.OneHot()
#         self.on_value = Tensor(m, ms.float32)
#         self.off_value = Tensor(0.0, ms.float32)

#     def construct(self, cosine, label):
#         m_hot = self.onehot(label, self.shape(cosine)[1], self.on_value, self.off_value)

#         cosine = self.acos(cosine)
#         cosine += m_hot
#         cosine = self.cos(cosine)
#         cosine = self.mul(cosine, self.s)
#         return cosine

class Asoftmax_loss(SoftmaxCrossEntropyWithLogits):
    def __init__(self, s=64.0, m=0.5):
        super(Asoftmax_loss,self).__init__(sparse=True)
        # SoftmaxCrossEntropyWithLogits.__init__(self)
        self.s = s
        self.m = m
        self.ToOnehot = ops.OneHot()
        self.on_value = Tensor(self.m, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)

    def construct(self, logits, labels):
        logits = self.add_m_at_correctplace(logits,labels)
        # label_onehot = self.ToOnehot(labels, ops.shape(logits)[1], Tensor(1.0, ms.float32),self.off_value)


        # loss = SoftmaxCrossEntropyWithLogits.construct(self,logits,label_onehot)
        loss = super().construct(logits,labels)
        return loss


    def add_m_at_correctplace(self,cosine,label):

        m_hot = self.ToOnehot(label, ops.shape(cosine)[1], self.on_value,self.off_value)

        cosine = ops.acos(cosine)
        cosine += m_hot
        cosine = ops.cos(cosine)
        cosine = ops.mul(cosine, self.s)

        return cosine



if __name__ == '__main__':
    input = Tensor(np.random.random((2,13938)),ms.float32)
    loss_fn = Asoftmax_loss()
    output = loss_fn(input,Tensor([0,1],ms.int32))

    loss_fn2 = SoftmaxCrossEntropyWithLogits(sparse=True)
    output2 = loss_fn2(input,Tensor([0,1],ms.int32))
    print(output.shape)
    print("output",output)
    print("output2",output2)


