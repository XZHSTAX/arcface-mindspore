import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

from resnet import *
from MyDataset import get_dataset
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

    # image_folder_dataset_dir = "data/CASIA-maxpy-clean"
    # train_dataset = get_dataset(image_folder_dataset_dir,"train")
    # train_dataset = train_dataset.batch(64)

    # net = resnet50(13938)
    # loss_fn = Asoftmax_loss()
    # loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True)
    # for d,l in train_dataset:
    #     output = net(d)
    #     loss = loss_fn(output,l)
    #     print(loss)   
    pass 
#FIXME: 遇到了问题，loss一开始就为nan。实际上，如果使用随机生成数据输入，不会出现nan，但只要是数据集输入，loss就位nan；
# 当使用随机生成数据输入：SoftmaxCrossEntropyWithLogits的loss值小于Asoftmax_loss
# 当使用数据集输入：SoftmaxCrossEntropyWithLogits正常，但Asoftmax_loss输出为nan
# 猜测是Asoftmax_loss什么地方写错了

# 解决方案如下：
# 1. 修复Asoftmax_loss中的错误
# 2. 重载resnet，使得输出值直接接上add_m_at_correctplace，然后使用SoftmaxCrossEntropyWithLogits。放弃写新的loss函数的念想


