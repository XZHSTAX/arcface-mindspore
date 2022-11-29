import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

from resnet import *
from MyDataset import get_dataset
from ArcModel import Arcface
# class Asoftmax():
#     '''
#     接收扩维后的数据，输出cos(\theta +m)
#     '''
#     def __init__(self, s=64.0, m=0.5):
#         # super(Asoftmax, self).__init__()
#         self.s = s
#         self.shape = ops.Shape()
#         self.mul = ops.Mul()
#         self.cos = ops.Cos()
#         self.acos = ops.ACos()
#         self.onehot = ops.OneHot()
#         self.on_value = Tensor(m, ms.float32)
#         self.off_value = Tensor(0.0, ms.float32)

#     def __call__(self, cosine, label):
#         m_hot = self.onehot(label, self.shape(cosine)[1], self.on_value, self.off_value)

#         cosine = self.acos(cosine)
#         cosine += m_hot
#         cosine = self.cos(cosine)
#         cosine = self.mul(cosine, self.s)
#         return cosine

class Asoftmax_loss(SoftmaxCrossEntropyWithLogits):
    '''
    这是一个损失函数。

    Arcface 的核心，A-softmax损失函数。其本质是对最后一层线性层输出处理，然后使用softmax，最后使用交叉熵损失函数来计算。
    为了最简单的实现，这里继承SoftmaxCrossEntropyWithLogits，重载其中的construct方法，对线性层输出使用add_m_at_correctplace方法处理后，再进行loss计算。

    Args:
        s: 对应论文的s
        m: 对应论文的m

    Note:
        继承时固定了sparse=True,reduction="mean"。所以标签不需要onehot模式，数字即可。loss将会以mean方式进行计算。
        这里记录一下mindspore中SoftmaxCrossEntropyWithLogits与pytorch中CrossEntropy的区别：
        
        主要在于pytorch中reduction="mean"是默认参数，而mindspore中默认参数是None，一开始没注意这一点，导致我的学习率不能设置很大，收敛速度非常慢，训练没有效果。
    '''
    def __init__(self, s=64.0, m=0.5):
        super(Asoftmax_loss,self).__init__(sparse=True,reduction="mean")
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
        '''
        对应Arcface中，把线性层输出标签位置的 cos(\theta) 变为 cos(\theta+m)的操作。
        这里大量使用了库函数，使用了反三角函数，实际上使用余弦函数的两角和差公式更好，更快：cos(a+b) = cos(a)cos(b)-sin(a)sin(b)
        但为了简单直观，就直接使用库函数吧。

        Args:
            cosine: 线性层的输出
            label:  数据标签，不需要onthot格式，直接数字即可
        '''
        m_hot = self.ToOnehot(label, ops.shape(cosine)[1], self.on_value,self.off_value)

        cosine = ops.acos(cosine)
        cosine += m_hot
        cosine = ops.cos(cosine)
        cosine = ops.mul(cosine, self.s)

        return cosine



if __name__ == '__main__':
    image_folder_dataset_dir = "data/CASIA-maxpy-clean"
    train_dataset = get_dataset(image_folder_dataset_dir,"train")
    train_dataset = train_dataset.batch(64)
    loss_fn = Asoftmax_loss()
    loss_fn2 = SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")
    net = Arcface(resnet18,512,13938)

    for d,l in train_dataset:
        output = net(d)
        L2Nor = ops.L2Normalize(axis=1)
        output = L2Nor(output)

        
        loss = loss_fn(output,l)
        loss2 = loss_fn2(output,l)
        print(loss)   
        print(loss2)

#FIXME: 遇到了问题，loss一开始就为nan。实际上，如果使用随机生成数据输入，不会出现nan，但只要是数据集输入，loss就位nan；
# 当使用随机生成数据输入：SoftmaxCrossEntropyWithLogits的loss值小于Asoftmax_loss
# 当使用数据集输入：SoftmaxCrossEntropyWithLogits正常，但Asoftmax_loss输出为nan
# 猜测是Asoftmax_loss什么地方写错了
# DONE: 已经解决loss为nan的问题，主要是对原模型掌握不清楚：
# 1. resnet 输出的特征为512维度（原文），然后需要使用扩维矩阵把特征变为类别，我忘写了
# 2. resnet 输出的特征需要归一化，扩维矩阵的权重需要归一化，否则Asoftmax_loss中使用arc cos 会导致没有值的问题（cos的最大值为1，如果输入大于1，arc cos就无法获得真值）
# 3. 为了解决上述问题，把扩维矩阵和resnet组合成一个模型，写了ArcModel.py,并且归一化了所需的数据
