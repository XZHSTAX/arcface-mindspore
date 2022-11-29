from resnet import *
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer


class Arcface(nn.Cell):
    '''
    构建Arcface模型，arcface的模型实际上就是backbone+线性层的结构。
    需要注意的是，线性层的输出为分类数。backbone的输出需要归一化，线性层的权重也需要归一化。否则后面的Asoftmax会出现loss值为nan问题（因为arccos没有对应值）。

    Args:
        backbone_net: backbone,需要是nn.cell类型的，并且能接收参数num_feature作为backbone输出的大小。
        num_feature:  backbone输出大小
        num_classes:  Arcface的线性层需要输出的大小。
        test:         是否测试，如果为True，则直接输出backbone的输出。False则输出线性层输出
    '''
    def __init__(self,backbone_net,num_feature,num_classes,test=False):
        super(Arcface,self).__init__()
        self.num_feature = num_feature
        self.num_classes = num_classes

        self.weight = Parameter(initializer("normal", (self.num_feature,self.num_classes),ms.float32), name="mp_weight")
        self.L2Norm = ops.L2Normalize(axis=1)
        self.linear = ops.MatMul(transpose_b=False)
        self.resnet = backbone_net(self.num_feature)
        self.test = test
        # self.tensor_summary = ops.TensorSummary()


    def construct(self,data):
        # self.tensor_summary("Weight", self.weight)
        output = self.resnet(data)                       # output.shape = (batch_size,num_feature)
        if self.test: return output
        norm_output = self.L2Norm(output)
        norm_weight = self.L2Norm(self.weight)           # norm_weight.shape = (num_feature,num_classes)
        output = self.linear(norm_output,norm_weight)    # output.shape = (batch_size,num_classes)
        return output




if __name__ == "__main__":
    # 用np.random.random生成一个模拟的图片输入
    input = Tensor(np.random.random((2,3,128,128)),ms.float32)
    model = Arcface(resnet50,512,13938)
    output = model(input)
    # print(output)
    print(output.shape)



