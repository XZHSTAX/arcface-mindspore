from resnet import *
import mindspore.nn as nn
from mindspore import Parameter
from mindspore.common.initializer import initializer


class Arcface(nn.Cell):
    def __init__(self,backbone_net,num_feature,num_classes):
        super(Arcface,self).__init__()
        self.num_feature = num_feature
        self.num_classes = num_classes

        self.weight = Parameter(initializer("normal", (self.num_classes, self.num_feature),ms.float16), name="mp_weight")
        self.L2Norm = ops.L2Normalize(axis=1)
        self.linear = ops.MatMul(transpose_b=True)
        self.resnet = backbone_net(self.num_feature)

    def construct(self,data):
        output = self.resnet(data)
        norm_output = self.L2Norm(output)
        norm_weight = self.L2Norm(self.weight)
        
        output = self.linear(norm_weight,norm_output)
        return output

        


