from resnet import *
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer


class Arcface(nn.Cell):
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



