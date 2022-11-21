import mindspore.nn as nn
from resnet import *
from A_softmax import Asoftmax



class ArcFace(nn.Cell):
    def __init__(self,backbone,my_softmax):
        super(ArcFace, self).__init__()
        self.backbone = backbone
        self.my_softmax = my_softmax
    def construct(self, data, label):
        x = self.backbone(data)
        x = self.my_softmax(x,label)
        return x

def get_ArcFaceModel(backbone=resnet50(13938),my_softmax=Asoftmax()):
    return ArcFace(backbone,my_softmax)


if __name__ == '__main__':
    input = Tensor(np.random.random((2,3,128,128)),ms.float32)
    model = get_ArcFaceModel()
    output = model(input,Tensor([0,1],ms.int32))
    print(output.shape)