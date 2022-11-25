import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.dataset import vision,transforms

from mindvision.engine.callback import LossMonitor

from MyDataset import get_dataset
from A_softmax import Asoftmax_loss
from resnet import *
from ArcModel import Arcface

lr = 0.0005
decay_rate = 0.9
decay_steps = 500
total_epoch = 1
m = 0.35
s = 30

if __name__ == '__main__':
    image_folder_dataset_dir = "data/CASIA-maxpy-clean"
    train_dataset = get_dataset(image_folder_dataset_dir,"train")
    train_dataset = train_dataset.batch(64)
    
    net = Arcface(resnet50,512,13938)
    loss_fn = Asoftmax_loss(s=s,m=m)


    one_epoch_step = train_dataset.get_dataset_size()
    total_step = one_epoch_step*total_epoch

    # 指数下降学习率（floor）
    exponential_decay_lr = nn.ExponentialDecayLR(lr, decay_rate, decay_steps,is_stair=True)
    lr_list = nn.exponential_decay_lr(lr,decay_rate,total_step,decay_steps,1,True)
        
    opt = nn.SGD(net.trainable_params(),learning_rate=exponential_decay_lr)

    # 模型保存callback
    config_ck = ms.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=10)
    ckpoint = ms.ModelCheckpoint(prefix="Arcface", directory="./Arcface_ckpt2", config=config_ck)

    # mindisight记录 callback
    specified = {"collect_metric": True,"collect_graph": True,
                    "collect_dataset_graph": True}

    summary_collector = ms.SummaryCollector(summary_dir="./summary_dir/summary_03", collect_specified_data=specified,
                                            collect_freq=10, keep_default_action=False)



    model = ms.Model(network=net,
                    loss_fn=loss_fn,
                    optimizer=opt,
                    metrics={"Accuracy": nn.Accuracy()})
    print("-"*20,"traing","-"*20)
    model.train(total_epoch, 
                train_dataset, 
                callbacks=[LossMonitor(lr_list,25),ckpoint,summary_collector],
                dataset_sink_mode=False)
# Epoch:[  0/  1], step:[   25/ 7119], loss:[20.135/20.139], time:174.653 ms, lr:0.00050
# Epoch:[  0/  1], step:[ 7100/ 7119], loss:[19.246/19.467], time:194.893 ms, lr:0.00011
# Epoch time: 12128485.731 ms, per step time: 1703.678 ms, avg loss: 19.466


#TODO: 已经把训练过程中要打印的数据模块加入，学习率自动变化加入，模型自动保存加入；下面需要定制mindinsight收集训练过程中的数据

