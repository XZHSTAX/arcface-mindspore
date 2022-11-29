import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.dataset import vision,transforms

from mindvision.engine.callback import LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from MyDataset import get_dataset
from A_softmax import Asoftmax_loss
from resnet import *
from ArcModel import Arcface

lr = 0.01                                                     # 初始学习率
decay_rate = 0.9                                              # 衰减指数底数 

total_epoch = 25                                              # 训练轮次
m = 0.5
s = 64
backbone_net = resnet18                                       # backbone
num_feature = 512                                             # backbone 输出大小
num_classes = 13938                                           # 线性层输出大小
batch_size = 128                                              # 训练包大小

# ckpt_url = "Arcface_ckpt5/Arcface-1_7119.ckpt"
ckpt_url = None                                               # 预训练模型路径
summary_dir= "./summary_dir/summary_01"                       # summary写入路径
directory_Arcface = "./Arcface_ckpt/Arcface_ckpt1"            # ckpt保存路径
collect_summary_freq = 50                                     # summary收集频率（step/次）
save_ckpt_steps = 3560                                        # ckpt保存频率 （step/次）
keep_ckpt_max   = 10                                          # ckpt文件最大保存个数
print_loss_step = 1000                                        # 打印loss值频次 (step/次)
image_folder_dataset_dir = "data/CASIA-maxpy-clean"

if __name__ == '__main__':
    
    train_dataset = get_dataset(image_folder_dataset_dir,"train")
    train_dataset = train_dataset.batch(batch_size)
    
    net = Arcface(backbone_net,num_feature,num_classes)
    if ckpt_url is not None:
        param_dict = load_checkpoint(ckpt_url)
        load_param_into_net(net, param_dict)
    
    loss_fn = Asoftmax_loss(s=s,m=m)
    # loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True) 


    one_epoch_step = train_dataset.get_dataset_size()
    total_step = one_epoch_step*total_epoch
    decay_steps = one_epoch_step

    # 指数下降学习率（floor）
    exponential_decay_lr = nn.ExponentialDecayLR(lr, decay_rate, decay_steps,is_stair=True)
    lr_list = nn.exponential_decay_lr(lr,decay_rate,total_step,decay_steps,1,True)
        
    opt = nn.SGD(net.trainable_params(),learning_rate=exponential_decay_lr)

    # 模型保存callback
    config_ck = ms.CheckpointConfig(save_checkpoint_steps=save_ckpt_steps, keep_checkpoint_max=keep_ckpt_max)
    ckpoint = ms.ModelCheckpoint(prefix="Arcface", directory=directory_Arcface, config=config_ck)

    # mindisight记录 callback
    specified = {"collect_metric": True,"collect_graph": True,"collect_dataset_graph": True}

    summary_collector = ms.SummaryCollector(summary_dir=summary_dir, collect_specified_data=specified,
                                            collect_freq=collect_summary_freq, keep_default_action=False)



    model = ms.Model(network=net,
                    loss_fn=loss_fn,
                    optimizer=opt,
                    metrics={"Accuracy": nn.Accuracy()})
    
    print("="*20,"traing","="*20)
    
    model.train(total_epoch, 
                train_dataset, 
                callbacks=[LossMonitor(lr_list,print_loss_step),ckpoint,summary_collector],
                dataset_sink_mode=False)
            