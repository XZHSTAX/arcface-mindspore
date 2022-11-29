import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision,transforms
import cv2
def get_dataset(image_folder_dataset_dir,phase,pic_size=[128,128],shuffle=True):
    '''
    输入数据集路径，进行相应预处理后返回一个datasets类型的可迭代对象。
    因为使用的是ds.ImageFolderDataset读取数据，从树状结构的文件目录中读取图片构建源数据集，同一个文件夹中的所有图片将被分配相同的label。
    标签顺序为文件夹名排序。

    Args:
        image_folder_dataset_dir: 数据集路径
        phase: "train"表示为数据集，"test"表示为测试集，会对应不同的数据处理方式
        pic_size: 数据处理中，Crop将会对数据进行缩放，传入列表来确定缩放后图片大小
        shuffle: 是否随机打乱
        
    Returns:
        train_dataset: 可迭代对象，datasets类型
    '''
    train_dataset = ds.ImageFolderDataset(image_folder_dataset_dir,decode=True,shuffle=shuffle)
    if phase == "train":
        composed = transforms.Compose(
            [
                # vision.Decode(to_pil=True),
                vision.RandomCrop(pic_size),
                # vision.Resize(pic_size),
                # vision.Grayscale(),
                vision.RandomHorizontalFlip(),
                # vision.ToTensor(),
                vision.Normalize(mean=[0.5*255,0.5*255,0.5*255],std=[0.5*255,0.5*255,0.5*255]),
                vision.HWC2CHW()
            ]
        )
    elif phase == "test":
        composed = transforms.Compose(
            [
                # vision.Decode(to_pil=True),
                vision.CenterCrop(pic_size),
                # vision.Grayscale(),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],is_hwc=False),
            ]
        )        
    train_dataset = train_dataset.map(composed,input_columns="image")
    return     train_dataset



if __name__ == '__main__':
    pic_size = [128,128]
    # image_folder_dataset_dir = "data/CASIA-maxpy-clean"
    image_folder_dataset_dir = "data/data_test"
    my_dataset =  get_dataset(image_folder_dataset_dir,"train",pic_size=pic_size,shuffle=False)
    # image = cv2.imread("data/data_test/0000045/001.jpg", 1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.transpose((2, 0, 1))
    for d,l in my_dataset:
        print(d.shape,l)
