import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision,transforms
import cv2
def get_dataset(image_folder_dataset_dir,phase,pic_size=[128,128],shuffle=True):
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
