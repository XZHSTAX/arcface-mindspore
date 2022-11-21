import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision,transforms
from MyDataset import get_dataset




if __name__ == '__main__':
    image_folder_dataset_dir = "data/data_test"
    train_dataset = get_dataset(image_folder_dataset_dir,"train")
    for d,l in train_dataset:
        print(d.shape)