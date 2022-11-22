import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision,transforms

def get_dataset(image_folder_dataset_dir,phase,pic_size=[128,128]):
    train_dataset = ds.ImageFolderDataset(image_folder_dataset_dir,decode=True,shuffle=True)
    if phase == "train":
        composed = transforms.Compose(
            [
                # vision.Decode(to_pil=True),
                vision.RandomCrop(pic_size),
                # vision.Grayscale(),
                vision.RandomHorizontalFlip(),
                vision.Normalize(mean=[0.5],std=[0.5]),
                vision.ToTensor()
            ]
        )
    elif phase == "test":
        composed = transforms.Compose(
            [
                # vision.Decode(to_pil=True),
                vision.CenterCrop(pic_size),
                # vision.Grayscale(),
                vision.Normalize(mean=[0.5],std=[0.5]),
                vision.ToTensor()
            ]
        )        
    train_dataset = train_dataset.map(composed,input_columns="image")
    return     train_dataset



if __name__ == '__main__':
    # image_folder_dataset_dir = "data/data_test"
    # train_dataset = ds.ImageFolderDataset(image_folder_dataset_dir,decode=False,shuffle=True)
    # for d,l in train_dataset:
    #     print(d.shape,l)

    # print("-"*20)

    # composed = transforms.Compose(
    #     [
    #         vision.Decode(to_pil=True),
    #         vision.RandomCrop(pic_size),
    #         vision.Grayscale(),
    #         vision.RandomHorizontalFlip(),
    #         vision.Normalize(mean=[0.5],std=[0.5]),
    #         vision.ToTensor()
    #     ]
    # )

    # train_dataset = train_dataset.map(composed,input_columns="image")
    # for d,l in train_dataset:
    #     print(d.shape,l)

    pic_size = [128,128]
    # image_folder_dataset_dir = "data/CASIA-maxpy-clean"
    image_folder_dataset_dir = "data/data_test"
    my_dataset =  get_dataset(image_folder_dataset_dir,"train",pic_size=pic_size)
    
    for d,l in my_dataset:
        print(d.shape,l)
