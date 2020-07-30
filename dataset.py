import numpy as np
from skimage import color
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import pandas as pd


class RSNA_Data(Dataset):
    def __init__(self, name_list, train_png_dir, transform):
        super(RSNA_Data, self).__init__()
        self.name_list = name_list
        self.transform = transform
        self.train_png_dir = train_png_dir

    def __getitem__(self, idx):

        filename = self.name_list[idx]
        filepath = os.path.join(self.train_png_dir, filename)
        image = Image.open(filepath)  # PIL [0,255]
        img = self.transform(image)  # pytorch tensor numpy [0,1]

        img2 = self.transform(image)
        img = torch.cat([img, img2], dim=0)

        # print(label)
        # exit(0)

        return img

    def __len__(self):
        return len(self.name_list)


class RSNA_Data_finetune(Dataset):
    def __init__(self, name_list, label_csv, train_png_dir, transform):
        super(RSNA_Data_finetune, self).__init__()
        self.name_list = name_list
        self.transform = transform
        self.train_png_dir = train_png_dir
        self.label_csv = pd.read_csv(label_csv)
        # print(self.label_csv.head())
        self.label_csv.set_index(['Image'], inplace=True)
        # print(self.label_csv.head())

    def __getitem__(self, idx):
        filename = self.name_list[idx]
        filepath = os.path.join(self.train_png_dir, filename)
        # print(filepath)
        image = Image.open(filepath)
        img = self.transform(image)

        #img2 = self.transform(image)
        #img = torch.cat([img, img2], dim=0)
        labels = torch.tensor(self.label_csv.loc[filename[:-4], ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])

        # print(label)
        # exit(0)
        return img, labels

    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    #root = "F:/MICCAI 2020/unsupervised represent learning for segmentation/MoCo/tiny-imagenet-200/train/"
    mean = [0.5]
    std = [0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
        # transforms.RandomGrayscale(p=0.2),
        #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    name_file = "/media/ubuntu/data/train.txt"
    f_train = open(name_file)
    c_train = f_train.readlines()
    f_train.close()
    name_file = [s.replace('\n', '') for s in c_train]
    csv_label = "train.csv"
    png_dir = "/media/ubuntu/data/RSNATR"
    dataset = RSNA_Data_finetune(name_file, csv_label, png_dir, train_transform)
    loader = DataLoader(dataset, batch_size=2)
    data, label = next(iter(loader))
    print(data.size(), label)
