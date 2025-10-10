import torch
from torchvision import datasets,transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
class MyDataset(Dataset):
    def __init__(self, X, Y, txt_path, transform = None, target_transform = None):
        super(MyDataset, self).__init__()
        # import and initialize dataset
        self.X =X
        self.Y = Y.astype(np.float32)
        self.root_dir = txt_path
        self.transform = transform

    def __getitem__(self, index):
        # get item by index
        # color imgs
        img = Image.open(self.root_dir + self.X[index].replace('./', '/')).convert('RGB')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.Y[index]
        #im = self.transforms(image=im)["image"]
        img = self.transform(img)
        return img, torch.from_numpy(label)
    def __len__(self):
        return len(self.X)

pipline_train = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #随机旋转图片
    transforms.RandomHorizontalFlip(),
    #将图片尺寸resize到224x224
    transforms.Resize((224,224)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

pipline_test = transforms.Compose([
    #将图片尺寸resize到224x224
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])


import pandas as pd
data_dir = "../../../../data/vexas_cnn_data/processed/"
data = pd.read_csv(data_dir + 'train.csv')
x_train = data["path"].values
y_train = data.drop(['Unnamed: 0','path', 'group'], axis=1).values
train_data = MyDataset(x_train, y_train, data_dir, transform=pipline_train)
data_test = pd.read_csv(data_dir + 'test.csv')
x_train_test = data_test["path"].values
y_train_test = data_test.drop(['path', 'N', 'group'], axis=1).values
test_data = MyDataset(x_train_test, y_train_test, data_dir, transform=pipline_test)
###Let us save the dataset
torch.save(train_data, './train_data_224.pt')
torch.save(test_data, './test_data_224.pt')
