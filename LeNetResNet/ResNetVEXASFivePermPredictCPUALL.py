import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import time
from matplotlib import pyplot as plt

from PIL import Image
from torch.utils.data import Dataset

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
    #将图片尺寸resize到240x240
    transforms.Resize((240,240)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])
pipline_test = transforms.Compose([
    #将图片尺寸resize到240x240
    transforms.Resize((240,240)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

###Let us load the dataset
permseed = 333
print('Resnet permutation: ' + str(permseed))
train_data = torch.load('../datasetFive' + str(permseed) + '/train_data_240.pt')
test_data = torch.load('../datasetFive' + str(permseed) + '/test_data_240.pt')

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
# 类别信息也是需要我们给定的

examples = enumerate(trainloader)
batch_idx, (example_data, example_label) = next(examples)
# 批量展示图片
class BasicBlock(nn.Module):
    multiplier = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.multiplier, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.multiplier),
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels * self.multiplier or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.multiplier, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * self.multiplier)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv2(self.conv1(x))
        shortcut = self.shortcut(x)
        return self.relu(residual + shortcut)


class Bottleneck(nn.Module):
    multiplier = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.multiplier, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels * self.multiplier)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels * self.multiplier or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.multiplier, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * self.multiplier)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        resiudual = self.conv3(self.conv2(self.conv1(x)))
        shortcut = self.shortcut(x)
        return self.relu(resiudual + shortcut)


class ResNet(nn.Module):

    def __init__(self, layer_num=18, label_num=10, in_channels=1):
        super(ResNet, self).__init__()
        self.base_channels = 64

        block_type, block_nums = self.res_net_params(layer_num)

        self.conv_pool_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.base_channels, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.res_layers = nn.Sequential(
            self.res_layer(block_type, 64, block_nums[0], stride=1),
            self.res_layer(block_type, 128, block_nums[1], stride=2),
            self.res_layer(block_type, 256, block_nums[2], stride=2),
            self.res_layer(block_type, 512, block_nums[3], stride=2)
        )

        # 平均池化，平均池化成1*1
        self.avg_pool_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * block_type.multiplier, label_num)
        )

    def res_layer(self, block_type, out_channel, block_num, stride):
        blocks = []
        for _ in range(block_num):
            new_block = block_type(in_channels=self.base_channels, out_channels=out_channel, stride=stride)
            blocks.append(new_block)
            self.base_channels = out_channel * new_block.multiplier
        return nn.Sequential(*blocks)

    def res_net_params(self, layer_num):
        if layer_num == 18:
            return BasicBlock, [2, 2, 2, 2]
        if layer_num == 34:
            return BasicBlock, [3, 4, 6, 3]
        if layer_num == 50:
            return Bottleneck, [3, 4, 6, 3]
        if layer_num == 101:
            return Bottleneck, [3, 4, 23, 3]
        if layer_num == 152:
            return Bottleneck, [3, 8, 36, 3]

    def forward(self, x):
        x = self.conv_pool_layer(x)
        x = self.res_layers(x)
        x = self.avg_pool_layer(x)
        x = self.fc_layer(x)
        return x
#创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = ResNet(layer_num=18, in_channels=3, label_num=5).to(device)
model = torch.load('./models/ResNet-vexas-Five' + str(permseed) + '.pth', map_location=torch.device('cpu'))  # 保存模型
#定义优化器
#optimizer = optim.Adam(model.parameters(), lr=0.001)

def test_runner(model, device, testloader):
    #模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
    #因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    #统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    #torch.no_grad将不会计算梯度, 也不会进行反向传播
    results = np.array([[0, 0, 0, 0,0]])
    labelresults = np.array([[0, 0, 0, 0,0]])
    iloc = 1
    with torch.no_grad():
        for data, label in testloader:
            print(iloc)
            iloc = iloc+1
            data, label = data.to(device), label.to(device)
            output = model(data)
            output_num = output.cpu().numpy()
            #print(output_num[0,0])
            #print(output_num.shape)
            results = np.concatenate((results, output_num), axis=0)
            labelresults = np.concatenate((labelresults, label.cpu().numpy()), axis=0)
            test_loss += F.cross_entropy(output, label).item()
            predict = output.argmax(dim=1)
            #计算正确数量
            total += label.size(0)
            # Shouguo used the only fours
            trueLabel = label[:, [True, True, True, True, True]].argmax(dim=1)
            correct += (predict == trueLabel).sum().item()  # Shouguo used the only fours
            #correct += (predict == label).sum().item()
        #计算损失值
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))
    return results, labelresults
#调用
epoch = 500
results, labelresults = test_runner(model, device, testloader)
results = np.delete(results, 0, axis=0)
labelresults = np.delete(labelresults, 0, axis=0)

resultsTrain, labelresultsTrain = test_runner(model, device, trainloader)
resultsTrain = np.delete(resultsTrain, 0, axis=0)
labelresultsTrain = np.delete(labelresultsTrain, 0, axis=0)

#Merge
results = np.concatenate((results, resultsTrain), axis=0)
labelresults = np.concatenate((labelresults, labelresultsTrain), axis=0)
fmt = '%1.2f','%1.2f','%1.2f','%1.2f','%1.2f'
np.savetxt('results/resnet_test_FINAL' + str(permseed) + '_output_TESTTRAIN.csv', results, delimiter=',', fmt=fmt)
np.savetxt('results/resnet_test_FINAL' + str(permseed) + '_labels_TESTTRAIN.csv', labelresults, delimiter=',', fmt=fmt)
print("Program Finished")
