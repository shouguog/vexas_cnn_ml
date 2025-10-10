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
    #将图片尺寸resize到299x299
    transforms.Resize((299,299)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])
pipline_test = transforms.Compose([
    #将图片尺寸resize到299x299
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

###Let us load the dataset
permseed = 333
print('GoogLeNet_InceptionV3 permutation: ' + str(permseed))
train_data = torch.load('../datasetFive' + str(permseed) + '/train_data_299.pt')
test_data = torch.load('../datasetFive' + str(permseed) + '/test_data_299.pt')

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
# 类别信息也是需要我们给定的

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionV3_A(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2red, out_channels_2, out_channels_3red, out_channels_3, out_channels_4):
        super(InceptionV3_A, self).__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels_1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2red, kernel_size=1),
            ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_3red, kernel_size=1),
            ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=3, padding=1),   # 保证输出大小等于输入大小
            ConvBNReLU(out_channels_3, out_channels_3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_4, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionV3_B(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2red, out_channels_2, out_channels_3red, out_channels_3, out_channels_4):
        super(InceptionV3_B, self).__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels_1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2red, kernel_size=1),
            ConvBNReLU(out_channels_2red, out_channels_2red, kernel_size=[1,7], padding=[0,3]),
            ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=[7,1], padding=[3,0])
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_3red, kernel_size=1),
            ConvBNReLU(out_channels_3red, out_channels_3red, kernel_size=[1,7], padding=[0,3]),
            ConvBNReLU(out_channels_3red, out_channels_3red, kernel_size=[7,1], padding=[3,0]),
            ConvBNReLU(out_channels_3red, out_channels_3red, kernel_size=[1,7], padding=[0,3]),
            ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=[7,1], padding=[3,0])
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_4, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionV3_C(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2red, out_channels_2, out_channels_3red,
                 out_channels_3, out_channels_4):
        super(InceptionV3_C, self).__init__()

        self.branch1 = ConvBNReLU(in_channels, out_channels_1, kernel_size=1)

        self.branch2_conv1x1 = ConvBNReLU(in_channels, out_channels_2red, kernel_size=1)
        self.branch2_conv1x3 = ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=[1, 3], padding=[0, 1])
        self.branch2_conv3x1 = ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=[3, 1], padding=[1, 0])

        self.branch3_conv1x1 = ConvBNReLU(in_channels, out_channels_3red, kernel_size=1)
        self.branch3_conv3x3 = ConvBNReLU(out_channels_3red, out_channels_3, kernel_size=3, padding=1)
        self.branch3_conv3x1 = ConvBNReLU(out_channels_3, out_channels_3, kernel_size=[3, 1], padding=[1, 0])
        self.branch3_conv1x3 = ConvBNReLU(out_channels_3, out_channels_3, kernel_size=[1, 3], padding=[0, 1])

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels_4, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2_tmp = self.branch2_conv1x1(x)
        branch2 = torch.cat([self.branch2_conv1x3(branch2_tmp), self.branch2_conv3x1(branch2_tmp)], dim=1)
        branch3_tmp = self.branch3_conv1x1(x)
        branch3_tmp = self.branch3_conv3x3(branch3_tmp)
        branch3 = torch.cat([self.branch3_conv3x1(branch3_tmp), self.branch3_conv1x3(branch3_tmp)], dim=1)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionV3_D(nn.Module):
    def __init__(self, in_channels, out_channels_1red, out_channels_1, out_channels_2red, out_channels_2):
        super(InceptionV3_D, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_1red, kernel_size=1),
            ConvBNReLU(out_channels_1red, out_channels_1, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2red, kernel_size=1),
            ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(out_channels_2, out_channels_2, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)

class InceptionV3_E(nn.Module):
    def __init__(self, in_channels, out_channels_1red, out_channels_1, out_channels_2red, out_channels_2):
        super(InceptionV3_E, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_1red, kernel_size=1),
            ConvBNReLU(out_channels_1red, out_channels_1, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels_2red, kernel_size=1),
            ConvBNReLU(out_channels_2red, out_channels_2red, kernel_size=[1, 7], padding=[0, 3]),
            ConvBNReLU(out_channels_2red, out_channels_2red, kernel_size=[7, 1], padding=[3, 0]),
            ConvBNReLU(out_channels_2red, out_channels_2, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = ConvBNReLU(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        self.conv2 = ConvBNReLU(128, 768, kernel_size=5)  # output[batch, 128, 4, 4]
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc(x)
        return x


class GoogLeNetV3(nn.Module):
    def __init__(self, num_classes=2, aux_logits=True, init_weights=False):
        super(GoogLeNetV3, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv4 = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv5 = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inceptionA1 = InceptionV3_A(in_channels=192, out_channels_1=64, out_channels_2red=48, out_channels_2=64,
                                         out_channels_3red=64, out_channels_3=96, out_channels_4=32)
        self.inceptionA2 = InceptionV3_A(in_channels=256, out_channels_1=64, out_channels_2red=48, out_channels_2=64,
                                         out_channels_3red=64, out_channels_3=96, out_channels_4=64)
        self.inceptionA3 = InceptionV3_A(in_channels=288, out_channels_1=64, out_channels_2red=48, out_channels_2=64,
                                         out_channels_3red=64, out_channels_3=96, out_channels_4=64)

        self.inceptionD1 = InceptionV3_D(in_channels=288, out_channels_1red=384, out_channels_1=384,
                                         out_channels_2red=64, out_channels_2=96)

        self.inceptionB1 = InceptionV3_B(in_channels=768, out_channels_1=192, out_channels_2red=128, out_channels_2=192,
                                         out_channels_3red=128, out_channels_3=192, out_channels_4=192)
        self.inceptionB2 = InceptionV3_B(in_channels=768, out_channels_1=192, out_channels_2red=160, out_channels_2=192,
                                         out_channels_3red=160, out_channels_3=192, out_channels_4=192)
        self.inceptionB3 = InceptionV3_B(in_channels=768, out_channels_1=192, out_channels_2red=160, out_channels_2=192,
                                         out_channels_3red=160, out_channels_3=192, out_channels_4=192)
        self.inceptionB4 = InceptionV3_B(in_channels=768, out_channels_1=192, out_channels_2red=192, out_channels_2=192,
                                         out_channels_3red=192, out_channels_3=192, out_channels_4=192)

        if self.aux_logits:
            self.aux = InceptionAux(in_channels=768, num_classes=num_classes)

        self.inceptionE1 = InceptionV3_E(in_channels=768, out_channels_1red=192, out_channels_1=320,
                                         out_channels_2red=192, out_channels_2=192)

        self.inceptionC1 = InceptionV3_C(in_channels=1280, out_channels_1=320, out_channels_2red=384,
                                         out_channels_2=384, out_channels_3red=448, out_channels_3=384,
                                         out_channels_4=192)
        self.inceptionC2 = InceptionV3_C(in_channels=2048, out_channels_1=320, out_channels_2red=384,
                                         out_channels_2=384, out_channels_3red=448, out_channels_3=384,
                                         out_channels_4=192)

        self.max_pool3 = nn.MaxPool2d(8, stride=1, ceil_mode=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # ------ 输入块 ------#
        # N x 3 x 224 x 224
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        # ------ Inception ------#
        # InceptionAx3
        x = self.inceptionA1(x)
        x = self.inceptionA2(x)
        x = self.inceptionA3(x)
        # InceptionDx1
        x = self.inceptionD1(x)
        # InceptionBx4
        x = self.inceptionB1(x)
        x = self.inceptionB2(x)
        x = self.inceptionB3(x)
        x = self.inceptionB4(x)
        # InceptionAux
        if self.training and self.aux_logits:  # eval model lose this layer
            aux = self.aux(x)
        # InceptionEx1
        x = self.inceptionE1(x)
        # InceptionCx1
        x = self.inceptionC1(x)
        x = self.inceptionC2(x)

        # ------ 输出块 ------#
        x = self.max_pool3(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Shouguo torch.cuda.set_device(1)  # 指定GPU
model_name = "googlenetV3"
#model = GoogLeNetV3(num_classes=5, aux_logits=True, init_weights=False)
model = torch.load('./models/googlenetv3-vexas-Five' + str(permseed) + '.pth', map_location=torch.device('cpu'))  # 保存模型
model.to(device)
#定义优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0004)

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
            iloc = iloc + 1
            data, label = data.to(device), label.to(device)
            output = model(data)
            output_num = output.cpu().numpy()
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
# Save the array to a CSV file
fmt = '%1.2f','%1.2f','%1.2f','%1.2f','%1.2f'
np.savetxt('results/V3_test_FINAL' + str(permseed) + '_output_TESTTRAIN.csv', results, delimiter=',', fmt=fmt)
np.savetxt('results/V3_test_FINAL' + str(permseed) + '_labels_TESTTRAIN.csv', labelresults, delimiter=',', fmt=fmt)
print("Program Finished")
