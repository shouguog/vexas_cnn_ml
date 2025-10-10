import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import time
from matplotlib import pyplot as plt
import sys
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
    #将图片尺寸resize到227x227
    transforms.Resize((227,227)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])
pipline_test = transforms.Compose([
    #将图片尺寸resize到227x227
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

###Let us load the dataset
#print('VGGNetBatchNormVEXASFive')
#train_data = torch.load('../datasetFive/train_data.pt')
#test_data = torch.load('../datasetFive/test_data.pt')
#print('VGGNetBatchNormVEXASFiveTest')
#train_data = torch.load('../datasetFiveTest/train_data.pt')
#test_data = torch.load('../datasetFiveTest/test_data.pt')
print('VGGNetBatchNormVEXASFiveTest2')
train_data = torch.load('../datasetFiveTest2/train_data.pt')
test_data = torch.load('../datasetFiveTest2/test_data.pt')

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
# 类别信息也是需要我们给定的

examples = enumerate(trainloader)
batch_idx, (example_data, example_label) = next(examples)
# 批量展示图片
###The structure is from https://www.kaggle.com/code/datastrophy/vgg16-pytorch-implementation
###https://stackoverflow.com/questions/57605094/the-training-loss-of-vgg16-implemented-in-pytorch-does-not-decrease
class VGG16(torch.nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()

        # construct model
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv11_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv12_bn = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv21_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv22_bn = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv31_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv32_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv33_bn = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv41_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv42_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv43_bn = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv51_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv52_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv53_bn = nn.BatchNorm2d(512)

        self.fc6 = nn.Linear(512*7*7, 500)
        self.fc7 = nn.Linear(500, 200)
        self.fc8 = nn.Linear(200, n_classes)

    def forward(self, x):
        x = F.relu(self.conv11_bn(self.conv1_1(x)))
        x = F.relu(self.conv12_bn(self.conv1_2(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv22_bn(self.conv2_1(x)))
        x = F.relu(self.conv21_bn(self.conv2_2(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv31_bn(self.conv3_1(x)))
        x = F.relu(self.conv32_bn(self.conv3_2(x)))
        x = F.relu(self.conv33_bn(self.conv3_3(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv41_bn(self.conv4_1(x)))
        x = F.relu(self.conv42_bn(self.conv4_2(x)))
        x = F.relu(self.conv43_bn(self.conv4_3(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv51_bn(self.conv5_1(x)))
        x = F.relu(self.conv52_bn(self.conv5_2(x)))
        x = F.relu(self.conv53_bn(self.conv5_3(x)))
        x = F.max_pool2d(x, (2, 2))

        ##x = x.view(-1, self.num_flat_features(x))
        x = x.reshape(x.shape[0], -1) #replace above line, Shouguo
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

#创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(n_classes=5).to(device) #Shouguo init_weights=True
#定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train_runner(model, device, trainloader, optimizer, epoch):
    # 训练模型, 启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0
    correct = 0.0

    # enumerate迭代已加载的数据集,同时获取数据和数据下标
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # 把模型部署到device上
        inputs, labels = inputs.to(device), labels.to(device)
        # 初始化梯度
        optimizer.zero_grad()
        # 保存训练结果
        outputs = model(inputs)
        # 计算损失和
        # 多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
        loss = F.cross_entropy(outputs, labels[:,[True, True,True, True, True]])  #Shouguo used the only fours
        # 获取最大概率的预测结果
        # dim=1表示返回每一行的最大值对应的列下标
        predict = outputs.argmax(dim=1)
        total += labels.size(0)
        # Shouguo used the only fours
        trueLabel=labels[:, [True,True, True, True, True]].argmax(dim=1)
        correct += (predict == trueLabel).sum().item()#Shouguo used the only fours
        #correct += (predict == labels).sum().item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        if i % 2 == 0:
            # loss.item()表示当前loss的数值
            print(
                "Train Epoch {} \t Iteration {} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, i, loss.item(), 100 * (correct / total)))
            Loss.append(loss.item())
            Accuracy.append(correct / total)
    return loss.item(), correct / total

def test_runner(model, device, testloader):
    #模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
    #因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    #统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    #torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
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

#调用
epoch = 300
Loss = []
Accuracy = []
for epoch in range(1, epoch+1):
    print("start_time",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    loss, acc = train_runner(model, device, trainloader, optimizer, epoch)
    Loss.append(loss)
    Accuracy.append(acc)
    test_runner(model, device, testloader)
    print("end_time: ",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'\n')
    if epoch % 10 == 0:
        #torch.save(model, './models/vggnet16_Five_epoch_' + str(epoch) + '-vexas.pth')
        #torch.save(model, './models/vggnet16_FiveTest_epoch_' + str(epoch) + '-vexas.pth')
        torch.save(model, './models/vggnet16_FiveTest2_epoch_' + str(epoch) + '-vexas.pth')
print('Finished Training')

print(model)
#torch.save(model, './models/vggnet16-vexas-Five.pth')  # 保存模型
#torch.save(model, './models/vggnet16-vexas-FiveTest.pth')  # 保存模型
torch.save(model, './models/vggnet16-vexas-FiveTest2.pth')  # 保存模型
print("End")

