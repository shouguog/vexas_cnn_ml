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
print('LeNet')
train_data = torch.load('../datasetFive/train_data_240.pt')
test_data = torch.load('../datasetFive/test_data_240.pt')

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
# 类别信息也是需要我们给定的

examples = enumerate(trainloader)
batch_idx, (example_data, example_label) = next(examples)
# 批量展示图片
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.tight_layout()  #自动调整子图参数，使之填充整个图像区域
    img = example_data[i]
    img = img.numpy() # FloatTensor转为ndarray
    img = np.transpose(img, (1,2,0)) # 把channel那一维放到最后
    #img = img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    plt.imshow(img)
    #plt.title("label:{}".format(example_label[i]))
    plt.title("label:{}".format(1)) #changed for
    plt.xticks([])
    plt.yticks([])
#plt.show()
plt.savefig('VEXAS_train_first4_Five.jpg')
plt.close()
class LeNet(nn.Module):
    def __init__(self, label_num=10, in_channels=1):
        super(LeNet, self).__init__()

        self.conv_pool_1 = nn.Sequential(
            #Output shape = (n + 2p — f + 1) x (n + 2p — f + 1)
            #Where n is input size, f is filter size, and p is the padding amount.
            # 卷积层 (1*240*240) -> 6*240*240)
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # 池化层 (6*240*240) -> (6*120*120)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_pool_2 = nn.Sequential(
            # 卷积层 (6*120*120) -> (16*116*116)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            # 池化层 (16*116*116) -> (16*58*58)
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            # 将卷积池化后的tensor拉成向量
            nn.Flatten(),
            # 全连接层 16*58*58 -> 5568
            nn.Linear(16 * 58 * 58, 1600),
            nn.ReLU(),
            # 全连接层 1600 -> 348
            nn.Linear(1600, 348),
            nn.ReLU(),
            # 全连接层 348 -> 4
            nn.Linear(348, label_num)
        )

    def forward(self, x):
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.fc(x)
        return x
#创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet(label_num=5, in_channels=3).to(device)
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
        loss = F.cross_entropy(outputs, labels[:,[True, True, True, True, True]])  #Shouguo used the only fours
        # 获取最大概率的预测结果
        # dim=1表示返回每一行的最大值对应的列下标
        predict = outputs.argmax(dim=1)
        total += labels.size(0)
        # Shouguo used the only fours
        trueLabel=labels[:, [True, True, True, True, True]].argmax(dim=1)
        correct += (predict == trueLabel).sum().item()#Shouguo used the only fours
        #correct += (predict == labels).sum().item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        if i % 5 == 0:
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
epoch = 500
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
        torch.save(model, './models/lenet_Five_epoch_' + str(epoch) + '-vexas.pth')
print('Finished Training')
plt.subplot(2,1,1)
plt.plot(Loss)
plt.title('Loss')
plt.show()
plt.subplot(2,1,2)
plt.plot(Accuracy)
plt.title('Accuracy')
plt.show()

print(model)
torch.save(model, './models/lenet-vexas_Five.pth')  # 保存模型
print("Program Finished")
