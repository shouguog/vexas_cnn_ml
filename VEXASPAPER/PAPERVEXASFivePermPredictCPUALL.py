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
permseed = 333
print('permutation: ' + str(permseed))
train_data = torch.load('../datasetFive' + str(permseed) + '/train_data.pt')
test_data = torch.load('../datasetFive' + str(permseed) + '/test_data.pt')

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
# 类别信息也是需要我们给定的

#examples = enumerate(trainloader)
#batch_idx, (example_data, example_label) = next(examples)
# 批量展示图片
class Custom_cnn(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, 5)  # Shouguo out_features:4
        )

    def forward(self, x):
        x = self.network(x)
        logits = self.fc1(x)
        return logits
#创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = Custom_cnn(0).to(device)
model = torch.load('./models/paper-vexas-Five' + str(permseed)   + '.pth', map_location=torch.device('cpu'))  # 保存模型

#定义优化器
#optimizer = optim.Adam(model.parameters(), lr=0.01)  #shouguo changed to 0.01 from 0.001

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

# Save the array to a CSV file
fmt = '%1.2f','%1.2f','%1.2f','%1.2f','%1.2f'
np.savetxt('results/vexastest_FINAL' + str(permseed)   + '_output_TESTTRAIN.csv', results, delimiter=',', fmt=fmt)
np.savetxt('results/vexastest_FINAL' + str(permseed)   + '_labels_TESTTRAIN.csv', labelresults, delimiter=',', fmt=fmt)
print("Program Finished")
