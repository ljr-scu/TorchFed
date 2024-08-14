import torch.nn as nn
import torch.nn.functional as F

from torchfed.types.named import Named


class CIFAR10Net(nn.Module, Named):
    @property
    def name(self) -> str:
        return "CIFAR10Net"

    def __init__(self):
        # 在python中创建类后，通常会创建一个\ __init__()方法，这个方法会在创建类的实例的时候自动执行。 \ __init__()方法必须包含一个self参数，而且要是第一个参数。
        # 继承原有模型，super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数，其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 定义conv1函数的是图像卷积函数：输入为图像（3通道）,输出为 6张特征图, 卷积核为5x5正方形
        self.pool = nn.MaxPool2d(2, 2)  #max pooling窗口大小为2，步长为2
        self.conv2 = nn.Conv2d(6, 16, 5)# 定义conv1函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1 = nn.Linear(16 * 5 * 5, 120)# 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2 = nn.Linear(120, 84)#定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3 = nn.Linear(84, 10)#定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#用relu激活函数作为一个池化层，先用conv1卷积，在激活
        x = self.pool(F.relu(self.conv2(x)))#用relu激活函数作为一个池化层，先用conv2卷积，在激活
        x = x.view(-1, 16 * 5 * 5)   #view函数相当于reshape，这行代码是将形状为 [batch_size, 16, 5, 5] 的张量 x 展平为形状为 [batch_size, 16 * 5* 5] 的张量。 其中 -1 表示该维度的大小由其他维度的大小自动推断得出
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CIFAR100Net(nn.Module, Named):
    @property
    def name(self) -> str:
        return "CIFAR100Net"

    def __init__(self):
        super(CIFAR100Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
