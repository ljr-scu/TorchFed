import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfed.types.named import Named

class FMNISTNet(nn.Module,Named):
    @property
    def name(self) -> str:
        return "FMNISTNet"

    def __init__(self):
        super(FMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 将输入通道数从 3 修改为 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 由于MNIST图像尺寸为 28x28，经过两次池化后为 4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 输出维度修改为 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class FMNISTDNN(nn.Module,Named):
    @property
    def name(self) -> str:
        return "FMNISTDNN"

    def __init__(self):
        super(FMNISTDNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(512, 256)      # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(256, 128)      # 第二个隐藏层到第三个隐藏层
        self.fc4 = nn.Linear(128, 10)       # 第三隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)            # 将输入图像展平成一维向量
        x = F.relu(self.fc1(x))            # 第一层激活函数
        x = F.relu(self.fc2(x))            # 第二层激活函数
        x = F.relu(self.fc3(x))            # 第三层激活函数
        x = self.fc4(x)                    # 输出层，不加激活
        return F.log_softmax(x, dim=1)     # 使用log softmax输出分类结果
