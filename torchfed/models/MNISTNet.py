import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfed.types.named import Named

class MNISTNet(nn.Module,Named):
    @property
    def name(self) -> str:
        return "MNISTNet"

    def __init__(self):
        super(MNISTNet, self).__init__()
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

class MNISTDNN(nn.Module,Named):
    @property
    def name(self) -> str:
        return "MNISTDNN"

    def __init__(self):
        super(MNISTDNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # 输入层：28x28 = 784维输入
        self.fc2 = nn.Linear(512, 256)      # 隐藏层1
        self.fc3 = nn.Linear(256, 128)      # 隐藏层2
        self.fc4 = nn.Linear(128, 64)       # 隐藏层3
        self.fc5 = nn.Linear(64, 10)        # 输出层：10个类别对应10个神经元

    def forward(self, x):
        x = x.view(-1, 28 * 28)            # 将图像展开为一维向量
        x = F.relu(self.fc1(x))            # 激活函数ReLU
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)                    # 输出层不过激活函数，交给损失函数处理
        return F.log_softmax(x, dim=1)     # 使用log softmax生成类别的概率对数值
