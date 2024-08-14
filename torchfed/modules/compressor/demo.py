import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchfed.modules.compressor.terngrad import TernGradCompressor
from torchfed.modules.compressor.threshold import ThresholdCompressor
from torchfed.modules.compressor.dgc import DgcCompressor
from torchfed.modules.compressor.randomk import RandomKCompressor

# 定义 ConvolutionalModel
class ConvolutionalModel(nn.Module):
    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)


model = ConvolutionalModel()
compressor = RandomKCompressor(compress_ratio=0.2)  # 设置阈值

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(2):  # 根据需要更改 epoch 数量
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 在测试集上评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Test Accuracy: {accuracy:.4f}')

    #tok的
    # 训练循环
    # for epoch in range(2):  # 根据需要更改 epoch 数量
    #     model.train()
    #     for images, labels in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #
    #         # 压缩和解压梯度
    #         for name, param in model.named_parameters():
    #             if param.grad is not None:
    #                 # 压缩梯度
    #                 tensors, ctx = compressor.compress(param.grad)
    #                 # 解压梯度
    #                 decompressed_grad = compressor.decompress(tensors, ctx)
    #                 # 更新梯度
    #                 param.grad.data = decompressed_grad
    #
    #         optimizer.step()

    # # QSGD 压缩和解压缩梯度
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             print(f"初始参数：{type(param.grad)},{param.grad}")
    #             compressed_grad, shape = compressor.compress(param.grad, name)
    #             print(f"压缩的参数：{type(compressed_grad)},{compressed_grad}")
    #             print(f"shape：{type(shape)},{shape}")
    #             decompressed_grad = compressor.decompress(compressed_grad, shape)
    #             print(f"解压缩后的参数：{type(decompressed_grad)},{decompressed_grad}")
    #             param.grad.copy_(decompressed_grad)


