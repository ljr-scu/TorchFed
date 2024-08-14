import sys
import os.path
import random
from typing import Optional, Callable, List

import numpy as np

import torch
import torchvision
from tqdm import trange, tqdm

from torchfed.datasets import TorchDataset, TorchUserDataset, TorchGlobalDataset


class TorchCIFAR10(TorchDataset):
    num_classes = 10

    def __init__(
            self,
            root: str,
            num_users: int,
            num_labels_for_users: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            rebuild: bool = False,
            cache_salt: int = 0,
    ) -> None:
        super().__init__(
            root,
            TorchCIFAR10.num_classes,
            num_users,
            num_labels_for_users,
            transform,
            target_transform,
            download,
            rebuild,
            cache_salt
        )

    # @property装饰器用于将一个方法转换为只读属性，即通过调用这个方法可以获取一个属性值，但不能通过直接赋值的方式修改属性的值。
    @property
    def name(self) -> str:
        return "CIFAR10"

    #加载用户数据集
    # 返回类型被注解为List[List[TorchUserDataset]]
    def load_user_dataset(self) -> List[List[TorchUserDataset]]:
        train_dataset = torchvision.datasets.CIFAR10(
            self.root, True, self.transform, self.target_transform, self.download)        #下载cifar10数据集，root表示存储路径，‘True’表示加载其训练集部分
        test_dataset = torchvision.datasets.CIFAR10(
            self.root, False, self.transform, self.target_transform, self.download)      #测试集部分
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(
                train_dataset.data), shuffle=False)                   #创建一个数据加载器，参数分别是：训练集对象，batch大小，这里是一次性加载整个训练集，shuffle是每个epoch是否需要洗牌
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(
                test_dataset.data), shuffle=False)

        tensor_train_dataset, tensor_test_dataset = {}, {}
        for _, data in tqdm(enumerate(train_dataloader, 0), file=sys.stdout):          #tqdm(..., file=sys.stdout)：这部分使用 tqdm 函数创建了一个进度条，用于可视化训练集的迭代进度。file=sys.stdout 表示将进度条显示在标准输出中
            tensor_train_dataset["data"], tensor_train_dataset["targets"] = data       #enumerate 函数遍历 train_dataloader 中的每个批次。enumerate 返回一个元组，其中包含批次的索引（在这里用不到，用 _ 表示忽略）和批次的数据。
        for _, data in tqdm(enumerate(test_dataloader, 0), file=sys.stdout):           #将训练集中每个批次的图像数据赋值给 tensor_train_dataset["data"]，将相应的标签数据赋值给 tensor_train_dataset["targets"]
            tensor_test_dataset["data"], tensor_test_dataset["targets"] = data

        inputs, split_inputs, labels = [], [], []     #分别用于存储所有图像数据、分割后的图像数据和所有标签数据
        inputs.extend(tensor_train_dataset["data"].cpu().detach().numpy())#将训练集的图像数据（存储在 tensor_train_dataset["data"] 中）添加到 inputs 列表中。cpu().detach().numpy() 将 PyTorch 张量转换为 NumPy 数组，并且 extend 方法将数组中的元素逐一添加到列表中
        inputs.extend(tensor_test_dataset["data"].cpu().detach().numpy())
        labels.extend(tensor_train_dataset["targets"].cpu().detach().numpy())
        labels.extend(tensor_test_dataset["targets"].cpu().detach().numpy())
        inputs = np.array(inputs)       #将 inputs 和 labels 转换为 NumPy 数组。
        labels = np.array(labels)

        for label in trange(self.num_classes):        #遍历每个类别的标签，从 inputs 数组中选择所有属于当前标签 label 的图像数据，并将其添加到 split_inputs 列表中
            split_inputs.append(inputs[labels == label])
        _, num_channels, num_height, num_width = split_inputs[0].shape    #返回一个包含维度信息的元组，而通过 _ 忽略第一个元素，将其余元素赋值给 num_channels、num_height 和 num_width

        user_x = [[] for _ in range(self.num_users)]       #创建了一个包含 self.num_users 个空列表的列表。这样的列表嵌套结构通常用于表示多个用户，每个用户的数据以列表的形式存储在 user_x 中。
        user_y = [[] for _ in range(self.num_users)]       #创建了一个包含 self.num_users 个空列表的列表，即 user_y。这里可能用于存储每个用户的标签数据
        idx = np.zeros(self.num_classes, dtype=np.int64)   #创建了一个长度为 self.num_classes 的 NumPy 数组 idx，并用零进行初始化。dtype=np.int64 指定数组的数据类型为 64 位整数。这个数组可能用于统计每个类别出现的次数或作为某种索引。

        for user_idx in trange(self.num_users):             #遍历每个用户
            for label_idx in range(self.num_labels_for_users):      #遍历每个用户所分配的标签数量
                assigned_label = (user_idx + label_idx) % self.num_classes      #计算当前用户分配的标签
                user_x[user_idx] += split_inputs[assigned_label][idx[assigned_label]: idx[assigned_label] + 10].tolist()      #[idx[assigned_label]: idx[assigned_label] + 10]这是一个切片操作，从当前类别的图像数据中选择从 idx[assigned_label] 开始到 idx[assigned_label] + 10 结束的部分。这样就获得了当前用户分配的标签对应的类别的前 10 个图像数据。
                user_y[user_idx] += (assigned_label * np.ones(10)).tolist()        #.tolist()：这将 NumPy 数组转换为 Python 列表。将当前用户的 user_y 列表扩展了一部分，包含了对应于上面图像数据的标签数据。这是通过创建一个包含 10 个 assigned_label 的 NumPy 数组，将其转换为列表
                idx[assigned_label] += 10         #更新了 idx 数组中当前类别已经分配的样本数量，增加了 10。
        #创建了一个形状为 (10, self.num_users, self.num_labels_for_users) 的三维数组 props，其中每个元素都是从指定的对数正态分布中随机生成的值。
        props = np.random.lognormal(
            0, 2., (10, self.num_users, self.num_labels_for_users)
        )
        #np.array([[[len(v) - self.num_users]] for v in split_inputs]])构建了一个三维数组，表示每个类别的剩余图像数量，
        # ‘* props’：这是逐元素的数组乘法，将上述构建的数组与之前生成的 props 数组相乘。这样就得到了一组调整后的权重，其中每个权重与相应的类别相关
        #/ np.sum(props, (1, 2), keepdims=True)：这一部分用于将权重进行归一化，以确保它们的总和为 1。
        # 具体来说，它计算了 props 数组在第一个维度上（即 axis=1）和第二个维度上（即 axis=2）的和，保持了维度的形状（keepdims=True）。然后，通过逐元素的数组除法，将之前的权重调整为归一化后的权重。
        props = np.array([[[len(v) - self.num_users]] for v in split_inputs]
                         ) * props / np.sum(props, (1, 2), keepdims=True)

        for user_idx in trange(self.num_users):
            for label_idx in range(self.num_labels_for_users):
                assigned_label = (user_idx + label_idx) % self.num_classes
                num_samples = int(
                    props[assigned_label, user_idx // int(self.num_users / 10), label_idx])       #根据之前计算的权重 props，获取当前用户分配的标签对应的类别以及该类别中的第几组（可能与用户数量有关）的权重，并将其转换为整数。
                num_samples += random.randint(300, 600)
                if self.num_users <= 20:
                    num_samples *= 2
                if idx[assigned_label] + \
                        num_samples < len(split_inputs[assigned_label]):
                    user_x[user_idx] += split_inputs[assigned_label][idx[assigned_label] : idx[assigned_label] + num_samples].tolist()
                    user_y[user_idx] += (assigned_label *
                                         np.ones(num_samples)).tolist()
                    idx[assigned_label] += num_samples
            #将用户的图像数据转换为 PyTorch 张量，并重新调整形状以匹配模型的输入格式。
            user_x[user_idx] = torch.Tensor(user_x[user_idx]) \
                .view(-1, num_channels, num_width, num_height) \
                .type(torch.float32)
            user_y[user_idx] = torch.Tensor(user_y[user_idx]) \
                .type(torch.int64)

        user_dataset = []
        for user_idx in trange(self.num_users):
            combined = list(zip(user_x[user_idx], user_y[user_idx]))    #将用户的图像数据 user_x[user_idx] 和标签数据 user_y[user_idx] 组合成一个元组列表。
            random.shuffle(combined)   #随机打乱组合后的列表。
            user_x[user_idx], user_y[user_idx] = zip(*combined)   #将打乱后的列表解压缩，分别赋值给 user_x[user_idx] 和 user_y[user_idx]。

            num_samples = len(user_x[user_idx])
            train_len = int(num_samples * 0.75)  #计算训练集的长度，这里设置为总样本数量的 75%。
            #创建一个 TorchUserDataset 类的实例，其中包含用户的训练数据，
            train_user_data = TorchUserDataset(user_idx,
                                               user_x[user_idx][:train_len],
                                               user_y[user_idx][:train_len],
                                               self.num_classes)
            #创建一个 TorchUserDataset 类的实例，其中包含用户的测试数据
            test_user_data = TorchUserDataset(user_idx,
                                              user_x[user_idx][train_len:],
                                              user_y[user_idx][train_len:],
                                              self.num_classes)
            #将训练数据集和测试数据集的元组添加到 user_dataset 列表中。每个用户的数据集信息都被组织成一个子列表。
            user_dataset.append([train_user_data, test_user_data])

        return user_dataset

    #加载全局数据集
    def load_global_dataset(self) -> List[TorchGlobalDataset]:
        train_dataset = torchvision.datasets.CIFAR10(
            self.root, True, self.transform, self.target_transform, self.download)
        test_dataset = torchvision.datasets.CIFAR10(
            self.root, False, self.transform, self.target_transform, self.download)
        return [
            TorchGlobalDataset(train_dataset, self.num_classes),
            TorchGlobalDataset(test_dataset, self.num_classes)
        ]
