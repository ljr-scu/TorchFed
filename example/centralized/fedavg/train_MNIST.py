import os
import random

from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.node import CentralizedFedAvgServer, CentralizedFedAvgClient

from torchvision.transforms import transforms
from torchfed.datasets.MNIST import TorchMNIST
from torchfed.managers.dataset_manager import DatasetManager

import config


if __name__ == '__main__':
    # init
    # MASTER_ADDR和MASTER_PORT是通信模块初始化需要的两个环境变量。
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router = TorchDistributedRPCRouter(0, 1, visualizer=True)    #用rpc方式通讯的router


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    #创建了一个名为 "cifar10_manager" 的数据集管理器，该管理器用于协调和管理数据集的加载和存储。
    dataset_manager = DatasetManager("mnist_manager",
                                     TorchMNIST(
                                         "../../data",
                                         config.num_users,
                                         config.num_labels,
                                         download=True,
                                         transform=transform)
                                     )

    server = CentralizedFedAvgServer(router, dataset_manager, visualizer=True)
    clients = []
    for rank in range(config.num_users):      #注册client
        clients.append(
            CentralizedFedAvgClient(
                router,
                rank,
                dataset_manager,
                visualizer=True))

    #定义拓扑结构
    router.connect(server, [client.get_node_name() for client in clients])
    for client in clients:
        router.connect(client, [server.get_node_name()])

    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        server.run()
        for client in random.sample(clients, 5):
            client.run()

    for client in clients:
        client.release()
    server.release()
    router.release()
