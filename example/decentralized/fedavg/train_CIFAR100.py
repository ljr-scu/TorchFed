import os
import random

from torchfed.routers import TorchDistributedRPCRouter
from torchfed.modules.node import DecentralizedFedAvgNode

from torchvision.transforms import transforms
from torchfed.datasets.CIFAR100 import TorchCIFAR100
from torchfed.managers.dataset_manager import DatasetManager

import config


if __name__ == '__main__':
    # init
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    router = TorchDistributedRPCRouter(0, 1, visualizer=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    dataset_manager = DatasetManager("cifar100_manager",
                                     TorchCIFAR100(
                                         "../../data",
                                         config.num_users,
                                         config.num_labels,
                                         download=True,
                                         transform=transform)
                                     )

    nodes = []
    for rank in range(config.num_users):
        nodes.append(DecentralizedFedAvgNode(router, rank,
                                dataset_manager,
                                visualizer=True))

    # bootstrap
    boostrap_node = nodes[0].get_node_name()
    for node in nodes:
        router.connect(node, [boostrap_node])
        node.bootstrap(boostrap_node)
        router.disconnect(node, [boostrap_node])

    # connect
    for node in nodes:
        current_node_name = node.get_node_name()
        other_nodes_names = [
            n.get_node_name() for n in nodes if n.get_node_name() != current_node_name]
        connected_peers = random.sample(
            other_nodes_names,
            5) + [current_node_name]  # self connect
        print(f"node {current_node_name} will connect to {connected_peers}")
        router.connect(node, connected_peers)

    total_data_transmission_list = []
    metrics_list = []
    # train
    for epoch in range(config.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        for node in nodes:
            node.aggregate()

        sum = 0
        for node in nodes:
            metric = node.train_and_test()
            sum += metric
        metrics = sum / config.num_users
        metrics_list.append(metrics)
        print(f"metrics_list:{metrics_list}")

        for node in nodes:
            node.upload()

        total_data_transmission = router.get_Total_datatransmission()
        total_data_transmission_list.append(total_data_transmission)

    epochs = list(range(1, len(total_data_transmission_list) + 1))

    for node in nodes:
        node.release()
    # print(total_data_transmission_list, epochs)
    router.release()