import torch
import torch.optim as optim

from torchfed.modules.module import Module
from torchfed.modules.compute.trainer import Trainer
from torchfed.modules.compute.tester import Tester
from torchfed.modules.distribute.weighted_data_distribute import WeightedDataDistributing
from torchfed.utils.helper import interface_join
import torchfed.models as models








class CentralizedFedProxServer(Module):        #中心化FedAvg算法服务器
    def __init__(
            self,
            router,
            dataset_manager,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        super(
            CentralizedFedProxServer,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer,
            override_hparams=override_hparams)
        self.model = getattr(
            models, self.hparams["model"])()

        self.dataset_manager = dataset_manager
        test_dataset = self.dataset_manager.get_global_dataset()[1]
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.distributor = self.register_submodule(
            WeightedDataDistributing, "distributor", router)
        self.global_tester = self.register_submodule(
            Tester, "global_tester", router, self.model, self.test_loader)

        self.distributor.update(self.model.state_dict())

    def get_default_hparams(self):
        return {
            "model": "CIFAR10Net",
            "batch_size": 32,
        }

    def run(self):
        self.global_tester.test()

        aggregated = self.distributor.aggregate()
        if aggregated is None:
            aggregated = self.model.state_dict()
        else:
            self.model.load_state_dict(aggregated)
        self.distributor.update(aggregated)



class CentralizedFedProxClient(Module):
    def __init__(
            self,
            router,
            rank,
            dataset_manager,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        super(
            CentralizedFedProxClient,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer,
            override_hparams=override_hparams)
        self.model = getattr(
            models, self.hparams["model"])()       #getattr() 函数用于返回一个对象属性值。

        self.dataset_manager = dataset_manager
        [self.train_dataset,
         self.test_dataset] = self.dataset_manager.get_user_dataset(rank)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams["batch_size"], shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.dataset_size = len(self.train_dataset)
        self.optimizer = getattr(
            optim, self.hparams["optimizer"])(
            self.model.parameters(), lr=self.hparams["lr"])
        self.loss_fn = getattr(torch.nn, self.hparams["loss_fn"])()

        self.trainer = self.register_submodule(
            Trainer,
            "trainer",
            router,
            self.model,
            self.train_loader,
            self.optimizer,
            self.loss_fn)
        self.tester = self.register_submodule(
            Tester, "tester", router, self.model, self.test_loader)

    def get_default_hparams(self):
        return {
            "lr": 1e-3,
            "batch_size": 32,
            "model": "CIFAR10Net",
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
            "local_iterations": 10,
        }

    def run(self):
        # 从邻居节点下载全局模型
        global_model = self.send(
            self.router.get_peers(self)[0],
            interface_join("distributor", WeightedDataDistributing.download),
            ())[0].data
        self.model.load_state_dict(global_model)
        # 在本地测试集上进行测试
        self.tester.test()
        # 进行本地迭代训练
        for i in range(self.hparams["local_iterations"]):
            self.trainer.train_fedprox()
        # 将本地模型上传到邻居节点
        self.send(
            self.router.get_peers(self)[0],
            interface_join("distributor", WeightedDataDistributing.upload),
            (self.name,
             self.dataset_size,
             self.model.state_dict()))