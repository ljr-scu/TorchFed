import torch
import torch.optim as optim

from torchfed.modules.module import Module
from torchfed.modules.compute.trainer import Trainer
from torchfed.modules.compute.tester import Tester
from torchfed.modules.distribute.weighted_data_distribute import WeightedDataDistributing
from torchfed.utils.helper import interface_join
import torchfed.models as models



class DecentralizedFedAvgNode(Module):
    def __init__(
            self,
            router,
            rank,
            dataset_manager,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None,
            ):
        super(
            DecentralizedFedAvgNode,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer,
            override_hparams=override_hparams)

        self.model = getattr(
            models, self.hparams["model"])()
        self.dataset_manager = dataset_manager
        # print(len(self.dataset_manager.get_user_dataset(rank)))

        [self.train_dataset,
         self.test_dataset] = self.dataset_manager.get_user_dataset(rank)

        self.global_test_dataset = self.dataset_manager.get_global_dataset()[1]
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams["batch_size"], shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)
        self.global_test_loader = torch.utils.data.DataLoader(
            self.global_test_dataset, batch_size=self.hparams["batch_size"], shuffle=True)

        self.dataset_size = len(self.train_dataset)
        self.optimizer = getattr(
            optim, self.hparams["optimizer"])(
            self.model.parameters(), lr=self.hparams["lr"])
        self.loss_fn = getattr(torch.nn, self.hparams["loss_fn"])()

        self.distributor = self.register_submodule(
            WeightedDataDistributing, "distributor", router)
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
        self.global_tester = self.register_submodule(
            Tester, "global_tester", router, self.model, self.global_test_loader)



        self.distributor.update(self.model.state_dict())



    def get_default_hparams(self):    #获取默认超参
        return {
            "lr": 1e-3,
            "batch_size": 32,
            "model": "CIFAR100Net",
            "optimizer": "Adam",
            "loss_fn": "CrossEntropyLoss",
            "local_iterations": 10,
        }

    def bootstrap(self, bootstrap_from):
        # 如果指定了启动源（bootstrap_from），则从该源获取全局模型
        if bootstrap_from is not None:
            # 向启动源请求全局模型数据
            global_model = self.send(
                bootstrap_from,
                interface_join(
                    "distributor",
                    WeightedDataDistributing.download),
                ())[0].data
            # 使用获取的全局模型数据加载本地模型
            self.model.load_state_dict(global_model)
        # 更新数据分发器（distributor）的状态，使用本地模型的参数
        self.distributor.update(self.model.state_dict())

    def aggregate(self):
        # generate latest local model，聚合最新本地模型
        # 从数据分发器中获取最新的本地模型参数
        aggregated = self.distributor.aggregate()
        # 如果本地模型参数为空，使用当前模型的参数作为聚合结果
        if aggregated is None:
            aggregated = self.model.state_dict()
        else:
            # 如果获取到了最新的本地模型参数，使用这些参数加载本地模型
            self.model.load_state_dict(aggregated)
        # 使用聚合后的模型参数更新数据分发器的状态
        self.distributor.update(aggregated)

    def train_and_test(self):
        # train and tests

        metrics = self.global_tester.global_test()

        self.tester.test()
        for i in range(self.hparams["local_iterations"]):
            self.trainer.train()
        return metrics

    def upload(self):
        # upload to peers 更新到每个节点
        # 获取当前模块的所有对等节点,并遍历
        for peer in self.router.get_peers(self):
            # 向对等节点发送上传请求，上传本地模型参数
            self.send(
                peer,
                interface_join("distributor", WeightedDataDistributing.upload),
                (self.name,
                 self.dataset_size,
                 self.model.state_dict()))  # 发送方模块的本地模型参数



class CompressorDecentralizedFedAvgNode(DecentralizedFedAvgNode):
    def __init__(self, router, rank, dataset_manager, compressor, *args, **kwargs):
        super().__init__(router, rank, dataset_manager, *args, **kwargs)
        self.compressor = compressor


    def upload(self):
        # Upload to peers

        for peer in self.router.get_peers(self):
            compressed = {}
            for param_name, param in self.model.state_dict().items():
                compressed_tensor, ctx = self.compressor.compress(param,param_name)
                compressed[param_name] = (compressed_tensor,ctx)

            self.send(
                peer,
                interface_join("distributor", WeightedDataDistributing.upload),
                (self.name,
                self.dataset_size,
                 compressed))

    def aggregate(self):

        ret = None
        if len(self.distributor.storage) == 0:
            aggregated_params = ret
        else:
            for data in self.distributor.storage.values():
                [weight, compressed] = data
                decompressed_params = {}
                for param_name, (compressed_tensor, ctx) in compressed.items():
                    decompressed_tensor = self.compressor.decompress(compressed_tensor, ctx)
                    decompressed_params[param_name] = decompressed_tensor
                if decompressed_params is None:
                    continue
                if isinstance(decompressed_params, dict):
                    if ret is None:
                        ret = {k: v * (weight / self.distributor.total_weight)
                               for k, v in decompressed_params.items()}
                    else:
                        ret = {k: ret[k] + v * (weight / self.distributor.total_weight)
                               for k, v in decompressed_params.items()}
                else:
                    if ret is None:
                        ret = decompressed_params * (weight / self.distributor.total_weight)
                    else:
                        ret += decompressed_params * (weight / self.distributor.total_weight)
            self.distributor.total_weight = 0
            self.distributor.storage.clear()
            aggregated_params = ret

        # 更新本地模型
        if aggregated_params is not None:
            self.model.load_state_dict(aggregated_params)
        else:
            # 如果没有聚合数据，使用当前模型参数
            aggregated_params = self.model.state_dict()

        # 6. 更新数据分发器的状态
        self.distributor.update(aggregated_params)






class CentralizedFedAvgServer(Module):        #中心化FedAvg算法服务器
    def __init__(
            self,
            router,
            dataset_manager,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        super(
            CentralizedFedAvgServer,
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


class CentralizedFedAvgClient(Module):
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
            CentralizedFedAvgClient,
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
            self.trainer.train()
        # 将本地模型上传到邻居节点
        self.send(
            self.router.get_peers(self)[0],
            interface_join("distributor", WeightedDataDistributing.upload),
            (self.name,
             self.dataset_size,
             self.model.state_dict()))


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