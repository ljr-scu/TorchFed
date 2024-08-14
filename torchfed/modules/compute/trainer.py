import aim
from prettytable import PrettyTable

from torchfed.modules.distribute.weighted_data_distribute import WeightedDataDistributing
from torchfed.modules.module import Module
from torchfed.third_party.aim_extension.distribution import Distribution
from torchfed.utils.helper import interface_join


class Trainer(Module):
    def __init__(
            self,
            router,
            model,
            dataloader,
            optimizer,
            loss_fn,
            alias=None,
            visualizer=False,
            writer=None):
        super(
            Trainer,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = None


        self._log_dataset_distribution()

        # if self.visualizer:
        #     # graph_writer = self.get_tensorboard_writer()
        #     inputs, _ = next(iter(self.dataloader))
        #     self.writer.add_graph(self.model, inputs)
        #     # graph_writer.close()

    def get_metrics(self):
        return self.metrics

    # 用于记录数据集分布的方法
    def _log_dataset_distribution(self):
        num_classes = self.dataloader.dataset.num_classes
        labels = []
        #dist 这个字典的键是从 0 到 num_classes - 1 的整数，每个键的值初始都被设置为 0
        dist = {k: 0 for k in range(num_classes)}
        # 遍历数据加载器中的标签数据
        for data in self.dataloader:
            labels.extend(data["labels"].tolist())
        #更新类别计数
        for label in labels:
            dist[label] += 1
        # 使用 PrettyTable 创建表格
        dist_table = PrettyTable()
        dist_table.field_names = dist.keys()
        dist_table.add_row(dist.values())
        # 打印日志信息
        self.logger.info(f"[{self.name}] Dataset distribution:")
        for row in dist_table.get_string().split("\n"):
            self.logger.info(row)
        # 如果存在可视化器，则创建 Distribution 对象并记录
        if self.visualizer:
            dist = Distribution(
                distribution=labels,
                bin_count=num_classes
            )
            self.writer.track(
                dist, name=f"Dataset Distribution/{self.get_path()}")

    def train(self):
        self.model.train()  # 设置模型为训练模式，启用 dropout 等训练时的特定行为
        counter = 0         # 用于计数训练批次的计数器
        running_loss = 0.0  # 用于累积每个批次的训练损失

        # 遍历数据加载器中的每个批次
        for batch_idx, data in enumerate(self.dataloader, 0):
            inputs, labels = data["inputs"], data["labels"]   # 从数据中获取输入和标签
            self.optimizer.zero_grad()       # 梯度清零，防止梯度累积
            outputs = self.model(inputs)      # 使用模型进行前向传播，得到预测值
            loss = self.loss_fn(outputs, labels)  # 计算损失
            loss.backward()    # 反向传播，计算梯度
            self.optimizer.step()   # 根据梯度更新模型参数
            running_loss += loss.item()  # 累积训练损失
            counter += 1     # 更新训练批次计数器
        # 计算平均训练损失
        self.metrics = running_loss / counter
        # 打印训练损失
        self.logger.info(
            f'[{self.name}] Training Loss: {self.metrics:.3f}')
        #可视化训练损失
        if self.visualizer:
            self.writer.track(
                self.metrics,
                name=f"Loss/Train/{self.get_path()}")

    def train_fedprox(self):
        global_model = self.send(
            self.router.get_peers(self)[0],
            interface_join("distributor", WeightedDataDistributing.download),
            ())[0].data
        #
        # print(f"global_model:{global_model}")
        # print(f"model:{self.model.state_dict()}")
        self.model.train()  # 设置模型为训练模式，启用 dropout 等训练时的特定行为
        counter = 0  # 用于计数训练批次的计数器
        running_loss = 0.0  # 用于累积每个批次的训练损失

        # 遍历数据加载器中的每个批次
        for batch_idx, data in enumerate(self.dataloader, 0):
            inputs, labels = data["inputs"], data["labels"]  # 从数据中获取输入和标签
            self.optimizer.zero_grad()  # 梯度清零，防止梯度累积
            outputs = self.model(inputs)  # 使用模型进行前向传播，得到预测值
            # 计算 Proximal Term
            proximal_term = 0.0
            # 确保两个 OrderedDict 的键（参数名称）是对齐的
            for name, cur_param in self.model.state_dict().items():
                if name in global_model:
                    glob_param = global_model[name]

                    # 计算 Proximal Term
                    proximal_term = (cur_param.data - glob_param.data).norm(2)
                    # print(f"proximal_term:{proximal_term}")
            # for w, w_t in zip(self.model.state_dict(), global_model):
            #     proximal_term += (w - w_t).norm(2)

            # 计算损失，包含基本损失和 Proximal Term
            loss = self.loss_fn(outputs, labels) + ( 0.01 / 2) * proximal_term
            loss.backward()  # 反向传播，计算梯度
            self.optimizer.step()  # 根据梯度更新模型参数
            running_loss += loss.item()  # 累积训练损失
            counter += 1  # 更新训练批次计数器
        # 计算平均训练损失
        self.metrics = running_loss / counter

        # 打印训练损失
        self.logger.info(
            f'[{self.name}] Training Loss: {self.metrics:.3f}')
        # 可视化训练损失
        if self.visualizer:
            self.writer.track(
                self.metrics,
                name=f"Loss/Train/{self.get_path()}")
