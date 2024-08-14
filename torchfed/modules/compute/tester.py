from prettytable import PrettyTable
import torch
from torchfed.modules.module import Module
from torchfed.third_party.aim_extension.distribution import Distribution


class Tester(Module):
    def __init__(
            self,
            router,
            model,
            dataloader,
            alias=None,
            visualizer=False,
            writer=None):
        super(
            Tester,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.model = model
        self.dataloader = dataloader
        self.metrics = None


        self._log_dataset_distribution()

    def get_metrics(self):
        return self.metrics

    def _log_dataset_distribution(self):
        num_classes = self.dataloader.dataset.num_classes
        labels = []
        dist = {k: 0 for k in range(num_classes)}
        for data in self.dataloader:
            labels.extend(data["labels"].tolist())
        for label in labels:
            dist[label] += 1
        dist_table = PrettyTable()
        dist_table.field_names = dist.keys()
        dist_table.add_row(dist.values())
        self.logger.info(f"[{self.name}] Dataset distribution:")
        for row in dist_table.get_string().split("\n"):
            self.logger.info(row)

        if self.visualizer:
            dist = Distribution(
                distribution=labels,
                bin_count=num_classes
            )
            self.writer.track(
                dist, name=f"Dataset Distribution/{self.get_path()}")

    def test(self):
        self.model.eval()  # 设置模型为评估模式，禁用 dropout 等评估时不需要的特定行为
        correct = 0   # 用于计数正确分类的样本数
        total = 0     # 用于计数总共的样本数
        # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
        with torch.no_grad():
            # 遍历测试数据加载器中的每个批次
            for batch_idx, data in enumerate(
                    self.dataloader, 0):
                inputs, labels = data["inputs"], data["labels"]
                outputs = self.model(inputs)
                #torch.max(outputs.data, 1)：该函数返回每行的最大值及其索引。第一个返回值是每行的最大值，而第二个返回值是每行最大值所在的索引，即模型预测的类别
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)     #统计总样本数
                correct += (predicted == labels).sum().item()  #将正确分类的样本数（即匹配的数量）累加到变量 correct 中。
        self.metrics = 100 * correct / total  # 计算测试准确率
        # 打印测试准确率
        self.logger.info(
            f'[{self.name}] Test Accuracy: {self.metrics:.3f} %')
        # 如果存在可视化器，将测试准确率记录下来
        if self.visualizer:
            self.writer.track(
                self.metrics,
                name=f"Accuracy/Test/{self.get_path()}")


    def global_test(self):
        self.model.eval()  # 设置模型为评估模式，禁用 dropout 等评估时不需要的特定行为
        correct = 0   # 用于计数正确分类的样本数
        total = 0     # 用于计数总共的样本数
        # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
        with torch.no_grad():
            # 遍历测试数据加载器中的每个批次
            for batch_idx, data in enumerate(
                    self.dataloader, 0):
                inputs, labels = data["inputs"], data["labels"]
                outputs = self.model(inputs)
                #torch.max(outputs.data, 1)：该函数返回每行的最大值及其索引。第一个返回值是每行的最大值，而第二个返回值是每行最大值所在的索引，即模型预测的类别
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)     #统计总样本数
                correct += (predicted == labels).sum().item()  #将正确分类的样本数（即匹配的数量）累加到变量 correct 中。
        self.metrics = 100 * correct / total  # 计算测试准确率

        # 打印测试准确率
        self.logger.info(
            f'[{self.name}] Test Accuracy: {self.metrics:.3f} %')

        # 如果存在可视化器，将测试准确率记录下来
        if self.visualizer:
            self.writer.track(
                self.metrics,
                name=f"Accuracy/Test/{self.get_path()}")
        return self.metrics
