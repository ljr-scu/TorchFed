import time
from typing import List

import aim
import matplotlib.pyplot as plt
from .router_msg import RouterMsg, RouterMsgResponse
from torchfed.logging import get_logger
from torchfed.utils.hash import hex_hash
from torchfed.utils.plotter import NetworkConnectionsPlotter, DataTransmitted


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, mode="singleton", ident=None, **kwargs):
        if mode == "singleton":
            if cls not in cls._instances:
                cls._instances[cls] = super(
                    Singleton, cls).__call__(
                    *args, **kwargs)
            return cls._instances[cls]
        elif mode == "simulate":
            if ident is None:
                raise ValueError(
                    "ident must be provided when mode is simulate")
            if ident not in cls._instances:
                cls._instances[ident] = super(
                    Singleton, cls).__call__(
                    *args, **kwargs)
            return cls._instances[ident]



class Router(metaclass=Singleton):      #Router 是路由器的基类，定义了通讯数据的收发转发流程，但是没有定义具体的收发方式
    context = None

    def __init__(
            self,
            alias=None,
            visualizer=False):
        Router.context = self
        self.ident = hex_hash(f"{time.time_ns()}")
        self.exp_id = hex_hash(f"{time.time_ns()}")
        self.alias = alias
        self.name = self.get_router_name()
        self.visualizer = visualizer
        self.logger = get_logger(self.exp_id, self.name)
        self.released = False

        if self.visualizer:
            self.logger.info(
                f"[{self.name}] Visualizer enabled. Run `aim up` to start.")
            self.writer = self.get_visualizer()

        self.owned_nodes = {}
        self.peers_table = {}

        self.network_plotter = NetworkConnectionsPlotter()
        self.data_transmitted = DataTransmitted()

        self.logger.info(
            f"[{self.name}] Initialized completed. Router ID: {self.ident}. Experiment ID: {self.exp_id}")

    def register(self, module):     #将模块注册到注册系统中。如果模块的名字在系统中不存在，它将被加入到系统的字典中
        if module.name not in self.owned_nodes.keys():
            self.owned_nodes[module.name] = module.receive

    def unregister(self, worker):
        if worker.name in self.owned_nodes.keys():
            del self.owned_nodes[worker.name]

    def connect(self, module, peers: list):  #定义网络拓扑结构
        # 如果模块不是根模块，直接返回
        if not module.is_root():
            return
        # 将所有连接的模块名称转换为根模块的名称
        peers = [self.get_root_name(peer) for peer in peers]
        # 如果模块在 peers_table 中已经存在，更新其连接的对等模块列表
        if hasattr(self.peers_table, module.name):
            for peer in peers:
                if peer not in self.peers_table[module.name]:
                    self.peers_table[module.name].append(peer)
        # 如果模块在 peers_table 中不存在，创建新的记录
        else:
            self.peers_table[module.name] = peers
        # 在网络图中添加模块之间的边
        for peer in peers:
            self.network_plotter.add_edge(module.name, peer)
        # 如果可视化工具被启用，更新网络图并显示
        if self.visualizer:
            fig = self.network_plotter.get_figure()
            self.writer.track(aim.Figure(fig), name="Network Graph")

    def disconnect(self, module, peers: list):
        if not module.is_root():
            return
        # 将所有断开连接的模块名称转换为根模块的名称
        peers = [self.get_root_name(peer) for peer in peers]
        # 如果模块在 peers_table 中存在，更新其连接的对等模块列表
        if hasattr(self.peers_table, module.name):
            for peer in peers:
                if peer in self.peers_table[module.name]:
                    self.peers_table[module.name].remove(peer)
                    self.network_plotter.remove_edge(module.name, peer)
        # 如果可视化工具被启用，更新网络图并显示
        if self.visualizer:
            fig = self.network_plotter.get_figure()
            self.writer.track(aim.Figure(fig), name="Network Graph")

    def get_peers(self, module):  #获取模块在网络中连接的其他对等模块的列表。
        # 获取给定模块的根模块名称
        name = module.get_root_name()
        # 返回根模块在 peers_table 中记录的对等模块列表
        return self.peers_table[name]

    #用于在路由器之间广播消息，并记录了消息的传输信息
    def broadcast(
            self,
            router_msg: List[RouterMsg]) -> List[RouterMsgResponse]:
        # 遍历要广播的消息列表
        for msg in router_msg:
            self.logger.debug(
                f"[{self.name}] broadcasting message {msg}")
            # 记录消息的传输信息，包括发送方、接收方和消息大小
            self.data_transmitted.add(
                self.get_root_name(msg.from_),
                self.get_root_name(msg.to),
                msg.size
            )
        # 调用实际的广播方法（由子类具体实现）
        return self.impl_broadcast(router_msg)

    def impl_broadcast(
            self,
            router_msg: List[RouterMsg]) -> List[RouterMsgResponse]:
        # 抛出 NotImplementedError 异常，提示子类需要实现该方法
        raise NotImplementedError

    @staticmethod
    def receive(router_msg: List[RouterMsg]):
        responses = []
        # 遍历接收到的消息列表
        for msg in router_msg:
            # 如果消息的接收方不在 Router.context.owned_nodes.keys() 中，跳过处理
            if msg.to not in Router.context.owned_nodes.keys():
                continue
            # 记录接收到的消息信息，包括消息的发送方、接收方和大小
            Router.context.logger.debug(
                f"[{Router.context.name}] receiving message {msg}")

            Router.context.data_transmitted.add(
                Router.get_root_name(msg.from_),
                Router.get_root_name(msg.to),
                msg.size
            )
            # 调用接收方模块的处理方法，并记录处理后的响应消息信息
            resp_msg = Router.context.owned_nodes[msg.to](msg)
            # 记录响应消息的传输信息，包括响应消息的发送方、接收方和大小
            Router.context.data_transmitted.add(
                Router.get_root_name(resp_msg.from_),
                Router.get_root_name(resp_msg.to),
                resp_msg.size
            )
            # 将响应消息添加到响应列表中
            responses.append(resp_msg)
        return responses

    @staticmethod
    def get_root_name(name):
        return name.split("/")[0]

    def get_visualizer(self):
        return aim.Run(
            run_hash=self.name,
            experiment=self.exp_id
        )

    def refresh_exp_id(self):
        self.exp_id = hex_hash(f"{time.time_ns()}")
        self.logger.info(
            f"[{self.name}] Experiment ID refreshed: {self.exp_id}")

    def release(self):

        if self.released:
            return
        # 调用实际的释放资源方法
        self.impl_release()
        # 记录数据传输矩阵的信息，包括发送方、接收方和传输大小
        self.logger.info(f"[{self.name}] Data transmission matrix:")
        for row in self.data_transmitted.get_transmission_matrix_str().split("\n"):
            self.logger.info(row)
        total_transmission = self.data_transmitted.get_total_data_transmitted()
        self.logger.info(f"Total transmission:{total_transmission}")
        # 如果启用了可视化，则绘制数据传输图并关闭可视化工具
        if self.visualizer:
            fig = self.data_transmitted.get_figure()
            self.writer.track(aim.Figure(fig), name="Data Transmission")
            self.writer.close()
        # 标记资源已释放
        self.released = True
        # 记录模块终止信息
        self.logger.info(f"[{self.name}] Terminated")

    def impl_release(self):
        raise NotImplementedError

    def get_router_name(self):
        if self.alias is not None:
            return self.alias
        return f"{self.__class__.__name__}_{self.ident[:4]}_{self.exp_id[:4]}"

    def get_Total_datatransmission(self):
        total_transmission = self.data_transmitted.get_total_data_transmitted()
        self.logger.info(f"Total transmission:{total_transmission}")
        return total_transmission




