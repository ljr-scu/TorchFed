import os
from typing import List

import torch
import torch.distributed.rpc as rpc

from .router import Router
from .router_msg import RouterMsg, RouterMsgResponse


class TorchDistributedRPCRouter(Router):   #基于router类写的用rpc进行收发的路由器
    def __init__(        #rank——表示当前进程编号，默认0为master进程，world_size——表示总进程数
            self,
            rank,
            world_size,
            backend=None,
            rpc_backend_options=None,
            alias=None,
            visualizer=False):
        super().__init__(alias=alias, visualizer=visualizer)    #调用父类__int__()
        self.rank = rank
        self.world_size = world_size
        if backend is None:
            backend = rpc.BackendType.TENSORPIPE

        if rpc_backend_options is None:
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                init_method="env://",
                rpc_timeout=0
            )
        torch.distributed.rpc.init_rpc(
            self.name, backend, rank, world_size, rpc_backend_options)
        self.received_messages = {}

    def impl_broadcast(
            self,
            router_msg: List[RouterMsg]) -> List[RouterMsgResponse]:
        futs, rets = [], []
        for rank in range(self.world_size):
            futs.append(
                rpc.rpc_async(
                    rank,
                    Router.receive,
                    args=(
                        router_msg,
                    )))
        for fut in futs:
            resp = fut.wait()
            if resp is not None:
                rets.extend(fut.wait())

        for ret in rets:
            self.data_transmitted.add(
                self.get_root_name(ret.from_),
                self.get_root_name(ret.to),
                ret.size
            )
        return rets

    def impl_release(self):
        rpc.shutdown()

    #新加的
    def get_last_received_msg(self, peer_id):
        # 获取指定对等节点最近接收到的消息
        if peer_id in self.received_messages and self.received_messages[peer_id]:
            return self.received_messages[peer_id][-1]  # 返回最近的消息
        else:
            raise ValueError(f"No messages received from peer {peer_id}")

    def receive_message(self, peer_id, message):
        # 接收消息并存储
        if peer_id not in self.received_messages:
            self.received_messages[peer_id] = []
        self.received_messages[peer_id].append(message)

