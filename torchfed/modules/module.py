import aim
import time
import types


from torchfed.routers.router_msg import RouterMsg, RouterMsgResponse
from typing import TypeVar, Type
from torchfed.logging import get_logger
from torchfed.types.meta import PostInitCaller

from prettytable import PrettyTable

from torchfed.utils.hash import hex_hash
from torchfed.utils.helper import interface_join

T = TypeVar('T')


class Module(metaclass=PostInitCaller):
    def __init__(
            self,
            router,
            alias=None,
            visualizer=False,
            writer=None,
            override_hparams=None):
        self.ident = hex_hash(f"{time.time_ns()}")
        self.router = router
        self.alias = alias
        self.name = self.get_node_name()
        self.logger = get_logger(router.exp_id, self.get_root_name())
        self.override_hparams = override_hparams
        self.hparams = None
        self.released = False

        self.visualizer = visualizer
        self.writer = writer
        if self.visualizer:
            #writer 是一个写入器对象，通常用于记录和可视化实验的指标
            if writer is None:
                #使用日志记录器（logger）输出一条信息，提示用户启动可视化工具（在这里是 aim up）。
                self.logger.info(
                    f"[{self.name}] Visualizer enabled. Run `aim up` to start.")
                self.writer = self.get_visualizer()

        self.data_sent, self.data_received = 0, 0

        self.routing_table = {}
        if self.is_root():
            router.register(self)

        if self.is_root():
            self.hparams = self.get_default_hparams()

            #检查是否提供了要覆盖的超参数
            if self.override_hparams is not None and isinstance(
                    self.override_hparams, dict):
                for key, value in self.override_hparams.items():
                    self.hparams[key] = value

            self.hparams["name"] = self.name
            self.hparams["visualizer"] = self.visualizer
            hp_table = PrettyTable()
            hp_table.field_names = self.hparams.keys()
            hp_table.add_row(self.hparams.values())
            self.logger.info(f"[{self.name}] Hyper-parameters:")
            for row in hp_table.get_string().split("\n"):
                self.logger.info(row)
            if self.visualizer:
                self.writer['hparams'] = self.hparams

        # 定义消息路径与处理函数的映射关系
        self.path_to_method = {
            "distributor/upload": self.handle_upload
        }

    def __post__init__(self):
        pass

    def get_default_hparams(self):
        return {}

    def get_metrics(self):
        return None

    def register_submodule(self, module: Type[T], name, router, *args) -> T:   #注册子模块
        submodule_name = f"{self.name}/{name}"   #子模块名称
        if submodule_name in self.routing_table:    #检查子模块名称是否已经存在于路由表
            self.logger.error("Cannot register modules with the same name")
            raise Exception("Cannot register modules with the same name")
        #实例化传入的模块类型 module，并传入相应的参数
        module_obj = module(
            router,
            *args,
            alias=submodule_name,
            visualizer=self.visualizer,
            writer=self.writer)
        #将子模块对象添加到路由表中，使用子模块的名称 name 作为键。
        self.routing_table[name] = module_obj
        return module_obj

    def send(self, to, path, args):
        path = path.__name__ if callable(path) else path
        if not isinstance(to, list):
            to = [to]
        router_msgs = []
        for t in to:
            # 创建一个 RouterMsg 对象，表示要发送的消息
            router_msg = RouterMsg(from_=self.name, to=t, path=path, args=args)
            # 累加发送的数据量
            self.data_sent += router_msg.size
            # 在可视化工具中跟踪发送的数据量
            if self.visualizer:
                self.writer.track(self.data_sent, name="Data Sent (bytes)")
            # 将消息添加到要发送的消息列表中
            router_msgs.append(router_msg)
        # 使用路由器向目标广播消息，并接收响应
        responses = self.router.broadcast(router_msgs)
        # 如果没有接收到任何响应，输出警告信息
        if len(responses) == 0:
            self.logger.warning(f"No response received for msgs {router_msgs}")
        resp_size = 0
        # 遍历响应列表，累加响应的大小
        for response in responses:
            resp_size += response.size
            # print(f"Response data: {response.data}")
        # 累加接收到的数据量
        self.data_received += resp_size
        # 在可视化工具中跟踪接收到的数据量
        if self.visualizer:
            self.writer.track(self.data_received, name="Data Received (bytes)")
        return responses

    def receive(self, router_msg: RouterMsg) -> RouterMsgResponse:
        self.logger.debug(
            f"Module {self.name} receiving data {router_msg}")
        self.logger.debug(
            f"Module {self.name} is calling path {router_msg.path} with args {router_msg.args}")
        # 累加接收到的数据量
        self.data_received += router_msg.size
        # 在可视化工具中跟踪接收到的数据量
        if self.visualizer:
            self.writer.track(self.data_received, name="Data Received (bytes)")

        # #新加的
        # # 处理消息
        # path = router_msg.path
        # method = self.path_to_method.get(path)
        #
        # if method:
        #     method(router_msg.args)  # 调用处理函数
        # else:
        #     self.logger.warning(f"Unknown path {path}")
        # ##到这里

        # infinite loop until there is some return value
        # 无限循环，直到有返回值
        while True:
            try:
                # 调用 entry 方法执行指定路径的操作，并构造 RouterMsgResponse 对象
                ret = RouterMsgResponse(
                    from_=self.name,
                    to=router_msg.from_,
                    data=self.entry(
                        router_msg.path,
                        router_msg.args))
                break
            except Exception as e:
                # 捕获异常，输出警告信息，并等待1秒后再次尝试
                self.logger.warning(
                    f"Error in {self.name} when calling {router_msg.path} from {router_msg.from_} "
                    f"with args {router_msg.args}: {e}")
                self.logger.warning(f"Will try again in 1 second")
                time.sleep(1)

        # 累加发送的数据量
        self.data_sent += ret.size
        # 在可视化工具中跟踪发送的数据量
        if self.visualizer:
            self.writer.track(self.data_sent, name="Data Sent (bytes)")
        # 如果返回的数据为空，输出警告信息
        if ret.data is None:
            self.logger.warning(
                f"Module {self.name} does not have path {router_msg.path}")
        # 输出调试信息，表示模块的响应
        self.logger.debug(f"Module {self.name} responses with data {ret}")
        return ret

    # 用于模块内部的调用，根据给定的路径调用对应的方法或属性，并返回结果。
    def entry(self, path, args, check_exposed=True):
        #判断 path 是否是 types.MethodType 类型的实例，即 path 是否是一个方法。如果 path 是方法，这个条件判断方法的 __self__ 属性是否是 Module 类型的实例，即这个方法是否属于一个模块对象。
        if isinstance(
                path,
                types.MethodType) and isinstance(
                path.__self__,
                Module):
            # 将模块中的方法路径进行格式化，去除了模块的根模块部分，留下了相对路径。
            path = f"{'/'.join(path.__self__.name.split('/')[1:])}/{path.__name__}"

        # 将路径按 "/" 分割成列表
        paths = path.split("/")
        # 获取路径的第一个元素作为目标
        target = paths.pop(0)

        # 如果目标在路由表中，递归调用目标模块的 entry 方法
        if target in self.routing_table:
            return self.routing_table[target].entry(
                "/".join(paths), args, check_exposed=check_exposed)
        # 如果目标是当前模块的属性，并且是可调用的
        elif hasattr(self, target):
            entrance = getattr(self, target)
            if callable(entrance) and (
                    not check_exposed or (
                    hasattr(
                        entrance,
                        "exposed") and entrance.exposed)):
                # 调用可调用的属性，并返回结果
                return entrance(*args)

    def release(self):
        # 如果已经释放过资源，直接返回
        if self.released:
            return
        # 递归调用路由表中所有模块的 release 方法
        for module in self.routing_table.values():
            module.release()
        # 如果可视化工具被启用，关闭可视化工具的写入操作
        if self.visualizer:
            self.writer.close()
        # 标记模块已释放资源
        self.released = True
        # 输出日志信息，表示模块已终止
        self.logger.info(f"[{self.name}] Terminated")

    def get_node_name(self):
        if self.alias is not None:
            return self.alias
        return f"{self.__class__.__name__}_{self.ident[:4]}_{self.router.exp_id[:4]}"

    def get_visualizer(self):
        return aim.Run(
            run_hash=self.get_node_name(),
            experiment=self.router.exp_id
        )

    def is_root(self):     #这个方法的目的是通过检查对象的名称字符串中是否包含字符 "/" 来判断对象是否被标记为根节点
        return "/" not in self.name

    def get_root_name(self):
        return self.name.split("/")[0]

    def get_path(self):
        return "/".join(self.name.split("/")[1:])

    #新加的
    def handle_upload(self, args):
        # 处理上传请求的逻辑
        self.received_compressed_data = args[2]
        # print(self.received_compressed_data)
        # self.logger.info(f"Received compressed data: {self.received_compressed_data}")
