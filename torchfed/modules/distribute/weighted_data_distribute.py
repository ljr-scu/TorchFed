import torch

from torchfed.modules.module import Module
from torchfed.utils.decorator import exposed


class WeightedDataDistributing(Module):
    def __init__(
            self,
            router,
            alias=None,
            visualizer=False,
            writer=None):
        super(
            WeightedDataDistributing,
            self).__init__(
            router,
            alias=alias,
            visualizer=visualizer,
            writer=writer)
        self.total_weight = 0
        self.storage = {}
        self.shared = None


    @exposed
    def upload(self, from_, weight, data):
        self.total_weight += weight
        self.storage[from_] = [weight, data]
        return True

    @exposed
    def download(self):
        return self.shared

    def update(self, data):
        self.shared = data

    #原来的
    def aggregate(self):
        # 1. read uploaded data from storage
        # 2. decompress data into gradients
        # 3. aggregate gradients
        ret = None
        if len(self.storage) == 0:
            return ret
        for data in self.storage.values():
            [weight, data] = data
            if data is None:
                continue
            if isinstance(data, dict):
                if ret is None:
                    ret = {k: v * (weight / self.total_weight)
                           for k, v in data.items()}
                else:
                    ret = {k: ret[k] + v * (weight / self.total_weight)
                           for k, v in data.items()}
            else:
                if ret is None:
                    ret = data * (weight / self.total_weight)
                else:
                    ret += data * (weight / self.total_weight)
        self.total_weight = 0
        self.storage.clear()
        return ret












