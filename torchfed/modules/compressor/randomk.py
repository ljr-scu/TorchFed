import torch

from torchfed.modules.compressor.Compressor import Compressor

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    indices = torch.randperm(numel, device=tensor.device)[:k]
    values = tensor[indices]
    return indices, values

# 通过选择一个随机子集来减小张量的大小
class RandomKCompressor(Compressor):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, compress_ratio):
        super().__init__()
        self.global_step = 0
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        indices, values = sparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        return [values], ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, numel, shape = ctx
        values, = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)

    # compress_ratio = 0.2
    # 初始参数： < class 'torch.Tensor'>, tensor([-0.0017, -0.0034, -0.0019, -0.0012, -0.0080, -0.0239, -0.0016, -0.0106,
    # -0.0117, -0.0060, -0.0105, 0.0020, 0.0016, -0.0029, -0.0143, -0.0020,
    # -0.0024, -0.0018, -0.0040, -0.0028, -0.0033, 0.0021, -0.0099, -0.0057,
    # -0.0065, -0.0051, -0.0022, -0.0026, 0.0060, -0.0258, -0.0044, -0.0038,
    # -0.0023, 0.0007, 0.0065, -0.0038, -0.0005, -0.0062, -0.0083, -0.0120,
    # -0.0086, -0.0052, -0.0060, -0.0042, -0.0025, 0.0028, -0.0221, -0.0142,
    # -0.0091, -0.0032, -0.0044, -0.0026, -0.0081, -0.0076, 0.0022, 0.0004,
    # -0.0096, -0.0258, -0.0012, -0.0077, -0.0051, 0.0018, -0.0142, -0.0049])
    #
    # 压缩的参数： < class 'list'>, [tensor([-0.0091, -0.0077, -0.0143, -0.0258, 0.0060, -0.0051, -0.0083, -0.0086,
    # 0.0065, -0.0239, -0.0060, -0.0020])]
    #
    # shape： <class 'tuple'>, (tensor([48, 59, 14, 57, 28, 60, 38, 40, 34, 5, 9, 15]), 64, torch.Size([64]))
    #
    # 解压缩后的参数： <class 'torch.Tensor'>, tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0239, 0.0000, 0.0000,
    # 0.0000, -0.0060, 0.0000, 0.0000, 0.0000, 0.0000, -0.0143, -0.0020,
    # 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    # 0.0000, 0.0000, 0.0000, 0.0000, 0.0060, 0.0000, 0.0000, 0.0000,
    # 0.0000, 0.0000, 0.0065, 0.0000, 0.0000, 0.0000, -0.0083, 0.0000,
    # -0.0086, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    # -0.0091, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    # 0.0000, -0.0258, 0.0000, -0.0077, -0.0051, 0.0000, 0.0000, 0.0000])