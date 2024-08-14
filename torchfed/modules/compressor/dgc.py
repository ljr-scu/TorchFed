import torch

from torchfed.modules.compressor.Compressor import Compressor

#深度梯度压缩
class DgcCompressor(Compressor):

    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        # 采样 1% 的数据以估计分位数
        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        # 计算 k 值，表示应保留的元素数
        k = max(1, int(numel * self.compress_ratio * 0.01))
        vals, indices = torch.topk(sample_tensor.abs(), k)

        # 使用采样结果计算阈值
        thr = vals.min()
        mask = tensor.abs() >= thr
        selected = mask.sum()

        # 调整阈值，使得压缩后的元素数量接近目标压缩率
        for _ in range(10):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        # 根据调整后的阈值选择压缩后的值和索引
        indices, = torch.where(mask)
        values = tensor[indices]

        tensor_compressed = values, indices
        ctx = shape, mask, numel
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        values, indices = tensor_compressed
        shape, _, numel = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)



    #compress_ratio = 0.2为例
    # 初始参数： < class 'torch.Tensor'>, tensor([-0.0003, 0.0326, 0.0258, 0.0208, -0.1314, 0.0278, 0.0365, -0.0223,
    # 0.0433, 0.0234, 0.0143, 0.0186, 0.0097, -0.0015, -0.0545, -0.0128,
    # 0.0135, 0.0456, 0.0074, -0.0904, 0.0359, 0.0322, -0.0670, -0.0632,
    # -0.0069, 0.0128, -0.0035, 0.0211, -0.0569, 0.0217, 0.0090, -0.0589])
    #
    # 压缩的参数： <class 'tuple'>, (tensor([-0.1314, -0.0545, -0.0904, -0.0670, -0.0632, -0.0569, -0.0589]), tensor([4, 14, 19, 22, 23, 28, 31]))
    #
    # ctx： <class 'tuple'>, (torch.Size([32]), tensor([False, False, False, False, True, False, False, False, False, False,
    # False, False, False, False, True, False, False, False, False, True,
    # False, False, True, True, False, False, False, False, True, False,
    # False, True]), 32)
    #
    # 解压缩后的参数： < class 'torch.Tensor'>, tensor([0.0000, 0.0000, 0.0000, 0.0000, -0.1314, 0.0000, 0.0000, 0.0000,
    # 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0545, 0.0000,
    # 0.0000, 0.0000, 0.0000, -0.0904, 0.0000, 0.0000, -0.0670, -0.0632,
    # 0.0000, 0.0000, 0.0000, 0.0000, -0.0569, 0.0000, 0.0000, -0.0589])