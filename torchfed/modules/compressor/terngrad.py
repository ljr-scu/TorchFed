
import torch

from torchfed.modules.compressor.Compressor import Compressor

#TernGradCompressor 通过将梯度值量化为三值（-1, 0, 1），并保存一个标量来实现梯度的压缩和解压缩。
class TernGradCompressor(Compressor):

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()

        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)


# 初始参数：<class 'torch.Tensor'>,tensor([ 0.0049,  0.0027, -0.0177,  0.0001, -0.0030, -0.0040, -0.0035, -0.0022,
#         -0.0774,  0.0078,  0.0078, -0.0044, -0.0026,  0.0019,  0.0040,  0.0010,
#          0.0071, -0.0077, -0.0061,  0.0140, -0.0204, -0.0046, -0.0063,  0.0116,
#         -0.0060,  0.0095, -0.0426,  0.0009, -0.0062,  0.0072, -0.0050, -0.0077,
#          0.0108,  0.0065, -0.0009,  0.0113,  0.0018,  0.0085, -0.0740, -0.0081,
#         -0.0107,  0.0003,  0.0552,  0.0019, -0.0026,  0.0045, -0.0184, -0.0120,
#         -0.0046, -0.0036,  0.0137, -0.0072,  0.0081,  0.0198, -0.0588, -0.0048,
#          0.0494, -0.0064, -0.0054, -0.0106, -0.0076,  0.0067,  0.0006, -0.0072])
# 压缩的参数：<class 'tuple'>,(tensor([ 0,  0,  0,  0,  0,  0, -1,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
#         -1,  1, -1, -1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0, -1, -1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,
#         -1,  0,  1,  0,  0,  0, -1,  0,  0, -1], dtype=torch.int8), tensor([0.0500]))
# shape：<class 'torch.Size'>,torch.Size([64])
# 解压缩后的参数：<class 'torch.Tensor'>,tensor([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0500,  0.0000,
#         -0.0500,  0.0500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
#          0.0000,  0.0000, -0.0500,  0.0500, -0.0500, -0.0500,  0.0000,  0.0000,
#          0.0000,  0.0000, -0.0500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
#          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0500, -0.0500,
#          0.0000,  0.0000,  0.0500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
#          0.0000,  0.0000,  0.0000, -0.0500,  0.0000,  0.0000, -0.0500,  0.0000,
#          0.0500,  0.0000,  0.0000,  0.0000, -0.0500,  0.0000,  0.0000, -0.0500])