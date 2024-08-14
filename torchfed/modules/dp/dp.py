import torch
import numpy as np

class DifferentialPrivacy():
    def __init__(self, sensitivity, epsilon, delta=None):
        """
        初始化差分隐私机制类。

        参数:
        - sensitivity: 函数的敏感性。
        - epsilon: 隐私参数。
        - delta: 允许的失败概率（仅适用于高斯机制）。
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta


    def add_laplace_noise(self, tensor, batch_size):
        """
        为梯度添加拉普拉斯噪声以实现差分隐私。

        参数:
        - tensor: 需要添加噪声的梯度张量。
        - batch_size: 本地训练的批次大小。

        返回:
        - 添加拉普拉斯噪声后的梯度张量。
        """
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, tensor.size()) / batch_size
        return tensor + torch.from_numpy(noise).float()

    def add_gaussian_noise(self, tensor, batch_size):
        """
        为梯度添加高斯噪声以实现差分隐私。

        参数:
        - tensor: 需要添加噪声的梯度张量。
        - batch_size: 本地训练的批次大小。

        返回:
        - 添加高斯噪声后的梯度张量。
        """
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        noise = np.random.normal(0, sigma, tensor.size()) / batch_size
        return tensor + torch.from_numpy(noise).float()
