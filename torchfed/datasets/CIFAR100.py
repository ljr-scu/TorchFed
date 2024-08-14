import sys
import os.path
import random
from typing import Optional, Callable, List

import numpy as np

import torch
import torchvision
from tqdm import trange, tqdm

from torchfed.datasets import TorchDataset, TorchUserDataset, TorchGlobalDataset

class TorchCIFAR100(TorchDataset):
    num_classes = 100

    def __init__(
            self,
            root: str,
            num_users: int,
            num_labels_for_users: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            rebuild: bool = False,
            cache_salt: int = 0,
    ) -> None:
        super().__init__(
            root,
            TorchCIFAR100.num_classes,
            num_users,
            num_labels_for_users,
            transform,
            target_transform,
            download,
            rebuild,
            cache_salt
        )

    @property
    def name(self) -> str:
        return "CIFAR100"

    def load_user_dataset(self) -> List[List[TorchUserDataset]]:
        train_dataset = torchvision.datasets.CIFAR100(
            self.root, True, self.transform, self.target_transform, self.download)
        test_dataset = torchvision.datasets.CIFAR100(
            self.root, False, self.transform, self.target_transform, self.download)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False)

        tensor_train_dataset, tensor_test_dataset = {}, {}
        tensor_train_dataset["data"], tensor_train_dataset["targets"] = next(iter(train_dataloader))
        tensor_test_dataset["data"], tensor_test_dataset["targets"] = next(iter(test_dataloader))

        inputs = np.concatenate((tensor_train_dataset["data"], tensor_test_dataset["data"]), axis=0)
        labels = np.concatenate((tensor_train_dataset["targets"], tensor_test_dataset["targets"]), axis=0)

        split_inputs = [[] for _ in range(self.num_classes)]
        for i in range(len(inputs)):
            split_inputs[labels[i]].append(inputs[i])

        num_channels, num_height, num_width = inputs.shape[1], inputs.shape[2], inputs.shape[3]

        user_x = [[] for _ in range(self.num_users)]
        user_y = [[] for _ in range(self.num_users)]
        idx = np.zeros(self.num_classes, dtype=np.int64)

        for user_idx in trange(self.num_users):
            for label_idx in range(self.num_labels_for_users):
                assigned_label = (user_idx + label_idx) % self.num_classes
                user_x[user_idx] += split_inputs[assigned_label][idx[assigned_label]: idx[assigned_label] + 100]
                user_y[user_idx] += [assigned_label] * 100
                idx[assigned_label] += 100

        props = np.random.lognormal(0, 2., (100, self.num_users, self.num_labels_for_users))
        props = props * (np.sum(np.sum(props, axis=2), axis=1) / (100 * self.num_labels_for_users))[:, np.newaxis,
                        np.newaxis]

        for user_idx in trange(self.num_users):
            for label_idx in range(self.num_labels_for_users):
                assigned_label = (user_idx + label_idx) % self.num_classes
                num_samples = int(props[assigned_label, user_idx, label_idx])
                # num_samples = int(props[assigned_label, user_idx // int(self.num_users / 100), label_idx])
                num_samples += random.randint(300, 600)
                if self.num_users <= 20:
                    num_samples *= 2
                if idx[assigned_label] + num_samples < len(split_inputs[assigned_label]):
                    user_x[user_idx] += split_inputs[assigned_label][
                                        idx[assigned_label]: idx[assigned_label] + num_samples]
                    user_y[user_idx] += [assigned_label] * num_samples
                    idx[assigned_label] += num_samples

            # user_x[user_idx] = torch.Tensor(user_x[user_idx]).view(-1, num_channels, num_height, num_width).type(
            #     torch.float32)
            user_x[user_idx] = torch.tensor(np.array(user_x[user_idx])).view(-1, num_channels, num_height,
                                                                             num_width).type(torch.float32)

            user_y[user_idx] = torch.Tensor(user_y[user_idx]).type(torch.int64)

        user_dataset = []
        for user_idx in trange(self.num_users):
            combined = list(zip(user_x[user_idx], user_y[user_idx]))
            random.shuffle(combined)
            user_x[user_idx], user_y[user_idx] = zip(*combined)

            num_samples = len(user_x[user_idx])
            train_len = int(num_samples * 0.75)
            train_user_data = TorchUserDataset(
                user_idx,
                user_x[user_idx][:train_len],
                user_y[user_idx][:train_len],
                self.num_classes
            )
            test_user_data = TorchUserDataset(
                user_idx,
                user_x[user_idx][train_len:],
                user_y[user_idx][train_len:],
                self.num_classes
            )
            user_dataset.append([train_user_data, test_user_data])

        return user_dataset

    def load_global_dataset(self) -> List[TorchGlobalDataset]:
        train_dataset = torchvision.datasets.CIFAR100(
            self.root, True, self.transform, self.target_transform, self.download)
        test_dataset = torchvision.datasets.CIFAR100(
            self.root, False, self.transform, self.target_transform, self.download)
        return [
            TorchGlobalDataset(train_dataset, self.num_classes),
            TorchGlobalDataset(test_dataset, self.num_classes)
        ]