"""Utilities for handling the MNIST Dataset."""
import jax.numpy as jnp
import torch
from jax.tree_util import tree_map
from torch.utils import data
from torchvision import datasets, transforms


def numpy_collate(batch):
    """Convert tensors to numpy arrays."""
    return tree_map(jnp.asarray, data.default_collate(batch))

def mnist_collate(batch):
    """Convert tensors to numpy arrays."""
    x, y = data.default_collate(batch)
    return jnp.asarray(x).transpose(0, 2, 3, 1), jnp.asarray(y)


class NumpyLoader(data.DataLoader):
    """DataLoader subclass that converts tensors to numpy arrays."""

    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: data.Sampler = None,
        batch_sampler: data.BatchSampler = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn=None,
    ):
        """Initialize the NumpyLoader."""
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=mnist_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class MNISTGenerator:
    """Utility class for handling the MNIST dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = '../data',
        seed: int = 2,
        class_subset: list = None,
        max_train_size: int = None,
    ):
        """Initialize the MNIST dataset generator."""
        self.seed = seed
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.class_subset = class_subset
        self.max_train_size = max_train_size
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self._create_np_data_loaders()

    def _create_np_data_loaders(self):
        """Create data loaders for the MNIST dataset."""
        # Download MNIST dataset if not already downloaded
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        # Split train dataset into train and validation sets
        num_train = len(train_dataset)
        len_val = int(num_train * 0.2)
        fold = self.seed % 5
        indices = list(range(num_train))
        val_indices = list(range(fold*len_val, (fold+1)*len_val))
        train_indices = list(set(indices) - set(val_indices))
        # train_indices, val_indices = indices[:split], indices[split:]
        val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        if self.class_subset is not None:
            train_dataset = torch.utils.data.Subset(
                train_dataset,
                [
                    i
                    for i in range(len(train_dataset))
                    if train_dataset[i][1] in self.class_subset
                ],
            )
            val_dataset = torch.utils.data.Subset(
                val_dataset,
                [
                    i
                    for i in range(len(val_dataset))
                    if val_dataset[i][1] in self.class_subset
                ],
            )
            test_dataset = torch.utils.data.Subset(
                test_dataset,
                [
                    i
                    for i in range(len(test_dataset))
                    if test_dataset[i][1] in self.class_subset
                ],
            )
        if self.max_train_size is not None:
            train_dataset = torch.utils.data.Subset(
                train_dataset, list(range(self.max_train_size))
            )

        train_loader = NumpyLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = NumpyLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = NumpyLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_loader, val_loader, test_loader
