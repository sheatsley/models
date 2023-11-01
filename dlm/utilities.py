"""
This module defines various utility functions designed to simplify or
accelerate common tasks in deep learning.
"""

import torch


class DataLoader:
    """
    This class is designed to be a simplier version of the PyTorch DataLoader
    class and (in most cases) can be used as a drop-in replacement. The purpose
    of this class is to accelerate data iteration, given that the dlm module
    assumes that: (1) all data is already loaded into memory, and (2) the data
    will always be tensors (thereby preventing data fetching to be driven by
    list comprehensions, which is the principal bottleneck in the PyTorch
    DataLoader class, especially with large batch sizes).


    :func:`__init__`: instantiates DataLoader objects
    :func:`__iter__`: returns an iterator over the dataset
    :func:`__len__`: returns the number of batches in the dataset
    :func:`__next__`: returns the next batch in the dataset
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        """
        This method instantiates DataLoader objects with the dataset to be
        iterated over.

        :param dataset: the dataset to be iterated over
        :type dataset: torch.utils.data.dataset Dataset-inherited object
        :param batch_size: the number of samples in each batch
        :type batch_size: int
        :param shuffle: whether or not to shuffle the dataset
        :type shuffle: bool
        :return: a data loader
        :rtype: DataLoader object
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        return None

    def __iter__(self):
        """
        This method intializes the iterator object. It initializes tracking the
        current dataset index and shuffles the indicies to iterate over, if
        desired.

        :return: an initialized iterator
        :rtype: DataLoader object
        """
        self.index = 0
        sampler = torch.randperm if self.shuffle else torch.arange
        self.indices = sampler(len(self.dataset))
        return self

    def __len__(self):
        """
        This method returns the number of batches in the dataset.

        :return: the number of batches in the dataset
        :rtype: int
        """
        return -((-len(self.dataset)) // self.batch_size)

    def __next__(self):
        """
        This method returns the current batch and increments the dataset index.
        :return: the current batch
        :rtype: tuple of torch Tensor objects (batch_size, m)  & (batch_size,)
        """
        if self.index >= len(self.dataset):
            raise StopIteration
        else:
            indices = self.indices[self.index : self.index + self.batch_size]
            self.index += self.batch_size
            return self.dataset[indices]
