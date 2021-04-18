# this code is inspired by torch lightning docs and FSDL data module
import pytorch_lightning as pl
from torchaudio import transforms
from torch.utils.data import random_split, DataLoader
from torch import nn
import torch
import argparse
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Collection, Dict, Optional, Tuple, Union
from typing import Any, Callable, Dict, Sequence, Tuple, Union
SequenceOrTensor = Union[Sequence, torch.Tensor]


BATCH_SIZE = 256
NUM_WORKERS = 0
NUM_MFCC = 40
SAMPLE_RATE = 16000

class CH_UnicampDataModule(pl.LightningDataModule):

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_dir = self.args.get("data_dir")

        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        # self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))



        # Make sure to set the variables below in subclasses
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        # self.mapping: Collection
        # self.data_train: Union[BaseDataset, ConcatDataset]
        # self.data_val: Union[BaseDataset, ConcatDataset]
        # self.data_test: Union[BaseDataset, ConcatDataset]
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
        
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target
