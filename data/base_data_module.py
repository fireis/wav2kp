# this code is inspired by torch lightning docs and FSDL data module
import pytorch_lightning as pl
from torchaudio import transforms
from torch.utils.data import random_split, DataLoader
from torch import nn
import argparse
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Collection, Dict, Optional, Tuple, Union


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
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        self.n_mfcc = self.args.get("num_mfcc", NUM_MFCC)
        self.sample_rate = self.args.get("sr", SAMPLE_RATE)

        self.transform = transforms.MFCC(sample_rate= self.sample_rate, 
                                         n_mfcc=self.n_mfcc)

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
        self.data_train = 
