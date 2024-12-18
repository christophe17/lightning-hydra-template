from collections import OrderedDict
from typing import Dict, List, Optional, Union

import hydra
import torch

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.datamodules.components.audio_transforms import AudioTransformsWrapper


class DaicDataModule(LightningDataModule):
    """Example of LightningDataModule for single dataset.

    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def predict_dataloader(self):
            # return predict dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        """DataModule with standalone train, val and test dataloaders.

        Args:
            datasets (DictConfig): Datasets config.
            loaders (DictConfig): Loaders config.
            transforms (DictConfig): Transforms config.
        """

        super().__init__()
        self.cfg_datasets = datasets
        self.cfg_loaders = loaders
        self.transforms = transforms
        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.predict_set: Dict[str, Dataset] = OrderedDict()

    def _get_dataset_(
        self, split_name: str, dataset_name: Optional[str] = None
    ) -> Dataset:
        transforms = AudioTransformsWrapper(self.transforms.get(split_name))
        cfg = self.cfg_datasets.get(split_name)
        if dataset_name:
            cfg = cfg.get(dataset_name)
        dataset: Dataset = hydra.utils.instantiate(cfg, transforms=transforms)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`, `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """

        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:

            transforms = AudioTransformsWrapper(self.transforms)
            cfg = self.cfg_datasets

            dataset: Dataset = hydra.utils.instantiate(cfg, transforms=transforms)
            seed = self.cfg_datasets.get("seed")

            self.train_set, self.valid_set, self.test_set = random_split(
                dataset=dataset,
                lengths=self.cfg_datasets.get("train_val_test_split"),
                generator=torch.Generator().manual_seed(seed),
            )

        # load predict dataset only if test set existed already
        if (stage == "predict") and self.test_set:
            self.predict_set = {"PredictDataset": self.test_set}

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, **self.cfg_loaders.get("train"))

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_set, **self.cfg_loaders.get("valid"))

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.cfg_loaders.get("test"))

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        loaders = []
        for _, dataset in self.predict_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get("predict"))
            )
        return loaders

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

