import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.datamodules.components.audio_dataset import BaseAudioDataset
from src.datamodules.components.h5_file import H5PyFile
from src.datamodules.components.parse import parse_image_paths


class DaicDataset(BaseAudioDataset):
    def __init__(
        self,
        json_path: Optional[str] = None,
        data_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        include_names: bool = False,
        **kwargs: Any,
    ) -> None:
        """ClassificationDataset.

        Args:
            json_path (:obj:`str`, optional): Path to annotation json.
            txt_path (:obj:`str`, optional): Path to annotation txt.
            data_path (:obj:`str`, optional): Path to HDF5 file or images source dir.
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
            include_names (bool): If True, then `__getitem__` method would return image
                name/path value with key `name`. Default to False.
            shuffle_on_load (bool): Deterministically shuffle the dataset on load
                to avoid the case when Dataset slice contains only one class due to
                annotations dict keys order. Default to True.
            label_type (str): Label torch.tensor type. Default to torch.FloatTensor.
            kwargs (Any): Additional keyword arguments for H5PyFile class.
        """

        super().__init__(transforms)
        if not json_path:
            raise ValueError("Requires json_path.")
        else:
            json_path = Path(json_path)
            if not json_path.is_file():
                raise RuntimeError(f"'{json_path}' must be a file.")
            with open(json_path) as json_file:
                self.annotation = json.load(json_file)
        

        self.keys = list(self.annotation)
        self.include_names = include_names
        
        data_path = "" if data_path is None else data_path
        self.data_path = data_path = Path(data_path)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        
        source = self.data_path / key
        source = str(source) + "_AUDIO.wav"
        signal = self._read_audio_(source)
        signal = self._process_audio_(signal)
        label = torch.tensor(self.annotation[key])
        if self.include_names:
            return {"audio": signal, "label": label, "name": key}
        return {"audio": signal, "label": label}

    def get_weights(self) -> List[float]:
        label_list = [self.annotation[key] for key in self.keys]
        weights = 1.0 / np.bincount(label_list)
        return weights.tolist()
