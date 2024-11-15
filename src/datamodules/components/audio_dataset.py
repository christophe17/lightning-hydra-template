import io
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio


class BaseAudioDataset(Dataset):
    def __init__(
        self,
        transforms: Optional[Callable] = None,
    ) -> None:
        """BaseDataset.

        Args:
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
        """
        self.transforms = transforms

    def _read_audio_(self, audio: Any) -> torch.Tensor:
        """Read audio from source.

        Args:
            audio (Any): Audio source. Could be str, Path or bytes.

        Returns:
            torch.Tensory: Loaded audio
            int: Sample rate
        """

        if isinstance(audio, (str, Path)):
            signal, sr = torchaudio.load(audio)
        else:
            signal, sr = torchaudio.load(io.BytesIO(audio))
        
        return signal


    def _process_audio_(self, signal: torch.Tensor) -> torch.Tensor:
        """Process audio, including transforms, etc.

        Args:
            image (np.ndarray): Image in np.ndarray format.

        Returns:
            torch.Tensor: Image prepared for dataloader.
        """

        if self.transforms:
            signal = self.transforms(signal)
        return signal

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
