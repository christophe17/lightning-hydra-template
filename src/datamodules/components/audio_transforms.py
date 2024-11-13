from typing import Any

import albumentations
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image


class AudioTransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """

        augmentations = []
        if not transforms_cfg.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for augmentation_name in transforms_cfg.get("order"):
            augmentation = hydra.utils.instantiate(
                transforms_cfg.get(augmentation_name), _convert_="object"
            )
            augmentations.append(augmentation)
        self.augmentations = albumentations.Compose(augmentations)

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        """Apply TransformsWrapper module.

        Args:
            image (Any): Input image.
            kwargs (Any): Additional arguments.

        Returns:
            Any: Transformation results.
        """

        if isinstance(image, Image.Image):
            image = np.asarray(image)
        return self.augmentations(image=image, **kwargs)



import torch
import torchaudio
import torchaudio.transforms as T
import random

class Normalize(torch.nn.Module):
    def forward(self, waveform):
        return waveform / waveform.abs().max()

class RandomResample(torch.nn.Module):
    def __init__(self, min_freq, max_freq):
        super().__init__()
        self.min_freq = min_freq
        self.max_freq = max_freq

    def forward(self, waveform):
        orig_freq = waveform.shape[1]  # assuming input is [channel, time]
        new_freq = random.randint(self.min_freq, self.max_freq)
        resampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
        return resampler(waveform)

class TimeStretch(torch.nn.Module):
    def __init__(self, min_rate=0.8, max_rate=1.2):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate

    def forward(self, waveform):
        rate = random.uniform(self.min_rate, self.max_rate)
        return T.TimeStretch()(waveform, rate)

class AddNoise(torch.nn.Module):
    def __init__(self, noise_factor=0.005):
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, waveform):
        noise = self.noise_factor * torch.randn_like(waveform)
        return waveform + noise


# Create the transformation pipeline using nn.Sequential
pipeline = torch.nn.Sequential(
    Normalize(),
    RandomResample(min_freq=8000, max_freq=16000),
    TimeStretch(min_rate=0.8, max_rate=1.2),
    AddNoise(noise_factor=0.01)
)

# Load the audio file
waveform, sample_rate = torchaudio.load(file_path)

# Apply the pipeline
processed_waveform = pipeline(waveform)