from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torchaudio
import torchaudio.transforms as T
import random



class AudioTransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """

        transformations = []
        if not transforms_cfg.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of transforms as List[transform name]"
            )
        for transform_name in transforms_cfg.get("order"):
            transform = hydra.utils.instantiate(
                transforms_cfg.get(transform_name), _convert_="object"
            )
            transformations.append(transform)

        
        self.transformations = torch.nn.Sequential(*transformations)

    def __call__(self, signal: Any, **kwargs: Any) -> Any:
        """Apply TransformsWrapper module.

        Args:
            image (Any): Input image.
            kwargs (Any): Additional arguments.

        Returns:
            Any: Transformation results.
        """

        # if isinstance(signal, Image.Image):
        #     image = np.asarray(image)
        return self.transformations(signal)



class Normalize(torch.nn.Module):
    def forward(self, waveform):
        return waveform / waveform.abs().max()

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

