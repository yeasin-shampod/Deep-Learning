"""
data.py
-------
Dataset definition for the solar-cell defect classification challenge
(Deep Learning Exercise 4, FAU Erlangen-Nuernberg).

``ChallengeDataset`` wraps the information stored in ``data.csv`` and turns
each row into a ``(image_tensor, label_tensor)`` pair consumable by a
``torch.utils.data.DataLoader``.
"""

from typing import Tuple

import numpy as np
import torch
import torchvision as tv
from skimage.color import gray2rgb
from skimage.io import imread
from torch.utils.data import Dataset

# Channel-wise statistics of the TRAINING data, computed AFTER ToTensor
# (i.e. on values already scaled to [0, 1]). The raw images are grayscale,
# so the three RGB channels share identical mean/std values.
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    """PyTorch ``Dataset`` for the electroluminescence solar-cell images.

    Parameters
    ----------
    data : pandas.DataFrame
        Table parsed from ``data.csv`` (separator ``;``). Column 0 holds the
        relative image path, columns 1 and 2 hold the binary ``crack`` and
        ``inactive`` labels.
    mode : str
        Either ``"train"`` or ``"val"``. In training mode light,
        label-preserving data augmentation is applied. In validation mode
        only the deterministic preprocessing pipeline is used, so that the
        evaluation is reproducible and the normalization matches the server.
    """

    def __init__(self, data, mode: str) -> None:
        super().__init__()
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got '{mode}'")

        # Reset the index so positional __getitem__ access always aligns,
        # even after a shuffled train/val split upstream.
        self._data = data.reset_index(drop=True)
        self._mode = mode

        if mode == "train":
            # Augmentation must operate on the PIL image, i.e. before ToTensor.
            # Flips are safe here: an image's orientation carries no class
            # information, so the labels remain valid after flipping.
            self._transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.RandomVerticalFlip(),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(train_mean, train_std),
                ]
            )
        else:
            # Deterministic pipeline only. Any randomness here would make the
            # validation metrics noisy and break the normalization test.
            self._transform = tv.transforms.Compose(
                [
                    tv.transforms.ToPILImage(),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(train_mean, train_std),
                ]
            )

    def __len__(self) -> int:
        """Return the number of samples currently held by the dataset."""
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ``index``-th ``(image, label)`` pair as tensors.

        The raw image is grayscale ``(H, W)``; it is expanded to a 3-channel
        RGB array before the transform pipeline turns it into a normalized
        ``(3, 300, 300)`` float tensor.
        """
        row = self._data.iloc[index]

        # Column 0 -> relative path to the grayscale PNG.
        image = imread(row.iloc[0])          # (H, W), dtype uint8
        image = gray2rgb(image)              # (H, W, 3), dtype uint8

        # ToPILImage -> [augment] -> ToTensor -> Normalize.
        image = self._transform(image)       # (3, 300, 300), float32

        # Columns 1, 2 -> crack / inactive labels. Float is required by BCELoss.
        label = torch.tensor(
            row.iloc[1:].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        return image, label
