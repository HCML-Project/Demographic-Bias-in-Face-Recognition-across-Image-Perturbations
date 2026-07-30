"""
Dataset classes for the RFW (Racial Faces in the Wild) dataset.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class RFWPairsDataset(Dataset):
    """
    Dataset class for the RFW (Racial Faces in the Wild) dataset.

    Each item in the dataset is a tuple of two images and a label indicating whether they are of the same person (1) or not (0).

    The dataset is organized into directories for each race, with each directory containing subdirectories for each pair of images.
    The subdirectory names follow the format "pairN_True" or "pairN_False", where N is the pair number and True/False indicates whether the images are of the same person.
    The images are expected to be in PNG format and named as "N.png" and "N+1.png" for each pair, where N is the pair number.
    The dataset can be transformed using a provided transform function, which should convert the images to tensors.
    """

    def __init__(
        self,
        root: Path,
        race: Literal["African", "Asian", "Caucasian", "Indian"],
        transform: Callable[[Image.Image], Tensor],
    ):
        """
        Initialize the RFWPairsDataset.

        Args:
            root (Path): The root directory of the RFW dataset.
            race (Literal["African", "Asian", "Caucasian", "Indian"]): The race for which to load the dataset.
            transform (Callable[[Image.Image], Tensor]): A function to transform the images to tensors.

        Raises:
            FileNotFoundError: If the race directory does not exist in the root directory.
        """

        race_dir = Path(root) / f"{race}_test"

        if not race_dir.exists():
            raise FileNotFoundError(f"Race directory not found: {race_dir}")

        self.race = race
        self.pairs = sorted(
            race_dir.iterdir(),
            key=lambda p: int(p.name.split("_")[0].replace("pair", "")),
        )
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of pairs in the dataset.

        Returns:
            int: The number of pairs in the dataset.
        """

        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        """
        Get a pair of images and their label.

        Args:
            idx (int): The index of the pair to retrieve.

        Returns:
            tuple[Tensor, Tensor, int]: A tuple containing the two images as tensors and the label (1 for same person, 0 for different persons).
        """

        pair_dir = self.pairs[idx]
        label = 1 if pair_dir.name.endswith("_True") else 0
        n = int(pair_dir.name.split("_")[0].replace("pair", ""))

        img0 = Image.open(pair_dir / f"{n}.png").convert("RGB")
        img1 = Image.open(pair_dir / f"{n + 1}.png").convert("RGB")

        return self.transform(img0), self.transform(img1), label
