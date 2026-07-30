"""
Module for extracting features and similarities from a model given a dataloader.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.config import SEED
from common.perturbations import Perturbation
from common.seed import seed_everything


@dataclass
class Results:
    """
    Dataclass to hold the results of feature extraction and similarity computation.
    """

    features_0: np.ndarray
    """Features of the first set of images."""

    features_1: np.ndarray
    """Features of the second set of images."""

    similarities: np.ndarray
    """Cosine similarities between the features of the two sets of images."""

    labels: np.ndarray
    """Labels indicating whether the pairs of images are of the same person (1) or not (0)."""


@torch.no_grad()
def extract_features_and_similarities(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    normalize: nn.Module,
    perturbation: Perturbation | None = None,
    seed: int = SEED,
) -> Results:
    """
    Extract features and compute similarities for a given model and dataloader.

    Args:
        model (nn.Module): The model to use for feature extraction.
        dataloader (DataLoader): The dataloader providing the image pairs and labels.
        device (str): The device to run the computations on (e.g., 'cpu', 'cuda').
        normalize (nn.Module): A normalization module to apply to the images before feature extraction.
        perturbation (Perturbation | None): An optional perturbation to apply to the images before feature extraction. If None, no perturbation is applied.
        seed (int): The seed value for reproducibility. Default is SEED.

    Returns:
        Results: A dataclass containing the extracted features, computed similarities, and labels.
    """

    seed_everything(seed)

    cos = nn.CosineSimilarity(dim=1).to(device)
    perturb = perturbation.transform.to(device) if perturbation is not None else None
    normalize = normalize.to(device)

    all_feat0, all_feat1, all_sims, all_labels = [], [], [], []

    desc = str(perturbation) if perturbation is not None else "original"
    for img0, img1, labels in tqdm(dataloader, desc=desc, leave=False):
        img0 = img0.to(device)
        img1 = img1.to(device)

        if perturb is not None:
            img0 = perturb(img0)
            img1 = perturb(img1)

        img0 = normalize(img0)
        img1 = normalize(img1)

        feat0 = model(img0)
        feat1 = model(img1)

        all_feat0.append(feat0.cpu())
        all_feat1.append(feat1.cpu())
        all_sims.append(cos(feat0, feat1).cpu())
        all_labels.append(labels)

    return Results(
        features_0=torch.cat(all_feat0).numpy(),
        features_1=torch.cat(all_feat1).numpy(),
        similarities=torch.cat(all_sims).numpy(),
        labels=torch.cat(all_labels).numpy(),
    )


def save_results(results: Results, out_dir: Path) -> None:
    """
    Save the results of feature extraction and similarity computation to disk.

    Args:
        results (Results): The results to save.
        out_dir (Path): The directory to save the results in. If it does not exist, it will be created.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features_0.npy", results.features_0)
    np.save(out_dir / "features_1.npy", results.features_1)
    np.save(out_dir / "similarities.npy", results.similarities)
    np.save(out_dir / "labels.npy", results.labels)


def load_results(out_dir: Path) -> Results:
    """
    Load the results of feature extraction and similarity computation from disk.

    Args:
        out_dir (Path): The directory to load the results from.

    Returns:
        Results: The loaded results.
    """

    keys = ["features_0", "features_1", "similarities", "labels"]
    arrays = {k: np.load(out_dir / f"{k}.npy") for k in keys}
    return Results(
        features_0=arrays["features_0"],
        features_1=arrays["features_1"],
        similarities=arrays["similarities"],
        labels=arrays["labels"],
    )


def results_exist(out_dir: Path) -> bool:
    """
    Check if the results of feature extraction and similarity computation already exist in the specified directory.

    Args:
        out_dir (Path): The directory to check for existing results.

    Returns:
        bool: True if all result files exist, False otherwise.
    """

    return all(
        (out_dir / f"{k}.npy").exists()
        for k in ["features_0", "features_1", "similarities", "labels"]
    )
