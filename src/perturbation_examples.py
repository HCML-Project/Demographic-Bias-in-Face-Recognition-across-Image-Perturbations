"""
Script to generate example images for every perturbation value.

Usage:
    python src/perturbation_examples.py \
        --output <path_to_output_directory> \
        --dataset <path_to_dataset_directory>
    Arguments:
        --output: Path to the output directory where example images will be saved (default: output).
        --dataset: Path to the RFW dataset directory (default: data/RFW_test_pairs).
"""

from itertools import groupby
from pathlib import Path

import torch

import matplotlib.pyplot as plt

from common.config import DEFAULT_DATASET, DEFAULT_OUTPUT
from common.perturbations import PERTURBATION_DEFS, Perturbation, load_perturbations
from common.seed import seed_everything
from model.model import get_pre_norm_transform

COLS: int = 6


def load_sample(sample_dir: Path, image_size: int = 112) -> torch.Tensor:
    """
    Load a sample image from the given directory and apply the pre-normalization transform.

    Args:
        sample_dir (Path): Path to the directory containing the sample image.
        image_size (int): Size to which the image will be resized (default: 112).

    Returns:
        torch.Tensor: The pre-normalized image tensor.
    """
    img_path = sorted(sample_dir.iterdir())[1]
    from PIL import Image

    pil = Image.open(img_path).convert("RGB")
    return get_pre_norm_transform(image_size)(pil)


def save_group(
    name: str,
    perturbations: list[Perturbation],
    original: torch.Tensor,
    out_dir: Path,
    cols: int = COLS,
) -> None:
    """
    Save a grid of example images for a group of perturbations.

    Args:
        name (str): Name of the perturbation group.
        perturbations (list[Perturbation]): List of perturbation objects.
        original (torch.Tensor): The original image tensor.
        out_dir (Path): Directory where the output images will be saved.
        cols (int): Number of columns in the grid (default: 6).
    """

    n = len(perturbations)
    n_cols = min(cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes_flat = list(axes.flat) if n_rows > 1 or n_cols > 1 else [axes]

    seed_everything(0)
    with torch.no_grad():
        for ax, p in zip(axes_flat, perturbations):
            out = p.transform(original.unsqueeze(0)).squeeze(0).clamp(0, 1)
            ax.imshow(out.permute(1, 2, 0).numpy())
            ax.set_title(p.display_param, fontsize=8)
            ax.axis("off")

    for ax in axes_flat[n:]:
        ax.axis("off")

    display_name = PERTURBATION_DEFS[name].display_name
    fig.suptitle(display_name, fontsize=11)
    fig.tight_layout()
    base = out_dir / name
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(base.with_suffix(suffix), dpi=300)
    plt.close(fig)


def generate_perturbation_examples(
    output_dir: Path,
    dataset_root: Path,
    cols: int = COLS,
) -> None:
    """
    Generate example images for every perturbation value and save them to the specified output directory.

    Args:
        output_dir (Path): Path to the output directory where example images will be saved.
        dataset_root (Path): Path to the RFW dataset directory.
        cols (int): Number of columns in the grid (default: 6).
    """

    sample_dir = dataset_root / "African_test" / "pair0_True"
    original = load_sample(sample_dir)
    perturbations = load_perturbations(include_neutrals=True)

    for perturbation_name, group in groupby(perturbations, key=lambda p: p.name):
        group_list = list(group)
        print(f"{perturbation_name}: {len(group_list)} perturbation levels")
        save_group(
            perturbation_name,
            group_list,
            original,
            output_dir / "perturbation_examples",
            cols=cols,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate example images for every perturbation value",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output directory where example images will be saved",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the RFW dataset directory",
    )
    parser.add_argument(
        "--cols", type=int, default=COLS, help="Number of columns in the grid"
    )
    args = parser.parse_args()

    generate_perturbation_examples(
        output_dir=args.output,
        dataset_root=args.dataset,
        cols=args.cols,
    )
