"""
Script to extract face embeddings and cosine similarities for all races and perturbations using a specified model checkpoint and dataset. The results are saved to the specified output directory.
The script supports caching of results to avoid redundant computations. If the results for a specific race and perturbation already exist in the output directory, they will be skipped unless the --no-cache flag is provided.

Usage:
    python src/extract.py \
        --checkpoint <path_to_checkpoint> \
        --dataset <path_to_dataset> \
        --output <path_to_output_directory> \
        --races <list_of_races> \
        --perturbations-config <path_to_perturbations_json> \
        --batch-size <batch_size> \
        --num-workers <num_workers> \
        --seed <random_seed> \
        --no-cache
    Arguments:
        --checkpoint: Path to the model checkpoint file (default: data/195520backbone.pth).
        --dataset: Path to the RFW dataset directory (default: data/RFW_test_pairs).
        --output: Path to the output directory where results will be saved (default: output).
        --races: List of races to process (default: ["African", "Asian", "Caucasian", "Indian"]).
        --perturbations-config: Path to the perturbations JSON file or an inline JSON string (default: None).
        --batch-size: Batch size for the DataLoader (default: 64).
        --num-workers: Number of worker threads for the DataLoader (default: 4).
        --seed: Random seed for reproducibility (default: 42).
        --no-cache: If provided, forces recomputation of results even if they already exist in the output directory.
"""

from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader

from common.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    DEFAULT_OUTPUT,
    DEFAULT_RACES,
    SEED,
)
from common.perturbations import load_perturbations
from common.seed import seed_everything
from data.dataset import RFWPairsDataset
from data.features import extract_features_and_similarities, results_exist, save_results
from model.model import (
    get_device,
    get_pre_norm_transform,
    get_normalize_transform,
    load_model,
)


def extract_all(
    dataset_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    races: list[Literal["African", "Asian", "Caucasian", "Indian"]],
    batch_size: int = 64,
    num_workers: int = 4,
    perturbations_config: Path | str | None = None,
    seed: int = SEED,
    skip_existing: bool = True,
) -> None:
    """
    Extract face embeddings and cosine similarities for all races and perturbations using a specified model checkpoint and dataset. The results are saved to the specified output directory.

    Args:
        dataset_root (Path): Path to the RFW dataset directory.
        checkpoint_path (Path): Path to the model checkpoint file.
        output_dir (Path): Path to the output directory where results will be saved.
        races (list[Literal["African", "Asian", "Caucasian", "Indian"]]): List of races to process.
        batch_size (int): Batch size for the DataLoader. Default is 64.
        num_workers (int): Number of worker threads for the DataLoader. Default is 4.
        perturbations_config (Path | str | None): Path to the perturbations JSON file or an inline JSON string. If None, no perturbations will be applied. Default is None.
        seed (int): Random seed for reproducibility. Default is SEED.
        skip_existing (bool): If True, skips processing for races and perturbations that already have results in the output directory. Default is True.
    """

    seed_everything(seed)
    device = get_device()
    print(f"Device: {device}")

    model = load_model(checkpoint_path, device)
    transform = get_pre_norm_transform(image_size=112)
    normalize = get_normalize_transform()
    perturbations = load_perturbations(perturbations_config, include_original=True)

    for race in races:
        print(f"\n=== Race: {race} ===")
        dataset = RFWPairsDataset(dataset_root, race, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        for perturbation in perturbations:
            out_dir = output_dir / "features" / race / perturbation.path
            print(f"  [{race}] {perturbation.path}", end=" ")

            if skip_existing and results_exist(out_dir):
                print("(cached)")
            else:
                print("(running)")
                results = extract_features_and_similarities(
                    model=model,
                    dataloader=loader,
                    device=device,
                    normalize=normalize,
                    perturbation=perturbation,
                    seed=seed,
                )
                save_results(results, out_dir)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract face embeddings and cosine similarities for all races and perturbations"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the RFW dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output directory where results will be saved",
    )
    parser.add_argument(
        "--races", nargs="+", default=DEFAULT_RACES, help="List of races to process"
    )
    parser.add_argument(
        "--perturbations-config",
        type=str,
        default=None,
        help="Path to perturbations JSON file, or an inline JSON string",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for the DataLoader"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for the DataLoader",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute even if results already exist",
    )
    args = parser.parse_args()

    extract_all(
        dataset_root=args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        races=args.races,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        perturbations_config=args.perturbations_config,
        seed=args.seed,
        skip_existing=not args.no_cache,
    )
