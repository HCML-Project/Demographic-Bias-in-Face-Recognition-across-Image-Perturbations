"""
Script to analyse saved features and generate a fairness report and plots.

Usage:
    python src/analysis.py \
        --output <path_to_output_directory> \
        --races <list_of_races> \
        --perturbations-config <path_to_perturbations_json>
    Arguments:
        --output: Path to the output directory containing features and where to save report/plots (default: output).
        --races: List of races to include in the analysis (default: ["African", "Asian", "Caucasian", "Indian"]).
        --perturbations-config: Path to the perturbations JSON file or an inline JSON string (default: None).
"""

from collections.abc import Sequence
from itertools import groupby, product
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from common.config import (
    DEFAULT_OUTPUT,
    DEFAULT_RACES,
    OVERALL_GROUP_KEY,
    ORIGINAL_PERTURBATION_KEY,
)
from common.perturbations import Perturbation, load_perturbations
from data.features import Results, load_results, results_exist
from evaluation.report import (
    build_fairness_report,
    build_verification_report,
    create_verification_records,
    save_report,
)
from evaluation.metrics import Metrics, compute_metrics
from evaluation.visualize import (
    plot_degree_of_bias_per_perturbation,
    plot_fairness_discrimination_rate_per_perturbation,
    plot_metrics_per_perturbation,
    plot_roc_det_curve,
    plot_score_distribution,
)


def load_results_with_fallback(
    output_dir: Path,
    race: str,
    pert_path: str,
    original_path: str = ORIGINAL_PERTURBATION_KEY,
) -> Results:
    """
    Load results for a perturbation, falling back to original if not found.

    Neutral perturbations (e.g. brightness_beta=1.0) are identical to the original
    but were never extracted as separate features, so we just use the original features for them.

    Args:
        output_dir: Path to the output directory containing features.
        race: Race name (e.g. "African", "Asian", "Caucasian", "Indian").
        pert_path: Path of the perturbation (e.g. "brightness_beta=0.5").
        original_path: Path of the original features (default: "original").

    Returns:
        Results object if found.

    Raises:
        FileNotFoundError: If neither the perturbation nor the original results are found.
    """

    out_dir = output_dir / "features" / race / pert_path

    if results_exist(out_dir):
        return load_results(out_dir)

    orig_dir = output_dir / "features" / race / original_path
    if results_exist(orig_dir):
        return load_results(orig_dir)

    raise FileNotFoundError(
        f"Results not found for race '{race}' and perturbation '{pert_path}', and no original results found either."
    )


def pool_overall(
    per_group: Mapping[str, Results],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate similarities + labels across all groups (in-memory only).

    Args:
        per_group: A mapping of group name to Results object containing similarities and labels.

    Returns:
        A tuple of two numpy arrays: all similarities and all labels.
    """
    all_sims = np.concatenate([v.similarities for v in per_group.values()])
    all_labels = np.concatenate([v.labels for v in per_group.values()])
    return all_sims, all_labels


def compute_all_metrics(
    output_dir: Path,
    races: Sequence[str],
    perturbations: list[Perturbation],
) -> tuple[dict[str, dict[str, Results]], dict[str, dict[str, Metrics]]]:
    """
    Compute metrics for all races and perturbations, including overall metrics by pooling all races together.

    Args:
        output_dir: Path to the output directory containing features
        races: List of races to include in the analysis (e.g. ["African", "Asian", "Caucasian", "Indian"]).
        perturbations: List of Perturbation objects to include in the analysis.

    Returns:
        A tuple containing:
            - A dictionary mapping perturbation paths to dictionaries of race names to Results objects.
            - A dictionary mapping perturbation paths to dictionaries of race names to Metrics objects, including an "overall" key for pooled metrics across all races
    """

    # Mapping of perturbation path -> race -> Results
    perturbation_results: dict[str, dict[str, Results]] = {
        p.path: {} for p in perturbations
    }
    # Mapping of perturbation path -> race -> Metrics
    perturbation_metrics: dict[str, dict[str, Metrics]] = {
        p.path: {} for p in perturbations
    }

    for race, perturbation in product(races, perturbations):
        try:
            results = load_results_with_fallback(output_dir, race, perturbation.path)
        except FileNotFoundError:
            print(f"  MISSING: {race}/{perturbation.path} — run extract_all first")
            continue

        perturbation_results[perturbation.path][race] = results
        perturbation_metrics[perturbation.path][race] = compute_metrics(
            results.similarities, results.labels
        )

    # compute overall Metrics for each perturbation by pooling all races together
    for perturbation in perturbations:
        race_results = perturbation_results.get(perturbation.path)
        if not race_results:
            continue

        sims, labels = pool_overall(race_results)
        perturbation_metrics[perturbation.path][OVERALL_GROUP_KEY] = compute_metrics(
            sims, labels
        )

    return perturbation_results, perturbation_metrics


def build_reports(
    output_dir: Path,
    perturbations: list[Perturbation],
    perturbation_metrics: dict[str, dict[str, Metrics]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build verification and fairness reports from computed metrics.

    Args:
        output_dir: Path to the output directory where reports will be saved.
        perturbations: List of Perturbation objects to include in the analysis.
        perturbation_metrics: A dictionary mapping perturbation paths to dictionaries of race names to Metrics

    Returns:
        A tuple containing:
            - A pandas DataFrame containing the verification report (one row per race per perturbation with verification metrics).
            - A pandas DataFrame containing the fairness report (one row per perturbation, with fairness metrics computed across races).
    """

    records = create_verification_records(perturbations, perturbation_metrics)

    verification_df = build_verification_report(records)
    fairness_df = build_fairness_report(records)

    save_report(verification_df, output_dir / "verification_report.csv")
    save_report(fairness_df, output_dir / "fairness_report.csv")

    return verification_df, fairness_df


def plot_score_distributions(
    output_dir: Path,
    perturbations: list[Perturbation],
    perturbation_results: dict[str, dict[str, Results]],
    perturbation_metrics: dict[str, dict[str, Metrics]],
    force: bool = False,
    include_overall: bool = True,
) -> None:
    """
    Plot score distributions for each race and perturbation, including overall distributions by pooling all races together.

    Args:
        output_dir: Path to the output directory where plots will be saved.
        perturbations: List of Perturbation objects to include in the analysis.
        perturbation_results: A dictionary mapping perturbation paths to dictionaries of race names to Results objects.
        perturbation_metrics: A dictionary mapping perturbation paths to dictionaries of race names to Metrics.
        force: If True, regenerate plots even if they already exist on disk.
        include_overall: If True, also plot the pooled overall score distribution.
    """

    for perturbation in perturbations:
        race_results = perturbation_results.get(perturbation.path, {})

        for race, results in race_results.items():
            score_distribution_path = (
                output_dir
                / "plots"
                / race
                / perturbation.path
                / "score_distribution.png"
            )

            if score_distribution_path.exists() and not force:
                continue

            metrics = perturbation_metrics[perturbation.path][race]
            plot_score_distribution(
                genuine_scores=results.similarities[results.labels == 1],
                impostor_scores=results.similarities[results.labels == 0],
                title=f"{race} — {perturbation.display}",
                out_path=score_distribution_path,
                eer_threshold=metrics.EER_threshold,
            )

        if not include_overall:
            continue

        overall_metrics = perturbation_metrics[perturbation.path].get(OVERALL_GROUP_KEY)
        if overall_metrics is None:
            continue

        score_distribution_path = (
            output_dir
            / "plots"
            / OVERALL_GROUP_KEY
            / perturbation.path
            / "score_distribution.png"
        )

        if score_distribution_path.exists() and not force:
            continue

        overall_similarities, overall_labels = pool_overall(race_results)
        plot_score_distribution(
            genuine_scores=overall_similarities[overall_labels == 1],
            impostor_scores=overall_similarities[overall_labels == 0],
            title=f"{OVERALL_GROUP_KEY} — {perturbation.display}",
            out_path=score_distribution_path,
            eer_threshold=overall_metrics.EER_threshold,
        )


def plot_curves(
    output_dir: Path,
    perturbations: list[Perturbation],
    perturbation_results: dict[str, dict[str, Results]],
    perturbation_metrics: dict[str, dict[str, Metrics]],
    force: bool = False,
    include_overall: bool = True,
) -> None:
    """
    Plot ROC and DET curves for each race and perturbation, including overall curves by pooling all races together.

    Args:
        output_dir: Path to the output directory where plots will be saved.
        perturbations: List of Perturbation objects to include in the analysis.
        perturbation_results: A dictionary mapping perturbation paths to dictionaries of race names to Results objects.
        perturbation_metrics: A dictionary mapping perturbation paths to dictionaries of race names to Metrics.
        force: If True, regenerate plots even if they already exist on disk.
        include_overall: If True, include the pooled overall curve in each plot.
    """

    plots_root = output_dir / "plots"

    # Get original Race -> Metrics/Similarities/Labels for reference curves
    original_metrics: dict[str, Metrics] | None = perturbation_metrics.get(
        ORIGINAL_PERTURBATION_KEY
    )
    original_sims_dict: dict[str, np.ndarray] | None = None
    original_labels_dict: dict[str, np.ndarray] | None = None

    if race_results := perturbation_results.get(ORIGINAL_PERTURBATION_KEY):
        original_sims_dict = {
            race: metrics.similarities for race, metrics in race_results.items()
        }
        original_labels_dict = {
            race: metrics.labels for race, metrics in race_results.items()
        }

        if include_overall:
            sims_all, labels_all = pool_overall(race_results)
            original_sims_dict[OVERALL_GROUP_KEY] = sims_all
            original_labels_dict[OVERALL_GROUP_KEY] = labels_all

    for pert in perturbations:
        if pert.name == ORIGINAL_PERTURBATION_KEY:
            continue

        race_results = perturbation_results.get(pert.path)
        if not race_results:
            continue

        out_path = plots_root / "roc_det" / pert.name / f"{pert.path}.png"  # 2.35s
        if out_path.exists() and not force:
            continue

        sims_dict = {r: res.similarities for r, res in race_results.items()}
        labels_dict = {r: res.labels for r, res in race_results.items()}

        if include_overall:
            sims_overall, labels_overall = pool_overall(race_results)
            sims_dict[OVERALL_GROUP_KEY] = sims_overall
            labels_dict[OVERALL_GROUP_KEY] = labels_overall

        plot_roc_det_curve(
            curves=perturbation_metrics[pert.path],
            similarities=sims_dict,
            labels=labels_dict,
            out_path=out_path,
            title_suffix=pert.display,
            reference_curves=original_metrics,
            reference_similarities=original_sims_dict,
            reference_labels=original_labels_dict,
        )


def plot_summarization_plots(
    output_dir: Path,
    perturbations: list[Perturbation],
    verification_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    force: bool = False,
    include_overall: bool = True,
    op_metrics: list[str] | None = None,
) -> None:
    """
    Plot summarization plots for each perturbation, including EER, FNMR at FMR=0.01, FNMR at FMR=0.001, Fairness Discrimination Rate (FDR), and Degree of Bias (DoB).

    Args:
        output_dir: Path to the output directory where plots will be saved.
        perturbations: List of Perturbation objects to include in the analysis.
        verification_df: A pandas DataFrame containing the verification report (one row per race per perturbation with verification metrics).
        fairness_df: A pandas DataFrame containing the fairness report (one row per perturbation, with fairness metrics computed across races).
        force: If True, regenerate plots even if they already exist on disk.
        include_overall: If True, include the pooled overall curve in the plots (not relevant for FDR and DoB plots).
        op_metrics: Operating points to include in all summarization plots. Defaults to all four (EER, ZeroFMR, FMR100, FMR1000).
    """
    if op_metrics is None:
        op_metrics = ["EER", "ZeroFMR", "FMR100", "FMR1000"]

    plots_root = output_dir / "plots"

    vdf = verification_df
    if not include_overall:
        vdf = vdf.loc[vdf["group"] != OVERALL_GROUP_KEY]

    all_metrics_plots: list[tuple[str, str]] = [
        ("EER", "EER"),
        ("ZeroFMR_FNMR", "ZeroFMR"),
        ("FMR100_FNMR", "FMR100"),
        ("FMR1000_FNMR", "FMR1000"),
    ]

    for name, _ in groupby(perturbations, key=lambda p: p.name):
        if name == ORIGINAL_PERTURBATION_KEY:
            continue

        for metric, folder in [(m, f) for m, f in all_metrics_plots if f in op_metrics]:
            out_path = plots_root / folder / f"{name}.png"
            if out_path.exists() and not force:
                continue

            plot_metrics_per_perturbation(
                vdf,
                metric=metric,
                perturbation_name=name,
                out_path=out_path,
            )

        fdr_path = plots_root / "FDR" / f"{name}.png"
        if not fdr_path.exists() or force:
            plot_fairness_discrimination_rate_per_perturbation(
                fairness_df,
                perturbation_name=name,
                out_path=fdr_path,
                metrics=op_metrics,
            )

        dob_path = plots_root / "DoB" / f"{name}.png"
        if not dob_path.exists() or force:
            plot_degree_of_bias_per_perturbation(
                fairness_df,
                perturbation_name=name,
                out_path=dob_path,
                metrics=op_metrics,
            )


def analyse_all(
    output_dir: Path,
    races: list[Literal["African", "Asian", "Caucasian", "Indian"]] = DEFAULT_RACES,
    perturbations_config: Path | str | None = None,
    force: bool = False,
    include_overall: bool = True,
    op_metrics: list[str] | None = None,
) -> None:
    """
    Analyse saved features and generate a fairness report and plots.

    Args:
        output_dir: Path to the output directory containing features and where to save report/plots.
        races: List of races to include in the analysis (default: all races).
        perturbations_config: Path to the perturbations JSON file or an inline JSON string (default: None).
        force: If True, regenerate all plots even if they already exist on disk.
        include_overall: If True, include the pooled overall curve in the plots.
        op_metrics: Operating points to include in all summarization plots. Defaults to all four (EER, ZeroFMR, FMR100, FMR1000).
    """

    perturbations = load_perturbations(
        perturbations_config, include_original=True, include_neutrals=True
    )
    results, metrics = compute_all_metrics(output_dir, races, perturbations)
    verification_df, fairness_df = build_reports(output_dir, perturbations, metrics)
    plot_score_distributions(
        output_dir,
        perturbations,
        results,
        metrics,
        force=force,
        include_overall=include_overall,
    )
    plot_curves(
        output_dir,
        perturbations,
        results,
        metrics,
        force=force,
        include_overall=include_overall,
    )
    plot_summarization_plots(
        output_dir,
        perturbations,
        verification_df,
        fairness_df,
        force=force,
        include_overall=include_overall,
        op_metrics=op_metrics,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse saved features and generate fairness report and plots"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory containing features and where to save report/plots",
    )
    parser.add_argument(
        "--races",
        nargs="+",
        default=DEFAULT_RACES,
        help="List of races to include in the analysis (default: all races)",
    )
    parser.add_argument(
        "--perturbations-config",
        type=str,
        default=None,
        help="Path to perturbations JSON file, or an inline JSON string",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Regenerate all plots even if they already exist on disk",
    )
    parser.add_argument(
        "--no-overall",
        action="store_false",
        dest="include_overall",
        help="Exclude the pooled overall curve from ROC/DET plots",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["EER", "ZeroFMR", "FMR100", "FMR1000"],
        choices=["EER", "ZeroFMR", "FMR100", "FMR1000"],
        help="Operating points to include in summarization plots (default: all four)",
    )
    args = parser.parse_args()

    analyse_all(
        output_dir=args.output,
        races=args.races,
        perturbations_config=args.perturbations_config,
        force=args.force,
        include_overall=args.include_overall,
        op_metrics=args.metrics,
    )
