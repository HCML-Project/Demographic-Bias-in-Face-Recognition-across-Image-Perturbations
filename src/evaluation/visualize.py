"""
Module for visualizing evaluation results, including score distributions, ROC/DET curves, and fairness metrics.
"""

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay

from common.config import OVERALL_GROUP_KEY
from common.perturbations import PERTURBATION_DEFS
from evaluation.metrics import Metrics


def save_fig(fig: Figure, out_path: Path) -> None:
    """
    Save a matplotlib figure to the specified path in multiple formats (PNG, SVG, PDF).
    Creates the parent directory if it does not exist.

    Args:
        fig: The matplotlib figure to save.
        out_path: The path where the figure will be saved.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=300)


def plot_score_distribution(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    title: str,
    out_path: Path,
    eer_threshold: float | None = None,
) -> None:
    """
    Plot Kernel Density Estimate (KDE) of genuine and impostor scores, optionally with EER threshold line.

    Args:
        genuine_scores: Array of genuine scores (cosine similarity).
        impostor_scores: Array of impostor scores (cosine similarity).
        title: Title of the plot.
        out_path: Path to save the plot.
        eer_threshold: Optional EER threshold to draw as a vertical line.
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        genuine_scores, ax=ax, label="Genuine", fill=True, alpha=0.4, color="steelblue"
    )
    sns.kdeplot(
        impostor_scores, ax=ax, label="Impostor", fill=True, alpha=0.4, color="salmon"
    )

    if eer_threshold is not None:
        ax.axvline(
            eer_threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"EER thr={eer_threshold:.3f}",
        )

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def param_axis(param_series: pd.Series) -> tuple[pd.Series, str]:
    """
    Splits a pandas Series of 'key=value' strings into numeric values and the key label.

    Args:
        param_series: A pandas Series containing strings in the format 'key=value'.

    Returns:
        A tuple containing:
            - A pandas Series of numeric values extracted from the 'value' part of the strings.
            - A string representing the 'key' part of the strings, which can be used as
    """

    key = str(param_series.iloc[0]).split("=")[0]
    numeric: pd.Series = pd.to_numeric(  # type: ignore[assignment]
        param_series.str.split("=").str[-1], errors="coerce"
    )
    return numeric, key


def plot_metrics_per_perturbation(
    verification_df: pd.DataFrame,
    metric: str,
    perturbation_name: str,
    out_path: Path,
) -> None:
    """
    Plot a specific metric vs. perturbation strength for a given perturbation.

    Args:
        verification_df: A pandas DataFrame containing the verification report.
        metric: The metric to plot (e.g., "EER", "ZeroFMR_FNMR").
        perturbation_name: The name of the perturbation to plot.
        out_path: Path to save the generated plot.
    """

    sub = verification_df.loc[
        verification_df["perturbation"] == perturbation_name
    ].copy()
    sub["x_param"], xlabel = param_axis(sub["param"])
    sub = sub.sort_values("x_param")

    fig, ax = plt.subplots(figsize=(8, 4))
    for group, gdf in sub.groupby("group"):
        gdf = gdf.sort_values("x_param")
        ax.plot(gdf["x_param"], gdf[metric], marker="o", label=group)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — {PERTURBATION_DEFS[perturbation_name].display_name}")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def plot_fairness_discrimination_rate_per_perturbation(
    fairness_df: pd.DataFrame,
    perturbation_name: str,
    out_path: Path,
    metrics: list[str] | None = None,
) -> None:
    """
    Plot Fairness Discrimination Rate (FDR) vs. perturbation strength for a given perturbation.

    Args:
        fairness_df: A pandas DataFrame containing the fairness report.
        perturbation_name: The name of the perturbation to plot.
        out_path: Path to save the generated plot.
        metrics: Operating points to include. Defaults to all four (EER, ZeroFMR, FMR100, FMR1000).
    """
    if metrics is None:
        metrics = ["EER", "ZeroFMR", "FMR100", "FMR1000"]

    sub = fairness_df.loc[fairness_df["perturbation"] == perturbation_name].copy()
    sub["x_param"], xlabel = param_axis(sub["param"])
    sub = sub.sort_values("x_param")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels_map = [
        ("EER_FDR", "@ EER"),
        ("ZeroFMR_FDR", "@ ZeroFMR"),
        ("FMR100_FDR", "@ FMR100"),
        ("FMR1000_FDR", "@ FMR1000"),
    ]

    for col, label in [
        (col, lbl) for col, lbl in labels_map if col.removesuffix("_FDR") in metrics
    ]:
        ax.plot(sub["x_param"], sub[col], marker="o", label=label)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FDR (1 = perfectly fair)")
    ax.set_title(f"FDR — {PERTURBATION_DEFS[perturbation_name].display_name}")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def plot_degree_of_bias_per_perturbation(
    fairness_df: pd.DataFrame,
    perturbation_name: str,
    out_path: Path,
    metrics: list[str] | None = None,
) -> None:
    """
    Plot Degree of Bias (std across groups) vs. perturbation strength.

    DoB = std(metric across demographic groups) per perturbation.
    One line per operating point (EER, ZeroFMR, FMR100, FMR1000).
    Higher DoB means higher disparity across groups.

    Args:
        fairness_df: A pandas DataFrame containing the fairness report.
        perturbation_name: The name of the perturbation to plot.
        out_path: Path to save the generated plot.
        metrics: Operating points to include. Defaults to all four (EER, ZeroFMR, FMR100, FMR1000).
    """
    if metrics is None:
        metrics = ["EER", "ZeroFMR", "FMR100", "FMR1000"]

    sub = fairness_df.loc[fairness_df["perturbation"] == perturbation_name].copy()
    sub["x_param"], xlabel = param_axis(sub["param"])
    sub = sub.sort_values("x_param")

    dob_cols = [
        ("EER_DoB", "EER"),
        ("ZeroFMR_DoB", "ZeroFMR"),
        ("FMR100_DoB", "FMR100"),
        ("FMR1000_DoB", "FMR1000"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in [(col, lbl) for col, lbl in dob_cols if lbl in metrics]:
        if col in sub.columns:
            ax.plot(sub["x_param"], sub[col] * 100, marker="o", label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoB — std of error rate across groups (pp)")
    ax.set_title(
        f"Degree of Bias — {PERTURBATION_DEFS[perturbation_name].display_name}"
    )
    ax.legend()

    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def plot_roc_det_curve(
    curves: Mapping[str, Metrics],
    similarities: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    out_path: Path,
    title_suffix: str = "",
    reference_curves: Mapping[str, Metrics] | None = None,
    reference_similarities: Mapping[str, np.ndarray] | None = None,
    reference_labels: Mapping[str, np.ndarray] | None = None,
) -> None:
    """
    ROC and DET curves for multiple groups, with optional reference curves.

    Also plots the overall curve (if present) with a black dashed line.

    Saves a combined ROC+DET plot (out_path), as well as separate ROC-only (out_path with "roc_det" → "roc") and DET-only plots (out_path with "roc_det" → "det").

    Uses sklearn's RocCurveDisplay and DetCurveDisplay for plotting. Uses probit scale for DET and linear scale for ROC (sklearn defaults).

    Args:
        curves: Mapping of group names to Metrics objects containing EER and other metrics.
        similarities: Mapping of group names to similarity score arrays.
        labels: Mapping of group names to label arrays (1 for genuine, 0 for impostor).
        out_path: Path to save the combined ROC+DET plot.
        title_suffix: Optional suffix to add to the plot titles.
        reference_curves: Optional mapping of group names to reference Metrics objects for comparison.
        reference_similarities: Optional mapping of group names to reference similarity score arrays.
        reference_labels: Optional mapping of group names to reference label arrays.
    """

    def draw_curves(ax_roc: Axes, ax_det: Axes) -> None:
        if (
            reference_curves is not None
            and reference_similarities is not None
            and reference_labels is not None
        ):
            ref_ordered = sorted(
                reference_similarities.keys(), key=lambda k: (k == OVERALL_GROUP_KEY, k)
            )
            for lbl in ref_ordered:
                eer = reference_curves[lbl].EER
                is_overall = lbl == OVERALL_GROUP_KEY
                lw = 1.5 if is_overall else 1.0
                ls = "--" if is_overall else "-"
                name = f"{lbl} orig (EER={eer * 100:.2f}%)"
                color = "#888888"

                RocCurveDisplay.from_predictions(
                    reference_labels[lbl],
                    reference_similarities[lbl],
                    ax=ax_roc,
                    name=name,
                    curve_kwargs={
                        "lw": lw,
                        "ls": ls,
                        "color": color,
                        "alpha": 0.5,
                    },
                )
                DetCurveDisplay.from_predictions(
                    reference_labels[lbl],
                    reference_similarities[lbl],
                    ax=ax_det,
                    name=name,
                    **{
                        "lw": lw,
                        "ls": ls,
                        "color": color,
                        "alpha": 0.5,
                    },
                )

        ordered = sorted(similarities.keys(), key=lambda k: (k == OVERALL_GROUP_KEY, k))
        for lbl in ordered:
            eer = curves[lbl].EER
            is_overall = lbl == OVERALL_GROUP_KEY
            lw = 2.0 if is_overall else 1.8
            ls = "--" if is_overall else "-"
            name = f"{lbl}  (EER={eer * 100:.2f}%)"
            color = "black" if is_overall else None

            RocCurveDisplay.from_predictions(
                labels[lbl],
                similarities[lbl],
                ax=ax_roc,
                name=name,
                curve_kwargs={
                    "lw": lw,
                    "ls": ls,
                    "color": color,
                },
            )
            DetCurveDisplay.from_predictions(
                labels[lbl],
                similarities[lbl],
                ax=ax_det,
                name=name,
                **{
                    "lw": lw,
                    "ls": ls,
                    "color": color,
                },
            )

    # Combined
    fig, (ax_roc, ax_det) = plt.subplots(1, 2, figsize=(13, 5))
    draw_curves(ax_roc, ax_det)

    ax_roc.set_xlabel("False Match Rate (FMR)")
    ax_roc.set_ylabel("True Match Rate (1 − FNMR)")
    ax_roc.set_title(f"ROC{' — ' + title_suffix if title_suffix else ''}")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    ax_det.set_xlabel("False Match Rate (FMR)")
    ax_det.set_ylabel("False Non-Match Rate (FNMR)")
    ax_det.set_title(f"DET{' — ' + title_suffix if title_suffix else ''}")
    ax_det.legend(fontsize=8, loc="upper right")
    ax_det.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)

    # ROC only
    fig_roc, ax_roc2 = plt.subplots(figsize=(7, 5))
    fig_det_dummy, ax_det_dummy = plt.subplots()
    draw_curves(ax_roc2, ax_det_dummy)
    plt.close(fig_det_dummy)

    ax_roc2.set_xlabel("False Match Rate (FMR)")
    ax_roc2.set_ylabel("True Match Rate (1 − FNMR)")
    ax_roc2.set_title(f"ROC{' — ' + title_suffix if title_suffix else ''}")
    ax_roc2.legend(fontsize=8, loc="lower right")
    ax_roc2.grid(True, alpha=0.3)

    fig_roc.tight_layout()
    save_fig(fig_roc, Path(str(out_path).replace("roc_det", "roc", 1)))
    plt.close(fig_roc)

    # DET only
    fig_det, ax_det2 = plt.subplots(figsize=(7, 5))
    fig_roc_dummy, ax_roc_dummy = plt.subplots()
    draw_curves(ax_roc_dummy, ax_det2)
    plt.close(fig_roc_dummy)

    ax_det2.set_xlabel("False Match Rate (FMR)")
    ax_det2.set_ylabel("False Non-Match Rate (FNMR)")
    ax_det2.set_title(f"DET{' — ' + title_suffix if title_suffix else ''}")
    ax_det2.legend(fontsize=8, loc="upper right")
    ax_det2.grid(True, alpha=0.3)

    fig_det.tight_layout()
    save_fig(fig_det, Path(str(out_path).replace("roc_det", "det", 1)))
    plt.close(fig_det)
