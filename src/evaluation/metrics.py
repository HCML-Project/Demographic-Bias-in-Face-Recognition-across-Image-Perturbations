"""
Contains functions for computing metrics for biometric verification performance, including Equal Error Rate (EER), False Match Rate (FMR), and False Non-Match Rate (FNMR) at various operating points.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


@dataclass
class Metrics:
    """
    Metrics for evaluating biometric verification performance.
    """

    EER: float
    """
    Equal Error Rate (EER) is the point where False Match Rate (FMR) equals False Non-Match Rate (FNMR).
    Lower EER indicates better performance.
    """
    EER_threshold: float
    """The threshold at which EER occurs."""

    ZeroFMR_FMR: float
    """The False Match Rate (FMR) at the threshold where False Match Rate (FMR) is zero. Equal to 0."""
    ZeroFMR_FNMR: float
    """The False Non-Match Rate (FNMR) at the threshold where False Match Rate (FMR) is zero."""
    ZeroFMR_threshold: float
    """The threshold at which FMR is zero."""

    FMR100_FMR: float
    """The False Match Rate (FMR) at the threshold where False Match Rate (FMR) is 0.01 (1%). Equal to 0.01"""
    FMR100_FNMR: float
    """The False Non-Match Rate (FNMR) at the threshold where False Match Rate (FMR) is 0.01 (1%)."""
    FMR100_threshold: float
    """The threshold at which FMR is 0.01 (1%)."""

    FMR1000_FMR: float
    """The False Match Rate (FMR) at the threshold where False Match Rate (FMR) is 0.001 (0.1%). Equal to 0.001"""
    FMR1000_FNMR: float
    """The False Non-Match Rate (FNMR) at the threshold where False Match Rate (FMR) is 0.001 (0.1%)."""
    FMR1000_threshold: float
    """The threshold at which FMR is 0.001 (0.1%)."""


def compute_fnmr_at_fmr(
    fpr: np.ndarray,
    fnr: np.ndarray,
    thresholds: np.ndarray,
    target_fmr: float,
) -> tuple[float, float, float]:
    """
    Find the False Non-Match Rate (FNMR) and threshold at a given target False Match Rate (FMR).

    Args:
        fpr: A 1D array of False Match Rates (FMR) at various thresholds.
        fnr: A 1D array of False Non-Match Rates (FNMR) at various thresholds.
        thresholds: A 1D array of thresholds corresponding to the FMR and FNMR values.
        target_fmr: The target False Match Rate (FMR) for which to find the corresponding FNMR and threshold.

    Returns:
        A tuple containing:
            - The FNMR at the target FMR.
            - The threshold at which the target FMR occurs.
            - The actual FMR at the found threshold (may not be exactly equal to target_fmr due to discrete thresholds).
    """

    if target_fmr == 0.0:
        # sklearn prepends a "wrong" (fpr=0, tpr=0) anchor at index 0 (highest threshold).
        # Skip it: find the last real point where fpr == 0.
        candidates = np.where(fpr == 0)[0]
        idx = int(candidates[-1]) if len(candidates) > 1 else int(candidates[0])
    else:
        idx = np.nanargmin(np.abs(fpr - target_fmr))
    return float(fnr[idx]), float(thresholds[idx]), float(fpr[idx])


def compute_eer(
    fpr: np.ndarray, fnr: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """
    Compute the Equal Error Rate (EER) and the corresponding threshold.

    Args:
        fpr: A 1D array of False Match Rates (FMR) at various thresholds.
        fnr: A 1D array of False Non-Match Rates (FNMR) at various thresholds.
        thresholds: A 1D array of thresholds corresponding to the FMR and FNMR values.

    Returns:
        A tuple containing:
            - The Equal Error Rate (EER).
            - The threshold at which EER occurs.
    """
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold


def compute_metrics(similarities: np.ndarray, labels: np.ndarray) -> Metrics:
    """
    Compute metrics for biometric verification performance.

    Args:
        similarities: A 1D array of similarity scores between pairs of samples.
        labels: A 1D array of binary labels (0 or 1) indicating whether the pairs are genuine (1) or impostor (0).
    Returns:
        A Metrics object containing EER, FMR, FNMR, and thresholds at various operating points.
    """

    fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1)
    fnr = 1.0 - tpr

    eer, eer_threshold = compute_eer(fpr, fnr, thresholds)
    zero_fnmr, zero_thr, zero_fmr = compute_fnmr_at_fmr(
        fpr, fnr, thresholds, target_fmr=0.0
    )
    fnmr100, thr100, fmr100_actual = compute_fnmr_at_fmr(
        fpr, fnr, thresholds, target_fmr=0.01
    )
    fnmr1000, thr1000, fmr1000_actual = compute_fnmr_at_fmr(
        fpr, fnr, thresholds, target_fmr=0.001
    )

    return Metrics(
        EER=eer,
        EER_threshold=eer_threshold,
        ZeroFMR_FMR=zero_fmr,
        ZeroFMR_FNMR=zero_fnmr,
        ZeroFMR_threshold=zero_thr,
        FMR100_FMR=fmr100_actual,
        FMR100_FNMR=fnmr100,
        FMR100_threshold=thr100,
        FMR1000_FMR=fmr1000_actual,
        FMR1000_FNMR=fnmr1000,
        FMR1000_threshold=thr1000,
    )


def eval_at_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """
    Evaluate FMR and FNMR at a fixed threshold.

    Args:
        similarities: 1D array of similarity scores.
        labels: 1D array of binary labels (1 = genuine, 0 = impostor).
        threshold: The decision threshold (similarities >= threshold → predicted genuine).

    Returns:
        Tuple of (FMR, FNMR).
    """
    predicted_positive = similarities >= threshold
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    fnmr = float(np.sum(~predicted_positive[positives]) / n_pos) if n_pos > 0 else float("nan")
    fmr = float(np.sum(predicted_positive[negatives]) / n_neg) if n_neg > 0 else float("nan")
    return fmr, fnmr


def compute_metrics_at_thresholds(
    similarities: np.ndarray,
    labels: np.ndarray,
    reference: "Metrics",
) -> "Metrics":
    """
    Compute Metrics for a group using thresholds derived from a reference (pooled) Metrics object.

    All threshold fields are copied from `reference`; FMR/FNMR fields are evaluated
    against the group's own similarities/labels at those shared thresholds.

    Args:
        similarities: 1D array of similarity scores for this group.
        labels: 1D array of binary labels for this group.
        reference: Metrics object from the pooled/overall distribution whose thresholds are used.

    Returns:
        Metrics object with shared thresholds and group-specific FMR/FNMR values.
    """
    eer_fmr, eer_fnmr = eval_at_threshold(similarities, labels, reference.EER_threshold)
    zero_fmr, zero_fnmr = eval_at_threshold(similarities, labels, reference.ZeroFMR_threshold)
    fmr100_fmr, fmr100_fnmr = eval_at_threshold(similarities, labels, reference.FMR100_threshold)
    fmr1000_fmr, fmr1000_fnmr = eval_at_threshold(similarities, labels, reference.FMR1000_threshold)

    return Metrics(
        EER=(eer_fmr + eer_fnmr) / 2,
        EER_threshold=reference.EER_threshold,
        ZeroFMR_FMR=zero_fmr,
        ZeroFMR_FNMR=zero_fnmr,
        ZeroFMR_threshold=reference.ZeroFMR_threshold,
        FMR100_FMR=fmr100_fmr,
        FMR100_FNMR=fmr100_fnmr,
        FMR100_threshold=reference.FMR100_threshold,
        FMR1000_FMR=fmr1000_fmr,
        FMR1000_FNMR=fmr1000_fnmr,
        FMR1000_threshold=reference.FMR1000_threshold,
    )


def compute_degree_of_bias(series: pd.Series) -> float:
    """
    Compute the Degree of Bias (DoB) for a given metric across demographic groups.

    The Degree of Bias is defined as the standard deviation of the metric across different demographic groups. A higher DoB indicates greater disparity in performance across groups.

    Args:
        series: A pandas Series containing the metric values for different demographic groups.

    Returns:
        The standard deviation of the metric across groups, representing the Degree of Bias (DoB).
    """
    return series.std(ddof=0)


def compute_fairness_discrimination_rate(
    fmr_series: pd.Series,
    fnmr_series: pd.Series,
    alpha: float,
) -> tuple[float, float, float]:
    """
    Compute Fairness Discrimination Rate (FDR) and its components A and B.

    FDR(τ) = 1 − (α · A + (1−α) · B)
      A = max|FMR_i − FMR_j|  (range of FMR across groups)
      B = max|FNMR_i − FNMR_j|  (range of FNMR across groups)

    FDR=1 means perfectly fair; FDR=0 is maximally biased.

    Args:
        fmr_series: A pandas Series containing the False Match Rates (FMR) for different demographic groups.
        fnmr_series: A pandas Series containing the False Non-Match Rates (FNMR) for different demographic groups.
        alpha: A weighting factor between 0 and 1 that determines the relative importance of FMR and FNMR in the FDR calculation.

    Returns:
        A tuple containing:
            - The Fairness Discrimination Rate (FDR).
            - The range of FMR across groups (A).
            - The range of FNMR across groups (B).
    """

    a = float(fmr_series.max() - fmr_series.min())
    b = float(fnmr_series.max() - fnmr_series.min())
    fdr = 1.0 - (alpha * a + (1.0 - alpha) * b)

    return fdr, a, b
