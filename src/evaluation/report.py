"""
Evaluation report generation.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping

import pandas as pd

from common.perturbations import Perturbation
from common.config import OVERALL_GROUP_KEY
from evaluation.metrics import Metrics
from evaluation.metrics import (
    compute_degree_of_bias,
    compute_fairness_discrimination_rate,
)


@dataclass
class VerificationRecord:
    """
    A record of verification metrics for a specific perturbation, parameter, and group.
    """

    perturbation: str
    """The perturbation name."""
    param: str
    """The perturbation parameter value."""
    group: str
    """The group name."""

    EER: float
    """The Equal Error Rate (EER) value."""
    EER_threshold: float
    """The threshold at which the EER occurs."""

    ZeroFMR_FMR: float
    """The False Match Rate (FMR) at ZeroFMR threshold."""
    ZeroFMR_FNMR: float
    """The False Non-Match Rate (FNMR) at ZeroFMR threshold."""
    ZeroFMR_threshold: float
    """The threshold at which the ZeroFMR occurs."""

    FMR100_FMR: float
    """The False Match Rate (FMR) at FMR100 threshold."""
    FMR100_FNMR: float
    """The False Non-Match Rate (FNMR) at FMR100 threshold."""
    FMR100_threshold: float
    """The threshold at which the FMR100 occurs."""

    FMR1000_FMR: float
    """The False Match Rate (FMR) at FMR1000 threshold."""
    FMR1000_FNMR: float
    """The False Non-Match Rate (FNMR) at FMR1000 threshold."""
    FMR1000_threshold: float
    """The threshold at which the FMR1000 occurs."""


@dataclass
class FairnessRecord:
    """
    A record of fairness metrics for a specific perturbation and parameter.
    Contains FDR and DoB computed across demographic groups, one entry per operating point.
    """

    perturbation: str
    """The perturbation name."""
    param: str
    """The perturbation parameter value."""

    EER_FDR: float
    """The Fairness Discrimination Rate (FDR) for EER."""
    EER_FDR_A: float
    """The A component of the Fairness Discrimination Rate (FDR) for EER."""
    EER_FDR_B: float
    """The B component of the Fairness Discrimination Rate (FDR) for EER."""
    EER_DoB: float
    """The Degree of Bias (DoB) for EER."""

    ZeroFMR_FDR: float
    """The Fairness Discrimination Rate (FDR) for ZeroFMR."""
    ZeroFMR_FDR_A: float
    """The A component of the Fairness Discrimination Rate (FDR) for ZeroFMR."""
    ZeroFMR_FDR_B: float
    """The B component of the Fairness Discrimination Rate (FDR) for ZeroFMR."""
    ZeroFMR_DoB: float
    """The Degree of Bias (DoB) for ZeroFMR."""

    FMR100_FDR: float
    """The Fairness Discrimination Rate (FDR) for FMR100."""
    FMR100_FDR_A: float
    """The A component of the Fairness Discrimination Rate (FDR) for FMR100."""
    FMR100_FDR_B: float
    """The B component of the Fairness Discrimination Rate (FDR) for FMR100."""
    FMR100_DoB: float
    """The Degree of Bias (DoB) for FMR100."""

    FMR1000_FDR: float
    """The Fairness Discrimination Rate (FDR) for FMR1000."""
    FMR1000_FDR_A: float
    """The A component of the Fairness Discrimination Rate (FDR) for FMR1000."""
    FMR1000_FDR_B: float
    """The B component of the Fairness Discrimination Rate (FDR) for FMR1000."""
    FMR1000_DoB: float
    """The Degree of Bias (DoB) for FMR1000."""


def create_verification_records(
    perturbations: list[Perturbation],
    perturbation_metrics: Mapping[str, Mapping[str, Metrics]],
) -> list[VerificationRecord]:
    """
    Create a list of VerificationRecord instances from the given perturbations and their corresponding metrics.
    Each record corresponds to a specific perturbation, parameter, and group.

    Args:
        perturbations (list[Perturbation]): A list of Perturbation instances.
        perturbation_metrics (Mapping[str, Mapping[str, Metrics]]): A mapping from perturbation paths to group names and their corresponding Metrics instances.+

    Returns:
        list[VerificationRecord]: A list of VerificationRecord instances.
    """

    records: list[VerificationRecord] = []
    for pert in perturbations:
        for group, metrics in perturbation_metrics[pert.path].items():
            records.append(
                VerificationRecord(
                    perturbation=pert.name,
                    param=pert.display_param,
                    group=group,
                    **{f.name: getattr(metrics, f.name) for f in fields(Metrics)},
                )
            )
    return records


def build_verification_report(records: list[VerificationRecord]) -> pd.DataFrame:
    """
    Build a verification report DataFrame from a list of VerificationRecord instances.

    Args:
        records (list[VerificationRecord]): A list of VerificationRecord instances.

    Returns:
        pd.DataFrame: A DataFrame containing the verification report, sorted by perturbation, parameter, and group.
    """
    return pd.DataFrame(asdict(r) for r in records)


def build_fairness_report(
    verification_records: list[VerificationRecord],
    fdr_alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Build a fairness report DataFrame from a list of VerificationRecord instances.

    Args:
        verification_records (list[VerificationRecord]): A list of VerificationRecord instances.
        fdr_alpha (float): The alpha value for computing the Fairness Discrimination Rate (FDR). Default is 0.5.

    Returns:
        pd.DataFrame: A DataFrame containing the fairness report, sorted by perturbation and parameter.
    """

    races_records: dict[tuple[str, str], list[VerificationRecord]] = {}
    for record in verification_records:
        if record.group == OVERALL_GROUP_KEY:
            continue

        races_records.setdefault((record.perturbation, record.param), []).append(record)

    fairness_records: list[FairnessRecord] = []

    for (perturbation, param), group in races_records.items():
        eer_fmr = pd.Series([r.EER for r in group])
        eer_fnmr = eer_fmr  # EER is symmetric, so FMR and FNMR are the same
        eer_fdr, eer_a, eer_b = compute_fairness_discrimination_rate(
            eer_fmr,
            eer_fnmr,
            alpha=fdr_alpha,
        )
        eer_dob = compute_degree_of_bias(eer_fnmr)

        zero_fmr = pd.Series([r.ZeroFMR_FMR for r in group])
        zero_fnmr = pd.Series([r.ZeroFMR_FNMR for r in group])
        zero_fdr, zero_a, zero_b = compute_fairness_discrimination_rate(
            zero_fmr,
            zero_fnmr,
            alpha=fdr_alpha,
        )
        zero_dob = compute_degree_of_bias(zero_fnmr)

        fmr100_fmr = pd.Series([r.FMR100_FMR for r in group])
        fmr100_fnmr = pd.Series([r.FMR100_FNMR for r in group])
        fmr100_fdr, fmr100_a, fmr100_b = compute_fairness_discrimination_rate(
            fmr100_fmr,
            fmr100_fnmr,
            alpha=fdr_alpha,
        )
        fmr100_dob = compute_degree_of_bias(fmr100_fnmr)

        fmr1000_fmr = pd.Series([r.FMR1000_FMR for r in group])
        fmr1000_fnmr = pd.Series([r.FMR1000_FNMR for r in group])
        fmr1000_fdr, fmr1000_a, fmr1000_b = compute_fairness_discrimination_rate(
            fmr1000_fmr,
            fmr1000_fnmr,
            alpha=fdr_alpha,
        )
        fmr1000_dob = compute_degree_of_bias(fmr1000_fnmr)

        fairness_records.append(
            FairnessRecord(
                perturbation=perturbation,
                param=param,
                EER_FDR=eer_fdr,
                EER_FDR_A=eer_a,
                EER_FDR_B=eer_b,
                EER_DoB=eer_dob,
                ZeroFMR_FDR=zero_fdr,
                ZeroFMR_FDR_A=zero_a,
                ZeroFMR_FDR_B=zero_b,
                ZeroFMR_DoB=zero_dob,
                FMR100_FDR=fmr100_fdr,
                FMR100_FDR_A=fmr100_a,
                FMR100_FDR_B=fmr100_b,
                FMR100_DoB=fmr100_dob,
                FMR1000_FDR=fmr1000_fdr,
                FMR1000_FDR_A=fmr1000_a,
                FMR1000_FDR_B=fmr1000_b,
                FMR1000_DoB=fmr1000_dob,
            )
        )

    return pd.DataFrame(asdict(r) for r in fairness_records)


def save_report(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save the verification report DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The verification report DataFrame.
        out_path (Path): The output file path.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
