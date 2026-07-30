"""
Default configurations for the project.
"""

from pathlib import Path
from typing import Literal

ROOT = Path(__file__).parent.parent.parent

SEED = 42
DEFAULT_CHECKPOINT = ROOT / "data/195520backbone.pth"
DEFAULT_DATASET = ROOT / "data/RFW_test_pairs"
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_RACES: list[Literal["African", "Asian", "Caucasian", "Indian"]] = [
    "African",
    "Asian",
    "Caucasian",
    "Indian",
]

OVERALL_GROUP_KEY: str = "overall"
ORIGINAL_PERTURBATION_KEY: str = "original"
