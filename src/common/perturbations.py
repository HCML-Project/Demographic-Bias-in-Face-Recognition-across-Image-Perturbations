"""
Perturbation definitions and builders for image transformations.

Includes the following perturbations:
- Gaussian blur (parameter: σ)
- Gaussian noise (parameter: σ)
- Brightness adjustment (parameter: β)
- Contrast adjustment (parameter: α)
- JPEG compression (parameter: q)
- Motion blur (parameter: k)
"""

import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from kornia.augmentation import RandomSaltAndPepperNoise
from kornia.filters import MotionBlur
from PIL import Image
from torchvision.transforms.v2 import GaussianBlur
from torchvision.transforms.v2.functional import (
    adjust_brightness,
    adjust_contrast,
    to_pil_image,
    to_image,
)

from common.config import ORIGINAL_PERTURBATION_KEY


class GaussianNoise(nn.Module):
    """
    Add Gaussian noise to an image.
    """

    def __init__(self, sigma_255: float):
        """
        Initialize the GaussianNoise module.

        Args:
            sigma_255 (float): Standard deviation of the Gaussian noise in the range [0, 255].
        """
        super().__init__()
        self.sigma = sigma_255 / 255.0

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the input image. The noise is applied independently for each pixel and is the same for all channels.

        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W) with values in the range [0, 1].

        Returns:
            torch.Tensor: Noisy image tensor of the same shape as the input, with values clamped to the range [0, 1].
        """

        noise = torch.randn(img.shape[-2:], device=img.device) * self.sigma
        return (img + noise).clamp(0.0, 1.0)


class AdjustBrightness(nn.Module):
    """
    Adjust the brightness of an image.
    1.0 means no change, <1.0 means darker, >1.0 means brighter.
    0.0 means black.
    2.0 means double the brightness.
    0.5 means half the brightness.
    """

    def __init__(self, beta: float):
        """
        Initialize the AdjustBrightness module.

        Args:
            beta (float): Brightness adjustment factor. 1.0 means no change, <1.0 means darker, >1.0 means brighter.
        """

        super().__init__()
        self.beta = beta

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Adjust the brightness of the input image.

        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W) with values in the range [0, 1].

        Returns:
            torch.Tensor: Brightness-adjusted image tensor of the same shape as the input, with values clamped to the range [0, 1].
        """

        return adjust_brightness(img, self.beta)


class AdjustContrast(nn.Module):
    """
    Adjust the contrast of an image.
    1.0 means no change, <1.0 means lower contrast, >1.0 means higher contrast.
    0.0 means gray image.
    2.0 means double the contrast.
    0.5 means half the contrast.
    """

    def __init__(self, factor: float):
        """
        Initialize the AdjustContrast module.

        Args:
            factor (float): Contrast adjustment factor. 1.0 means no change, <1.0 means lower contrast, >1.0 means higher contrast.
        """

        super().__init__()
        self.factor = factor

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Adjust the contrast of the input image.

        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W) with values in the range [0, 1].

        Returns:
            torch.Tensor: Contrast-adjusted image tensor of the same shape as the input, with values clamped to the range [0, 1].
        """

        return adjust_contrast(img, self.factor)


class JPEGCompression(nn.Module):
    """
    Apply JPEG compression to an image with a specified quality factor.
    100 is the best quality (least compression), lower values mean more compression.
    """

    def __init__(self, quality: int):
        """
        Initialize the JPEGCompression module.

        Args:
            quality (int): JPEG quality factor. 100 is the best quality (least compression), lower values mean more compression.
        """

        super().__init__()
        self.quality = quality

    def compress_single(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compress a single image tensor using JPEG compression.

        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W) with values in the range [0, 1].

        Returns:
            torch.Tensor: JPEG-compressed image tensor of the same shape as the input, with values clamped to the range [0, 1].
        """

        pil = to_pil_image(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return (to_image(Image.open(buf)).float() / 255.0).to(
            dtype=img.dtype, device=img.device
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply JPEG compression to a batch of images.

        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W) with values in the range [0, 1].

        Returns:
            torch.Tensor: JPEG-compressed image tensor of the same shape as the input, with values clamped to the range [0, 1].
        """

        if img.ndim == 3:
            return self.compress_single(img)

        return torch.stack([self.compress_single(x) for x in img])


class HorizontalMotionBlur(nn.Module):
    """
    Apply horizontal motion blur to an image using a specified kernel size.
    """

    def __init__(self, kernel_size: int):
        """
        Initialize the HorizontalMotionBlur module.

        Args:
            kernel_size (int): Size of the motion blur kernel.
        """

        super().__init__()
        self.blur = MotionBlur(kernel_size=kernel_size, angle=0.0, direction=0.0)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply horizontal motion blur to the input image.

        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W) with values in the range [0, 1].

        Returns:
            torch.Tensor: Motion-blurred image tensor of the same shape as the input, with values clamped to the range [0, 1].
        """

        return self.blur(img).clamp(0.0, 1.0)


def build_gaussian_blur(params: dict[str, float]) -> nn.Module:
    """
    Build a Gaussian blur transformation based on the provided parameters.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for the Gaussian blur. Expected key is "σ" for the standard deviation of the Gaussian kernel.

    Returns:
        nn.Module: A PyTorch module that applies Gaussian blur to an input image.
    """
    sigma = float(params["σ"])
    k = 2 * int(np.ceil(2 * sigma)) + 1
    return GaussianBlur(kernel_size=(k, k), sigma=sigma)


def build_gaussian_noise(params: dict[str, float]) -> nn.Module:
    """
    Build a Gaussian noise transformation based on the provided parameters.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for the Gaussian noise. Expected key is "σ" for the standard deviation of the Gaussian noise.

    Returns:
        nn.Module: A PyTorch module that adds Gaussian noise to an input image.
    """

    return GaussianNoise(sigma_255=float(params["σ"]))


def build_brightness(params: dict[str, float]) -> nn.Module:
    """
    Build a brightness adjustment transformation based on the provided parameters.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for brightness adjustment. Expected key is "β" for the brightness factor.

    Returns:
        nn.Module: A PyTorch module that adjusts the brightness of an input image.
    """

    return AdjustBrightness(beta=float(params["β"]))


def build_contrast(params: dict[str, float]) -> nn.Module:
    """
    Build a contrast adjustment transformation based on the provided parameters.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for contrast adjustment. Expected key is "α" for the contrast factor.

    Returns:
        nn.Module: A PyTorch module that adjusts the contrast of an input image.
    """

    return AdjustContrast(factor=float(params["α"]))


def build_jpeg(params: dict[str, float]) -> nn.Module:
    """
    Build a JPEG compression transformation based on the provided parameters.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for JPEG compression. Expected key is "q" for the quality factor.

    Returns:
        nn.Module: A PyTorch module that applies JPEG compression to an input image.
    """

    return JPEGCompression(quality=int(params["q"]))


def build_motion_blur(params: dict[str, float]) -> nn.Module:
    """
    Build a horizontal motion blur transformation based on the provided parameters.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for motion blur. Expected key is "k" for the kernel size.

    Returns:
        nn.Module: A PyTorch module that applies horizontal motion blur to an input image.
    """

    k = int(params["k"])
    if k % 2 == 0:
        k += 1
    return HorizontalMotionBlur(kernel_size=k)


def build_salt_and_pepper(params: dict[str, float]) -> nn.Module:
    """
    Build a salt-and-pepper noise transformation based on the provided parameters.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for salt-and-pepper noise.

    Returns:
        nn.Module: A PyTorch module that adds salt-and-pepper noise to an input image.
    """

    p = float(params["p"])
    return RandomSaltAndPepperNoise(
        amount=(p, p), salt_vs_pepper=(0.5, 0.5), p=1.0, keepdim=True
    )


def build_original(params: dict[str, float]) -> nn.Module:
    """
    Build an identity transformation that returns the original image without any perturbation.

    Args:
        params (dict[str, float]): Dictionary containing the parameters for the original image. This is typically empty as no parameters are needed for the identity transformation.

    Returns:
        nn.Module: A PyTorch module that returns the input image unchanged.
    """

    return nn.Identity()


@dataclass(frozen=True)
class PerturbationDef:
    """
    Definition of a perturbation, including its display name, neutral parameters, parameter names, and builder function.
    """

    display_name: str
    """Display name of the perturbation."""

    neutral: dict[str, float]
    """Neutral parameters for the perturbation, representing the identity transformation."""

    param_names: dict[str, str]
    """Mapping from internal parameter names to display names."""

    builder: Callable[[dict[str, float]], nn.Module]
    """Function that builds the perturbation module based on the provided parameters."""


PERTURBATION_DEFS: dict[str, PerturbationDef] = {
    "gaussian_blur": PerturbationDef(
        "Gaussian Blur",
        neutral={"σ": 0.0},
        param_names={"σ": "sigma"},
        builder=build_gaussian_blur,
    ),
    "gaussian_noise": PerturbationDef(
        "Gaussian Noise",
        neutral={"σ": 0.0},
        param_names={"σ": "sigma"},
        builder=build_gaussian_noise,
    ),
    "brightness": PerturbationDef(
        "Brightness",
        neutral={"β": 1.0},
        param_names={"β": "beta"},
        builder=build_brightness,
    ),
    "S&P": PerturbationDef(
        "Salt & Pepper Noise",
        neutral={"p": 0.0},
        param_names={},
        builder=build_salt_and_pepper,
    ),
    "contrast": PerturbationDef(
        "Contrast",
        neutral={"α": 1.0},
        param_names={"α": "alpha"},
        builder=build_contrast,
    ),
    "jpeg": PerturbationDef(
        "JPEG Compression",
        neutral={"q": 100},
        param_names={"q": "q"},
        builder=build_jpeg,
    ),
    "motion_blur": PerturbationDef(
        "Motion Blur",
        neutral={"k": 1},
        param_names={"k": "k"},
        builder=build_motion_blur,
    ),
    "original": PerturbationDef(
        "Original", neutral={}, param_names={}, builder=build_original
    ),
}


@dataclass
class Perturbation:
    """
    Represents a specific perturbation with its name, parameters, and transformation module.
    """

    name: str
    """Name of the perturbation, corresponding to a key in PERTURBATION_DEFS."""

    params: dict[str, float]
    """Parameters for the perturbation, which may vary from the neutral parameters."""

    transform: nn.Module = field(repr=False)
    """PyTorch module that applies the perturbation to an image."""

    @property
    def display_param(self) -> str:
        """Param string with Greek letters, e.g. 'σ=1.0'."""

        def fmt(v: float) -> str:
            return str(int(v)) if v == int(v) else str(v)

        return ", ".join(f"{k}={fmt(v)}" for k, v in self.params.items())

    @property
    def sort_key(self) -> tuple[float | str, ...]:
        """Numeric sort key for ordering perturbations of the same type."""

        return tuple(self.name) + (
            tuple(
                self.params.get(k, 0.0)
                for k in PERTURBATION_DEFS[self.name].param_names
            )
            or (0.0,)
        )

    @property
    def is_neutral(self) -> bool:
        """Check if the perturbation is neutral (identity)."""

        return PERTURBATION_DEFS[self.name].neutral == self.params

    @property
    def path(self) -> str:
        """ASCII identifier for filenames and directories."""

        param_names = PERTURBATION_DEFS[self.name].param_names
        param = "_".join(f"{param_names.get(k, k)}={v}" for k, v in self.params.items())
        return f"{self.name}_{param}" if param else self.name

    @property
    def display(self) -> str:
        """Human-readable label with Greek letters for plot titles and legends."""

        label = PERTURBATION_DEFS[self.name].display_name
        return f"{label} ({self.display_param})" if self.params else label


def build_perturbation(
    spec: dict[str, str | float | int],
) -> Perturbation:
    """
    Build a Perturbation object from a specification dictionary.

    Args:
        spec (dict[str, str | float | int]): Dictionary containing the perturbation specification.
            Expected keys:
                - "name": Name of the perturbation (must be a key in PERTURBATION_DEFS).
                - Other keys corresponding to the parameters of the perturbation.

    Returns:
        Perturbation: A Perturbation object with the specified name, parameters, and transformation module.
    Raises:
        ValueError: If the perturbation name is unknown (not in PERTURBATION_DEFS).
    """

    name = str(spec["name"])
    params: dict[str, float] = {k: float(v) for k, v in spec.items() if k != "name"}

    if name not in PERTURBATION_DEFS:
        raise ValueError(
            f"Unknown perturbation name '{name}'. Known: {list(PERTURBATION_DEFS.keys())}"
        )

    perturbation = Perturbation(name=name, params=params, transform=nn.Identity())
    if not perturbation.is_neutral:
        perturbation.transform = PERTURBATION_DEFS[name].builder(params)

    return perturbation


def neutral_perturbation(perturbation_name: str) -> Perturbation:
    """Return the neutral (identity) Perturbation for the given perturbation name."""

    return build_perturbation(
        {"name": perturbation_name, **PERTURBATION_DEFS[perturbation_name].neutral}
    )


def load_perturbations(
    config_path: Path | str | None = None,
    include_original: bool = False,
    include_neutrals: bool = False,
) -> list[Perturbation]:
    """
    Load a list of Perturbation objects from a configuration file or JSON string.

    Args:
        config_path (Path | str | None): Path to the JSON configuration file or a JSON string containing the perturbation specifications. If None, a default path is used.
        include_original (bool): Whether to additionally include the original (identity) perturbation in the returned list.
        include_neutrals (bool): Whether to additionally include neutral perturbations for each perturbation type in the returned list.

    Returns:
        list[Perturbation]: A list of Perturbation objects, sorted by name and parameter values. The list may include the original perturbation and neutral perturbations based on the provided flags.
    """

    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "perturbations.json"
        )

    if isinstance(config_path, str) and config_path.lstrip().startswith("["):
        specs = json.loads(config_path)
    else:
        with open(config_path, encoding="utf-8") as f:
            specs = json.loads(f.read())

    perturbations = [build_perturbation(s) for s in specs]
    if include_neutrals:
        seen: set[str] = set()
        prepend = []
        for p in perturbations:
            if p.name != ORIGINAL_PERTURBATION_KEY and p.name not in seen:
                seen.add(p.name)
                prepend.append(neutral_perturbation(p.name))
        perturbations = prepend + perturbations
    if include_original:
        perturbations = [
            neutral_perturbation(ORIGINAL_PERTURBATION_KEY)
        ] + perturbations

    perturbations.sort(key=lambda p: p.sort_key)
    return perturbations
