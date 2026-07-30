"""
This module provides utility functions for loading and preparing a face recognition model.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from model.iresnet import iresnet34


def get_pre_norm_transform(image_size: int = 112) -> transforms.Compose:
    """
    Returns a torchvision transform that resizes an image to the specified size and converts it to a tensor.
    The image is also scaled to the range [0, 1] and converted to float32. This transform should be applied before normalization.

    Args:
        image_size (int): The size to which the image will be resized. Default is 112.

    Returns:
        transforms.Compose: A composed transform that resizes the image, converts it to a tensor, and scales it to [0, 1].
    """

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )


def get_normalize_transform() -> transforms.Normalize:
    """
    Returns a torchvision transform that normalizes an image tensor to have a mean of 0.5 and a standard deviation of 0.5 for each channel.
    This normalization should be applied after the pre-normalization transform.

    Returns:
        transforms.Normalize: A normalization transform that scales the image tensor to have a mean of 0.5 and a standard deviation of 0.5 for each channel.
    """

    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def load_model(checkpoint_path: Path, device: str) -> nn.Module:
    """
    Loads a face recognition model from the specified checkpoint path and moves it to the specified device.

    Args:
        checkpoint_path (Path): The path to the model checkpoint file.
        device (str): The device to which the model will be moved.

    Returns:
        nn.Module: The loaded face recognition model.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".pt":
        model = torch.jit.load(str(checkpoint_path), map_location=device)
    else:
        model = iresnet34()
        model.load_state_dict(
            torch.load(str(checkpoint_path), weights_only=True, map_location=device)
        )

    model.to(device)
    model.eval()
    return model


def get_device() -> str:
    """
    Returns the best available device for PyTorch (CUDA, MPS, or CPU).

    Returns:
        str: The name of the best available device ("cuda", "mps", or "cpu").
    """

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
