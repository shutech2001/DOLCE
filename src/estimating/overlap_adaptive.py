from __future__ import annotations

from numpy.typing import NDArray
import numpy as np


def unsupported_mass(pi: NDArray, pi_0: NDArray, eps: float = 1e-8) -> float:
    """Estimate the target mass on unsupported actions under the logging policy.

    Args:
        pi (NDArray): The logged policy.
        pi_0 (NDArray): The true policy.
        eps (float, optional): The epsilon value. Defaults to 1e-8.

    Returns:
        float: The unsupported mass.
    """
    unsupported = pi_0 <= eps
    return float(np.mean((pi * unsupported).sum(axis=1)))


def adaptive_clip(
    unsupported_mass_value: float,
    clip_min: float = 20.0,
    clip_max: float = 100.0,
    low: float = 0.01,
    high: float = 0.1,
) -> float:
    """Linearly interpolate clipping threshold based on unsupported mass.

    Args:
        unsupported_mass_value (float): The unsupported mass.
        clip_min (float, optional): The minimum clip value. Defaults to 20.0.
        clip_max (float, optional): The maximum clip value. Defaults to 100.0.
        low (float, optional): The low value. Defaults to 0.01.
        high (float, optional): The high value. Defaults to 0.1.

    Returns:
        float: The clipped value.
    """
    if unsupported_mass_value <= low:
        return clip_min
    if unsupported_mass_value >= high:
        return clip_max
    t = (unsupported_mass_value - low) / (high - low)
    return float(clip_min + t * (clip_max - clip_min))
