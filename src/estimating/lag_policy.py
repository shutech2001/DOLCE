from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray

import torch
from torch.types import Tensor
from scipy.spatial.distance import cdist  # type: ignore


@dataclass
class LagPolicyMLPConfig:
    """Config for lag-marginal policy estimation.

    Args:
        n_folds (int, optional): Number of folds for cross-fitting. Defaults to 2.
        hidden_layer_sizes (Tuple[int, ...], optional):
            Number of hidden layers and neurons per layer. Defaults to (64, 64).
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
        validation_fraction (float, optional): Fraction of data for validation. Defaults to 0.2.
        alpha (float, optional): L2 penalty. Defaults to 1e-4.
        random_state (int, optional): Random state. Defaults to 0.
        prob_floor (float, optional): Probability floor. Defaults to 1e-6.
    """

    n_folds: int = 2
    hidden_layer_sizes: Tuple[int, ...] = (64, 64)
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.2
    alpha: float = 1e-4  # L2 penalty
    random_state: int = 0
    prob_floor: float = 1e-6


def _make_folds(n: int, n_folds: int, seed: int) -> List[NDArray[np.int64]]:
    """Make folds for cross-fitting.

    Args:
        n (int): Number of data points.
        n_folds (int): Number of folds.
        seed (int): Random state.

    Returns:
        List[NDArray[np.int64]]: List of fold indices.
    """
    if n_folds <= 1:
        return [np.arange(n, dtype=np.int64)]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return [np.sort(block).astype(np.int64) for block in np.array_split(perm, n_folds)]


def _normalize_folds(
    n: int,
    n_folds: int,
    seed: int,
    folds: Optional[NDArray[np.int64] | List[NDArray[np.int64]]],
) -> List[NDArray[np.int64]]:
    """Normalize folds for cross-fitting.

    Args:
        n (int): Number of data points.
        n_folds (int): Number of folds.
        seed (int): Random state.
        folds (Optional[NDArray[np.int64] | List[NDArray[np.int64]]]): Fold indices.

    Returns:
        List[NDArray[np.int64]]: List of fold indices.
    """
    if folds is None:
        return _make_folds(n, n_folds, seed)
    if isinstance(folds, list):
        return folds
    # treat as fold-id vector of shape (n,)
    fold_list: List[NDArray[np.int64]] = []
    for k in range(int(np.max(folds)) + 1):
        fold_list.append(np.where(folds == k)[0].astype(np.int64))
    return fold_list


class LaggedPolicyEstimator:
    """Kernel regression estimator for lag-marginalized policies (NumPy)."""

    def __init__(self, bandwidth: float = 1.0, eps: float = 1e-6):
        """Initialize the kernel regression estimator.

        Args:
            bandwidth (float, optional): Bandwidth for kernel regression. Defaults to 1.0.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-6.
        """
        self.bandwidth = bandwidth
        self.eps = eps

    def _kernel_weights(self, x_query: NDArray, x_ref: NDArray) -> NDArray:
        """Compute the kernel weights.

        Args:
            x_query (NDArray): Query points.
            x_ref (NDArray): Reference points.

        Returns:
            NDArray: Kernel weights.
        """
        dist_sq = cdist(x_query, x_ref, metric="sqeuclidean")
        return np.exp(-dist_sq / (2.0 * self.bandwidth**2))

    def estimate_bar_pi(self, x_query: NDArray, x_ref: NDArray, pi_ref: NDArray) -> NDArray:
        """Estimate the lag-marginalized policy.

        Args:
            x_query (NDArray): Query points.
            x_ref (NDArray): Reference points.
            pi_ref (NDArray): Reference policy.

        Returns:
            NDArray: Estimated policy.
        """
        weights = self._kernel_weights(x_query, x_ref)
        weights_sum = weights.sum(axis=1, keepdims=True) + self.eps
        bar_pi = weights @ pi_ref / weights_sum
        bar_pi = np.clip(bar_pi, self.eps, None)
        return bar_pi / bar_pi.sum(axis=1, keepdims=True)

    def estimate_bar_pi_from_actions(
        self, x_query: NDArray, x_ref: NDArray, actions_ref: NDArray, num_actions: int
    ) -> NDArray:
        """Estimate the lag-marginalized policy from actions.

        Args:
            x_query (NDArray): Query points.
            x_ref (NDArray): Reference points.
            actions_ref (NDArray): Reference actions.
            num_actions (int): Number of actions.

        Returns:
            NDArray: Estimated policy.
        """
        one_hot = np.eye(num_actions)[actions_ref]
        return self.estimate_bar_pi(x_query, x_ref, one_hot)


class LaggedPolicyEstimatorNumpy(LaggedPolicyEstimator):
    """Alias for the NumPy kernel estimator (kept for API compatibility)."""


class LaggedPolicyEstimatorTorch:
    """Kernel regression estimator for lag-marginalized policies (Torch)."""

    def __init__(self, bandwidth: float = 1.0, eps: float = 1e-6):
        """Initialize the kernel regression estimator.

        Args:
            bandwidth (float, optional): Bandwidth for kernel regression. Defaults to 1.0.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-6.
        """
        self.bandwidth = bandwidth
        self.eps = eps

    def _kernel_weights(self, x_query: Tensor, x_ref: Tensor) -> Tensor:
        """Compute the kernel weights.

        Args:
            x_query (Tensor): Query points.
            x_ref (Tensor): Reference points.

        Returns:
            Tensor: Kernel weights.
        """
        dist = torch.cdist(x_query, x_ref) ** 2
        return torch.exp(-dist / (2.0 * self.bandwidth**2))

    def estimate_bar_pi(self, x_query: Tensor, x_ref: Tensor, pi_ref: Tensor) -> Tensor:
        """Estimate the lag-marginalized policy.

        Args:
            x_query (Tensor): Query points.
            x_ref (Tensor): Reference points.
            pi_ref (Tensor): Reference policy.

        Returns:
            Tensor: Estimated policy.
        """
        weights = self._kernel_weights(x_query, x_ref)
        weights_sum = weights.sum(dim=1, keepdim=True) + self.eps
        bar_pi = weights @ pi_ref / weights_sum
        bar_pi = torch.clamp(bar_pi, min=self.eps)
        return bar_pi / bar_pi.sum(dim=1, keepdim=True)

    def estimate_bar_pi_from_actions(
        self, x_query: Tensor, x_ref: Tensor, actions_ref: Tensor, num_actions: int
    ) -> Tensor:
        """Estimate the lag-marginalized policy from actions.

        Args:
            x_query (Tensor): Query points.
            x_ref (Tensor): Reference points.
            actions_ref (Tensor): Reference actions.
            num_actions (int): Number of actions.

        Returns:
            Tensor: Estimated policy.
        """
        one_hot = torch.nn.functional.one_hot(actions_ref, num_classes=num_actions).float()
        return self.estimate_bar_pi(x_query, x_ref, one_hot)


def compute_lag_weights(bar_pi_theta: NDArray, bar_pi0: NDArray, eps: float = 1e-6) -> NDArray:
    """Compute the lag weights.

    Args:
        bar_pi_theta (NDArray): Estimated policy.
        bar_pi0 (NDArray): Reference policy.
        eps (float, optional): Epsilon for numerical stability. Defaults to 1e-6.

    Returns:
        NDArray: Lag weights.
    """
    denom = np.clip(bar_pi0, eps, None)
    return bar_pi_theta / denom
