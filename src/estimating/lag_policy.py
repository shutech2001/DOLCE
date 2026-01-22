from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
import torch
from torch.types import Tensor


class LaggedPolicyEstimator:
    """Kernel regression estimator for lag-marginalized policies."""

    def __init__(self, bandwidth: float = 1.0, eps: float = 1e-6):
        """Initialize the kernel regression estimator.

        Args:
            bandwidth (float, optional): bandwidth of the kernel. Defaults to 1.0.
            eps (float, optional): epsilon for numerical stability. Defaults to 1e-6.
        """
        self.bandwidth: float = bandwidth
        self.eps: float = eps


class LaggedPolicyEstimatorNumpy(LaggedPolicyEstimator):
    """Kernel regression estimator for lag-marginalized policies using numpy."""

    def __init__(self, bandwidth: float = 1.0, eps: float = 1e-6):
        """Initialize the kernel regression estimator using numpy."""
        super().__init__(bandwidth, eps)

    def _kernel_weights(self, x_query: NDArray, x_ref: NDArray) -> NDArray:
        """Calculate the kernel weights.

        Args:
            x_query (NDArray): query features
            x_ref (NDArray): reference features

        Returns:
            NDArray: kernel weights
        """
        dist_sq: NDArray = cdist(x_query, x_ref, metric="sqeuclidean")
        weights: NDArray = np.exp(-dist_sq / (2.0 * self.bandwidth**2))
        return weights

    def estimate_bar_pi(
        self,
        x_query: NDArray,
        x_ref: NDArray,
        pi_ref: NDArray,
    ) -> NDArray:
        """Estimate the lag-marginalized policy.

        Args:
            x_query (NDArray): query features
            x_ref (NDArray): reference features
            pi_ref (NDArray): reference policy

        Returns:
            NDArray: estimated lag-marginalized policy
        """
        weights: NDArray = self._kernel_weights(x_query, x_ref)
        weights_sum = weights.sum(axis=1, keepdims=True) + self.eps
        bar_pi: NDArray = weights @ pi_ref / weights_sum
        bar_pi: NDArray = np.clip(bar_pi, self.eps, None)
        bar_pi: NDArray = bar_pi / bar_pi.sum(axis=1, keepdims=True)
        return bar_pi

    def estimate_bar_pi_from_actions(
        self,
        x_query: NDArray,
        x_ref: NDArray,
        actions_ref: NDArray,
        num_actions: int,
    ) -> NDArray:
        """Estimate the lag-marginalized policy from actions.

        Args:
            x_query (NDArray): query features
            x_ref (NDArray): reference features
            actions_ref (NDArray): reference actions
            num_actions (int): number of actions

        Returns:
            NDArray: estimated lag-marginalized policy
        """
        one_hot: NDArray = np.eye(num_actions)[actions_ref]
        return self.estimate_bar_pi(x_query, x_ref, one_hot)


class LaggedPolicyEstimatorTorch(LaggedPolicyEstimator):
    """Kernel regression estimator for lag-marginalized policies using torch."""

    def __init__(self, bandwidth: float = 1.0, eps: float = 1e-6):
        """Initialize the kernel regression estimator using torch."""
        super().__init__(bandwidth, eps)

    def _kernel_weights(self, x_query: Tensor, x_ref: Tensor) -> Tensor:
        """Calculate the kernel weights.

        Args:
            x_query (Tensor): query features
            x_ref (Tensor): reference features

        Returns:
            Tensor: kernel weights
        """
        dist: Tensor = torch.cdist(x_query, x_ref) ** 2
        weights: Tensor = torch.exp(-dist / (2.0 * self.bandwidth**2))
        return weights

    def estimate_bar_pi(
        self,
        x_query: Tensor,
        x_ref: Tensor,
        pi_ref: Tensor,
    ) -> Tensor:
        """Estimate the lag-marginalized policy.

        Args:
            x_query (Tensor): query features
            x_ref (Tensor): reference features
            pi_ref (Tensor): reference policy

        Returns:
            Tensor: estimated lag-marginalized policy
        """
        weights: Tensor = self._kernel_weights(x_query, x_ref)
        weights_sum = weights.sum(dim=1, keepdim=True) + self.eps
        bar_pi: Tensor = weights @ pi_ref / weights_sum
        bar_pi: Tensor = torch.clamp(bar_pi, min=self.eps)
        bar_pi: Tensor = bar_pi / bar_pi.sum(dim=1, keepdim=True)
        return bar_pi

    def estimate_bar_pi_from_actions(
        self,
        x_query: Tensor,
        x_ref: Tensor,
        actions_ref: Tensor,
        num_actions: int,
    ) -> Tensor:
        """Estimate the lag-marginalized policy from actions.

        Args:
            x_query (Tensor): query features
            x_ref (Tensor): reference features
            actions_ref (Tensor): reference actions
            num_actions (int): number of actions

        Returns:
            Tensor: estimated lag-marginalized policy
        """
        one_hot: Tensor = torch.nn.functional.one_hot(actions_ref, num_classes=num_actions).float()
        return self.estimate_bar_pi(x_query, x_ref, one_hot)
