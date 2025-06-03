from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity


class PolicyValueGivenLagFeaturesEstimator:
    """class of calculation pi(a_t | x_{t-l}) = int_{x_t} p(x_t | x_{t-l}) pi(a_t | x_t)"""

    def __init__(self, kernel: str = "gaussian", bandwidth: float = 0.5):
        """define kde parameters and initialize kde"""
        self.kernel: str = kernel
        self.bandwidth: float = bandwidth
        self.joint_kde: KernelDensity | None = None
        self.marginal_kde: KernelDensity | None = None

    def estimate_conditional_prob(
        self, x_t_l: NDArray, x_t: NDArray
    ) -> NDArray:
        """estimate conditional probability p(x_t | x_{t-l}) by kernel density estimation"""
        combined_data = np.column_stack([x_t_l, x_t])
        self.joint_kde = KernelDensity(
            kernel=self.kernel, bandwidth=self.bandwidth
        ).fit(combined_data)
        self.marginal_kde = KernelDensity(
            kernel=self.kernel, bandwidth=self.bandwidth
        ).fit(x_t_l)

        combined_data = np.column_stack([x_t_l, x_t])
        log_joint_prob = self.joint_kde.score_samples(combined_data)

        log_marginal_prob = self.marginal_kde.score_samples(x_t_l)

        log_conditional_prob = log_joint_prob - log_marginal_prob

        return np.exp(log_conditional_prob)

    def mc_int_cond_prob_times_pi(
        self,
        x_t_l: NDArray,
        x_t: NDArray,
        pi: NDArray,
        action_indices: NDArray,
    ) -> float:
        """calculate monte carlo integration value of p(x_t | x_{t-l}) * pi(a_t | x_t)"""
        conditional_probs = self.estimate_conditional_prob(x_t_l, x_t)
        selected_probs = pi[np.arange(len(action_indices)), action_indices]
        integration_value = conditional_probs * selected_probs

        return np.mean(integration_value)
