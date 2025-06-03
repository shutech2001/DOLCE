from __future__ import annotations

import torch
from torch.types import Tensor
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
        self, x_t_l: Tensor, x_t: Tensor
    ) -> Tensor:
        """estimate conditional probability p(x_t | x_{t-l}) by kernel density estimation"""
        combined_data = torch.cat([x_t_l, x_t], dim=1)

        combined_data_np = combined_data.numpy()
        self.joint_kde = KernelDensity(
            kernel=self.kernel, bandwidth=self.bandwidth
        ).fit(combined_data_np)

        x_t_l_np = x_t_l.numpy()
        self.marginal_kde = KernelDensity(
            kernel=self.kernel, bandwidth=self.bandwidth
        ).fit(x_t_l_np)

        log_joint_prob = torch.from_numpy(
            self.joint_kde.score_samples(combined_data_np)
        ).float()

        log_marginal_prob = torch.from_numpy(
            self.marginal_kde.score_samples(x_t_l_np)
        ).float()

        log_conditional_prob = log_joint_prob - log_marginal_prob

        return torch.exp(log_conditional_prob)

    def mc_int_cond_prob_times_pi(
        self,
        x_t_l: Tensor,
        x_t: Tensor,
        pi: Tensor,
        action_indices: Tensor,
    ) -> Tensor:
        """calculate monte carlo integration value of p(x_t | x_{t-l}) * pi(a_t | x_t)"""
        conditional_probs = self.estimate_conditional_prob(x_t_l, x_t)
        selected_probs = pi[torch.arange(len(action_indices)), action_indices]
        integration_value = conditional_probs * selected_probs

        return integration_value.mean()
