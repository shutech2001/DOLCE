from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from estimate_policy_value_given_lag_features import (
    PolicyValueGivenLagFeaturesEstimator,
)
from estimate_reward import train_reward_model


def calc_dm(pi: NDArray, q_hat: NDArray) -> float:
    """Direct Method"""
    return (q_hat * pi).sum(1).mean()


def calc_ips(dataset: dict, pi: NDArray, max_value: float = 100) -> float:
    """Inverse Propensity Score"""
    num_data = dataset["num_data"]
    actions = dataset["a_t"]
    rewards = dataset["r"]
    pi_0 = dataset["pi_0"]

    idx = np.arange(num_data)
    w = pi[idx, actions] / pi_0[idx, actions]

    return np.minimum((w * rewards).mean(), max_value)


def calc_dr(
    dataset: dict, pi: NDArray, q_hat: NDArray, max_value: float = 100
) -> float:
    """Doubly Robust"""
    num_data = dataset["num_data"]
    actions = dataset["a_t"]
    rewards = dataset["r"]
    pi_0 = dataset["pi_0"]

    idx = np.arange(num_data)
    w = pi[idx, actions] / pi_0[idx, actions]

    dr = (q_hat * pi).sum(1)
    dr += w * (rewards - q_hat[idx, actions])

    return np.minimum(dr.mean(), max_value)


def calc_dolce(dataset: dict, pi: NDArray, max_value: float = 100) -> float:
    """Decomposing Off-Policy Evaluation into Lagged and Current Effects"""
    num_data = dataset["num_data"]
    actions = dataset["a_t"]
    rewards = dataset["r"]
    pi_0 = dataset["pi_0"]
    q_hat = train_reward_model(dataset=dataset)

    x_t = dataset["x_t"]
    x_t_l = dataset["x_t_l"]

    # estimate conditional probability
    int_estimator = PolicyValueGivenLagFeaturesEstimator()
    integral_value_e: float = int_estimator.mc_int_cond_prob_times_pi(
        x_t_l=x_t_l, x_t=x_t, pi=pi, action_indices=actions
    )
    integral_value_0: float = int_estimator.mc_int_cond_prob_times_pi(
        x_t_l=x_t_l, x_t=x_t, pi=pi_0, action_indices=actions
    )

    w = integral_value_e / integral_value_0

    dolce = (q_hat * pi).sum(1)
    idx = np.arange(num_data)
    dolce += w * (rewards - q_hat[idx, actions])

    return np.minimum(dolce.mean(), max_value)
