from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from utils import sample_action, eps_greedy_policy


def generate_synthetic_data(
    num_data: int,
    num_features: int = 5,
    num_actions: int = 2,
    non_overlap_ratio: float = 0.5,
    lambda_: float = 0.5,
    beta: float = 0.3,
    random_state: int = 42,
) -> dict:
    if num_features < 1:
        raise ValueError("num_features must be a positive integer.")
    if num_actions < 2:
        raise ValueError("num_actions must be an integer greater than or equal to 2.")
    np.random.seed(random_state)
    # features at time t-l
    x_t_l: NDArray = np.random.normal(size=(num_data, num_features))
    # features at time t based on features at time t-l
    x_t: NDArray = np.random.normal(
        loc=x_t_l, scale=3, size=(num_data, num_features)
    )

    # define reward function (q = lambda_*g_x_t_a_t + (1-lambda_)*h_x_t_l_a_t)
    # define g(x_t, a_t)
    g_x_t_a_t = np.zeros((num_data, num_actions))
    # effect of x_t_1
    g_x_t_a_t[:, 0] += np.where(x_t[:, 0] > 0.5, -0.2, 0.2)
    for a in range(1, num_actions):
        reward_if_action = np.random.uniform(0.4, 0.9)
        reward_if_not_action = np.random.uniform(-0.1, 0.1)
        g_x_t_a_t[:, a] += np.where(
            x_t[:, 0] > 0.5, reward_if_action, reward_if_not_action
        )
    # effect of x_t_i (num_features > i > 1)
    for x in range(1, num_features - 1):
        g_x_t_a_t[:, 0] += np.where(x_t[:, x] > 0.5, -0.2, 0.2)
        for a in range(1, num_actions):
            reward_if_action = np.random.uniform(0.4, 0.9)
            reward_if_not_action = np.random.uniform(-0.1, 0.1)
            g_x_t_a_t[:, a] += np.where(
                x_t[:, x] > 0.5, reward_if_action, reward_if_not_action
            )
    # add more rewards if act when two or more of x_t_i (num_features > i > 1) are greater
    large_count = np.sum(x_t[:, 1:num_features - 1] > 0.5, axis=1)
    g_x_t_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = np.random.uniform(0.7, 1.3)
        reward_if_not_action = np.random.uniform(-0.1, 0.1)
        g_x_t_a_t[:, a] += np.where(large_count >= 2, 1, 0)

    # define h(x_{t_l}, a_t)
    h_x_t_l_a_t = np.zeros((num_data, num_actions))
    # effect of x_{t-l}_1
    h_x_t_l_a_t[:, 0] += np.where(x_t_l[:, 0] > 0.5, -0.2, 0.2)
    for a in range(1, num_actions):
        reward_if_action = np.random.uniform(0.4, 0.9)
        reward_if_not_action = np.random.uniform(-0.1, 0.1)
        h_x_t_l_a_t[:, a] += np.where(
            x_t_l[:, 0] > 0.5, reward_if_action, reward_if_not_action
        )
    # effect of x_{t-l}_i (num_features > i > 1)
    for x in range(1, num_features - 1):
        h_x_t_l_a_t[:, 0] += np.where(x_t_l[:, x] > 0.5, -0.2, 0.2)
        for a in range(1, num_actions):
            reward_if_action = np.random.uniform(0.4, 0.9)
            reward_if_not_action = np.random.uniform(-0.1, 0.1)
            h_x_t_l_a_t[:, a] += np.where(
                x_t_l[:, x] > 0.5, reward_if_action, reward_if_not_action
            )
    # add more rewards if act when two or more of x_{t_l}_i (num_features > i > 1) are greater
    large_count = np.sum(x_t_l[:, 1:num_features - 1] > 0.5, axis=1)
    h_x_t_l_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = np.random.uniform(0.7, 1.3)
        reward_if_not_action = np.random.uniform(-0.1, 0.1)
        h_x_t_l_a_t[:, a] += np.where(large_count >= 2, 1, 0)
    q = lambda_ * g_x_t_a_t + (1 - lambda_) * h_x_t_l_a_t

    # define logging policy and sample action
    pi_0 = softmax(beta * q, axis=1)
    a_t = sample_action(pi_0, random_state * 2)

    # For a certain feature, it is assumed that there is no intervention when the condition is met.
    percentile = np.percentile(x_t[:, 0], (1 - non_overlap_ratio) * 100)
    indices_to_non_intervention = x_t[:, 0] > percentile
    a_t[indices_to_non_intervention] = 0
    # Update pi_0 for the rows
    pi_0[indices_to_non_intervention, 0] = 1
    pi_0[indices_to_non_intervention, 1] = 0

    # define reward
    q_fact = q[np.arange(num_data), a_t]
    r = np.random.normal(q_fact)

    return dict(
        num_data=num_data,
        num_features=num_features,
        num_actions=num_actions,
        non_overlap_ratio=non_overlap_ratio,
        x_t=x_t,
        x_t_l=x_t_l,
        a_t=a_t,
        r=r,
        pi_0=pi_0,
        g_x_t_a_t=lambda_ * g_x_t_a_t,
        h_x_t_l_a_t=(1 - lambda_) * h_x_t_l_a_t,
        q=q,
    )


def calc_true_value(
    num_features,
    num_actions,
    non_overlap_ratio,
    lambda_,
) -> float:
    test_data = generate_synthetic_data(
        num_data=10000,
        num_features=num_features,
        num_actions=num_actions,
        non_overlap_ratio=non_overlap_ratio,
        lambda_=lambda_,
    )
    q = test_data["q"]
    pi = eps_greedy_policy(q)
    return (q * pi).sum(1).mean()
