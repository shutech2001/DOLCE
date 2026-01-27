from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from utils.utils import sample_action


def generate_synthetic_data(
    num_data: int,
    num_features: int = 5,
    num_actions: int = 2,
    non_overlap_ratio: float = 0.5,
    lambda_: float = 0.5,
    eta: float = 0.0,
    beta: float = 0.3,
    random_state: int = 42,
    env_random_state: int = 0,  # ★追加：環境固定用
    logging_eps: float = 0.0,
    x_t_dep: float = 1.0,
) -> dict:
    """Generate synthetic data for off-policy evaluation with separated RNG.

    random_state      : data randomness (X, A sampling, reward noise)
    env_random_state  : environment randomness (reward-function coefficients)
    eta               : strength of x_t/x_t_l interaction term (eta=0 keeps additive structure)
    logging_eps       : exploration floor for the logging policy (before support violation)
    x_t_dep           : dependence strength of x_t on x_{t-l} (0 = independent)
    """
    if num_features < 1:
        raise ValueError("num_features must be a positive integer.")
    if num_actions < 2:
        raise ValueError("num_actions must be an integer >= 2.")

    rng_data = np.random.default_rng(int(random_state))
    rng_env = np.random.default_rng(int(env_random_state))

    # features at time t-l
    x_t_l: NDArray = rng_data.normal(size=(num_data, num_features))
    # features at time t based on features at time t-l (x_t_dep=0 -> independent)
    x_t: NDArray = rng_data.normal(loc=x_t_dep * x_t_l, scale=3, size=(num_data, num_features))

    # define reward function (q = lambda_*g(x_t,a) + (1-lambda_)*h(x_t_l,a) + eta*u(x_t,x_t_l,a))
    g_x_t_a_t = np.zeros((num_data, num_actions))
    g_x_t_a_t[:, 0] += np.where(x_t[:, 0] > 0.5, -0.2, 0.2)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.4, 0.9)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        g_x_t_a_t[:, a] += np.where(x_t[:, 0] > 0.5, reward_if_action, reward_if_not_action)

    for x in range(1, num_features - 1):
        g_x_t_a_t[:, 0] += np.where(x_t[:, x] > 0.5, -0.2, 0.2)
        for a in range(1, num_actions):
            reward_if_action = rng_env.uniform(0.4, 0.9)
            reward_if_not_action = rng_env.uniform(-0.1, 0.1)
            g_x_t_a_t[:, a] += np.where(x_t[:, x] > 0.5, reward_if_action, reward_if_not_action)

    large_count = np.sum(x_t[:, 1 : num_features - 1] > 0.5, axis=1)
    g_x_t_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.7, 1.3)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        g_x_t_a_t[:, a] += np.where(large_count >= 2, 1, 0)

    # lag component h(x_{t-l}, a)
    h_x_t_l_a_t = np.zeros((num_data, num_actions))
    h_x_t_l_a_t[:, 0] += np.where(x_t_l[:, 0] > 0.5, -0.2, 0.2)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.4, 0.9)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        h_x_t_l_a_t[:, a] += np.where(x_t_l[:, 0] > 0.5, reward_if_action, reward_if_not_action)

    for x in range(1, num_features - 1):
        h_x_t_l_a_t[:, 0] += np.where(x_t_l[:, x] > 0.5, -0.2, 0.2)
        for a in range(1, num_actions):
            reward_if_action = rng_env.uniform(0.4, 0.9)
            reward_if_not_action = rng_env.uniform(-0.1, 0.1)
            h_x_t_l_a_t[:, a] += np.where(x_t_l[:, x] > 0.5, reward_if_action, reward_if_not_action)

    large_count = np.sum(x_t_l[:, 1 : num_features - 1] > 0.5, axis=1)
    h_x_t_l_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.7, 1.3)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        h_x_t_l_a_t[:, a] += np.where(large_count >= 2, 1, 0)

    # optional interaction term between current and lag features (eta=0 keeps additive structure)
    u_x_t_x_t_l_a_t = np.zeros((num_data, num_actions))
    if eta != 0.0:
        base = x_t[:, 0] * x_t_l[:, 0]
        if num_features > 1:
            base += 0.5 * x_t[:, 1] * x_t_l[:, 1]
        if num_features > 2:
            base += 0.25 * np.sin(x_t[:, 2] - x_t_l[:, 2])
        for a in range(num_actions):
            coef = rng_env.uniform(0.3, 0.7) * (1.0 if a % 2 == 0 else -1.0)
            u_x_t_x_t_l_a_t[:, a] = coef * np.tanh(base)

    q = lambda_ * g_x_t_a_t + (1 - lambda_) * h_x_t_l_a_t + eta * u_x_t_x_t_l_a_t

    # logging policy depends only on current context (current-action sufficiency)
    pi_0 = softmax(beta * g_x_t_a_t, axis=1)
    if logging_eps > 0.0:
        pi_0 = (1.0 - logging_eps) * pi_0 + logging_eps / num_actions
    a_t = sample_action(pi_0, random_state * 2)  # data randomness

    # support violation (deterministic no-intervention for high x_t[:,0])
    percentile = np.percentile(x_t[:, 0], (1 - non_overlap_ratio) * 100)
    idx = x_t[:, 0] > percentile
    a_t[idx] = 0
    pi_0[idx, 0] = 1.0
    pi_0[idx, 1:] = 0.0

    # reward noise uses data rng
    q_fact = q[np.arange(num_data), a_t]
    r = rng_data.normal(loc=q_fact, scale=1.0)

    return dict(
        num_data=num_data,
        num_features=num_features,
        num_actions=num_actions,
        non_overlap_ratio=non_overlap_ratio,
        eta=eta,
        x_t=x_t,
        x_t_l=x_t_l,
        a_t=a_t,
        r=r,
        pi_0=pi_0,
        g_x_t_a_t=lambda_ * g_x_t_a_t,
        h_x_t_l_a_t=(1 - lambda_) * h_x_t_l_a_t,
        u_x_t_x_t_l_a_t=eta * u_x_t_x_t_l_a_t,
        q=q,
    )
