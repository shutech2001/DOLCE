from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax  # type: ignore

from utils.utils import sample_action


def generate_synthetic_data(
    num_data: int,
    num_features: int = 5,
    num_actions: int = 5,
    non_overlap_ratio: float = 0.5,
    lambda_: float = 0.5,
    eta: float = 0.0,
    beta: float = 0.3,
    random_state: int = 42,
    *,
    # env_random_state controls *environment parameters* (reward function coefficients)
    # data_random_state controls *data sampling* (x's, a's, and reward noise)
    # If you do not specify them, we keep the old behavior (env and data both tied to
    # random_state) for backward compatibility.
    env_random_state: Optional[int] = None,
    data_random_state: Optional[int] = None,
    x_noise_scale: float = 3.0,
    reward_noise_scale: float = 1.0,
) -> Dict:
    """Generate synthetic data for off-policy evaluation/learning

    Args:
        num_data (int): number of data
        num_features (int, optional): number of features. Defaults to 5.
        num_actions (int, optional): number of actions. Defaults to 5.
        non_overlap_ratio (float, optional): non-overlap ratio. Defaults to 0.5.
        lambda_ (float, optional): lambda. Defaults to 0.5.
        eta (float, optional): eta. Defaults to 0.0.
        beta (float, optional): beta. Defaults to 0.3.
        random_state (int, optional): random state. Defaults to 42.

    Raises:
        ValueError: num_features must be a positive integer.
        ValueError: num_actions must be an integer greater than or equal to 2.

    Returns:
        Dict: synthetic data
    """
    if num_features < 1:
        raise ValueError("num_features must be a positive integer.")
    if num_actions < 2:
        raise ValueError("num_actions must be an integer greater than or equal to 2.")
    if env_random_state is None:
        env_random_state = random_state
    if data_random_state is None:
        data_random_state = random_state

    rng_env = np.random.RandomState(env_random_state)
    rng_data = np.random.RandomState(data_random_state)

    # features at time t-l
    x_t_l: NDArray = rng_data.normal(size=(num_data, num_features))
    # features at time t based on features at time t-l
    x_t: NDArray = rng_data.normal(
        loc=x_t_l,
        scale=x_noise_scale,
        size=(num_data, num_features),
    )

    # define reward function
    # q = lambda_*g_x_t_a_t + (1-lambda_)*h_x_t_l_a_t + eta*u_x_t_x_t_l_a_t
    # define g(x_t, a_t)
    g_x_t_a_t: NDArray = np.zeros((num_data, num_actions))
    # effect of x_t_1
    g_x_t_a_t[:, 0] += np.where(x_t[:, 0] > 0.5, -0.2, 0.2)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.4, 0.9)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        g_x_t_a_t[:, a] += np.where(x_t[:, 0] > 0.5, reward_if_action, reward_if_not_action)
    # effect of x_t_i (num_features > i > 1)
    for x in range(1, num_features - 1):
        g_x_t_a_t[:, 0] += np.where(x_t[:, x] > 0.5, -0.2, 0.2)
        for a in range(1, num_actions):
            reward_if_action = rng_env.uniform(0.4, 0.9)
            reward_if_not_action = rng_env.uniform(-0.1, 0.1)
            g_x_t_a_t[:, a] += np.where(x_t[:, x] > 0.5, reward_if_action, reward_if_not_action)
    # add more rewards if act when two or more of x_t_i (num_features > i > 1) are greater
    large_count: NDArray = np.sum(x_t[:, 1 : num_features - 1] > 0.5, axis=1)  # noqa: E203
    g_x_t_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.7, 1.3)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        g_x_t_a_t[:, a] += np.where(large_count >= 2, 1, 0)

    # define h(x_{t_l}, a_t)
    h_x_t_l_a_t: NDArray = np.zeros((num_data, num_actions))
    # effect of x_{t-l}_1
    h_x_t_l_a_t[:, 0] += np.where(x_t_l[:, 0] > 0.5, -0.2, 0.2)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.4, 0.9)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        h_x_t_l_a_t[:, a] += np.where(x_t_l[:, 0] > 0.5, reward_if_action, reward_if_not_action)
    # effect of x_{t-l}_i (num_features > i > 1)
    for x in range(1, num_features - 1):
        h_x_t_l_a_t[:, 0] += np.where(x_t_l[:, x] > 0.5, -0.2, 0.2)
        for a in range(1, num_actions):
            reward_if_action = rng_env.uniform(0.4, 0.9)
            reward_if_not_action = rng_env.uniform(-0.1, 0.1)
            h_x_t_l_a_t[:, a] += np.where(x_t_l[:, x] > 0.5, reward_if_action, reward_if_not_action)
    # add more rewards if act when two or more of x_{t_l}_i (num_features > i > 1) are greater
    large_count: NDArray = np.sum(x_t_l[:, 1 : num_features - 1] > 0.5, axis=1)  # noqa: E203
    h_x_t_l_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.7, 1.3)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        h_x_t_l_a_t[:, a] += np.where(large_count >= 2, 1, 0)
    u_x_t_x_t_l_a_t: NDArray = np.zeros((num_data, num_actions))
    base = x_t[:, 0] * x_t_l[:, 0]
    if num_features > 1:
        base += 0.5 * x_t[:, 1] * x_t_l[:, 1]
    if num_features > 2:
        base += 0.25 * np.sin(x_t[:, 2] - x_t_l[:, 2])
    for a in range(num_actions):
        coef = rng_env.uniform(0.3, 0.7) * (1.0 if a % 2 == 0 else -1.0)
        u_x_t_x_t_l_a_t[:, a] = coef * np.tanh(base)

    q = lambda_ * g_x_t_a_t + (1 - lambda_) * h_x_t_l_a_t + eta * u_x_t_x_t_l_a_t

    # define logging policy using current context only
    pi_0: NDArray = softmax(beta * g_x_t_a_t, axis=1)
    # Use the data RNG for action sampling so that (env_random_state, data_random_state)
    # cleanly separates environment parameters vs sampling randomness.
    a_t: NDArray = sample_action(pi_0, random_state * 2, rng=rng_data)

    # For a certain feature, it is assumed that there is no intervention when the condition is met.
    percentile = np.percentile(x_t[:, 0], (1 - non_overlap_ratio) * 100)
    indices_to_non_intervention: NDArray = x_t[:, 0] > percentile
    a_t[indices_to_non_intervention] = 0
    # Update pi_0 for the rows to be fully deterministic.
    pi_0[indices_to_non_intervention, :] = 0.0
    pi_0[indices_to_non_intervention, 0] = 1.0

    # define reward
    q_fact: NDArray = q[np.arange(num_data), a_t]
    # Reward noise should use the data RNG as well (not the env RNG).
    r: NDArray = rng_data.normal(loc=q_fact, scale=reward_noise_scale)

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
