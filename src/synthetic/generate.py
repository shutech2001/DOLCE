from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax  # type: ignore

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
    env_random_state: int = 0,
    logging_eps: float = 0.0,
    x_t_dep: float = 1.0,
    lag_scale: float = 1.0,
) -> Dict[str, NDArray]:
    """
    Generates synthetic contextual bandit data with customizable feature distributions,
    action spaces, and non-overlap support for offline RL/off-policy evaluation.

    The context features at time t are drawn from a normal distribution and may be
    partially dependent on lagged features (at time t-l) depending on x_t_dep.
    Rewards are generated as a combination of functions of x_t, lagged features,
    and an optional interaction term, with support for non-overlap between actions.

    Designed for reproducible experiments by specifying random_state and env_random_state.

    Args:
        num_data (int): Number of data points to generate.
        num_features (int, optional): Number of contextual features. Defaults to 5.
        num_actions (int, optional): Number of possible actions. Defaults to 2.
        non_overlap_ratio (float, optional): Proportion of data where action supports do not overlap. Defaults to 0.5.
        lambda_ (float, optional): Mixture weight for reward as function of current vs. lagged features. Defaults to 0.5.
        eta (float, optional): Weight for u(x_t, x_t_l, a) interaction term in reward. Defaults to 0.0.
        beta (float, optional): Parameter for non-overlap boundary. Defaults to 0.3.
        random_state (int, optional): RNG seed for data generating process. Defaults to 42.
        env_random_state (int, optional): RNG seed for reward function/environment. Defaults to 0.
        logging_eps (float, optional): Exploration parameter for logging policy. Defaults to 0.0.
        x_t_dep (float, optional): Correlation between lagged and current features (0: independent, 1: identical). Defaults to 1.0.
        lag_scale (float, optional): Standard deviation multiplier for lagged features. Defaults to 1.0.

    Raises:
        ValueError: If num_features < 1.
        ValueError: If num_actions < 2.

    Returns:
        Dict[str, NDArray]: Dictionary containing generated features, actions, rewards,
              logged propensities, and other metadata for offline RL evaluation.
    """
    if num_features < 1:
        raise ValueError("num_features must be a positive integer.")
    if num_actions < 2:
        raise ValueError("num_actions must be an integer >= 2.")

    rng_data: np.random.Generator = np.random.default_rng(int(random_state))
    rng_env: np.random.Generator = np.random.default_rng(int(env_random_state))

    # features at time t-l
    x_t_l: NDArray = rng_data.normal(size=(num_data, num_features))
    # features at time t based on features at time t-l (x_t_dep=0 -> independent)
    x_t: NDArray = rng_data.normal(loc=x_t_dep * x_t_l, scale=3, size=(num_data, num_features))
    # Make the first feature independent of x_{t-l} so support violation can
    # depend on X_t alone without inducing lag-support violations.
    x_t[:, 0] = rng_data.normal(scale=3, size=num_data)

    # define reward function (q = lambda_*g(x_t,a) + (1-lambda_)*h(x_t_l,a) + eta*u(x_t,x_t_l,a))
    g_x_t_a_t: NDArray = np.zeros((num_data, num_actions))
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

    large_count: NDArray = np.sum(x_t[:, 1 : num_features - 1] > 0.5, axis=1)  # noqa: E203
    g_x_t_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.7, 1.3)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        g_x_t_a_t[:, a] += np.where(large_count >= 2, 1, 0)

    # lag component h(x_{t-l}, a)
    h_x_t_l_a_t: NDArray = np.zeros((num_data, num_actions))
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

    large_count: NDArray = np.sum(x_t_l[:, 1 : num_features - 1] > 0.5, axis=1)  # noqa: E203
    h_x_t_l_a_t[:, 0] += np.where(large_count >= 2, -0.7, 0)
    for a in range(1, num_actions):
        reward_if_action = rng_env.uniform(0.7, 1.3)
        reward_if_not_action = rng_env.uniform(-0.1, 0.1)
        h_x_t_l_a_t[:, a] += np.where(large_count >= 2, 1, 0)

    # optional interaction term between current and lag features (eta=0 keeps additive structure)
    u_x_t_x_t_l_a_t: NDArray = np.zeros((num_data, num_actions))
    if eta != 0.0:
        base = x_t[:, 0] * x_t_l[:, 0]
        if num_features > 1:
            base += 0.5 * x_t[:, 1] * x_t_l[:, 1]
        if num_features > 2:
            base += 0.25 * np.sin(x_t[:, 2] - x_t_l[:, 2])
        for a in range(num_actions):
            coef = rng_env.uniform(0.3, 0.7) * (1.0 if a % 2 == 0 else -1.0)
            u_x_t_x_t_l_a_t[:, a] = coef * np.tanh(base)

    q: NDArray = lambda_ * g_x_t_a_t + (1 - lambda_) * (lag_scale * h_x_t_l_a_t) + eta * u_x_t_x_t_l_a_t

    # logging policy depends only on current context (current-action sufficiency)
    pi_0: NDArray = softmax(beta * g_x_t_a_t, axis=1)
    if logging_eps > 0.0:
        pi_0 = (1.0 - logging_eps) * pi_0 + logging_eps / num_actions
    a_t: NDArray = sample_action(pi_0, random_state * 2)  # data randomness

    # support violation (deterministic no-intervention)
    # Use X_t[:,0] only (current-action sufficiency), and keep it independent of X_{t-l}.
    percentile: NDArray = np.percentile(x_t[:, 0], (1 - non_overlap_ratio) * 100)
    idx: NDArray = x_t[:, 0] > percentile
    a_t[idx] = 0
    pi_0[idx, 0] = 1.0
    pi_0[idx, 1:] = 0.0

    # reward noise uses data rng
    q_fact: NDArray = q[np.arange(num_data), a_t]
    r: NDArray = rng_data.normal(loc=q_fact, scale=1.0)

    return dict(
        num_data=num_data,
        num_features=num_features,
        num_actions=num_actions,
        non_overlap_ratio=non_overlap_ratio,
        eta=eta,
        lag_scale=lag_scale,
        x_t=x_t,
        x_t_l=x_t_l,
        a_t=a_t,
        r=r,
        pi_0=pi_0,
        g_x_t_a_t=lambda_ * g_x_t_a_t,
        h_x_t_l_a_t=(1 - lambda_) * (lag_scale * h_x_t_l_a_t),
        u_x_t_x_t_l_a_t=eta * u_x_t_x_t_l_a_t,
        q=q,
    )
