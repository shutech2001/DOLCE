from __future__ import annotations

from typing import Dict
from numpy.typing import NDArray

from .generate import generate_synthetic_data
from utils.utils import eps_greedy_policy


def calc_true_value(
    num_features: int,
    num_actions: int,
    lambda_: float,
    eta: float = 0.0,
    beta: float = 0.3,
    random_state: int = 42,
    env_random_state: int = 0,
    num_mc: int = 100000,
    logging_eps: float = 0.0,
    x_t_dep: float = 1.0,
    lag_scale: float = 1.0,
) -> float:
    """Calculate the true value of the epsilon-greedy policy (Monte Carlo).

    Args:
        num_features (int): Number of features.
        num_actions (int): Number of actions.
        lambda_ (float): Mixture weight for reward as function of current vs. lagged features.
        eta (float, optional): Weight for u(x_t, x_t_l, a) interaction term in reward. Defaults to 0.0.
        beta (float, optional): Parameter for non-overlap boundary. Defaults to 0.3.
        random_state (int, optional): RNG seed for data generating process. Defaults to 42.
        env_random_state (int, optional): RNG seed for reward function/environment. Defaults to 0.
        num_mc (int, optional): Number of Monte Carlo samples. Defaults to 100000.
        logging_eps (float, optional): Exploration parameter for logging policy. Defaults to 0.0.
        x_t_dep (float, optional):
            Correlation between lagged and current features (0: independent, 1: identical). Defaults to 1.0.
        lag_scale (float, optional): Standard deviation multiplier for lagged features. Defaults to 1.0.

    Returns:
        float: The true value of the epsilon-greedy policy.
    """
    test_data: Dict = generate_synthetic_data(
        num_data=num_mc,
        num_features=num_features,
        num_actions=num_actions,
        non_overlap_ratio=0.0,
        lambda_=lambda_,
        eta=eta,
        beta=beta,
        random_state=random_state,
        env_random_state=env_random_state,
        logging_eps=logging_eps,
        x_t_dep=x_t_dep,
        lag_scale=lag_scale,
    )
    q: NDArray = test_data["q"]
    # Target policy uses only current-context component when available.
    q_for_pi: NDArray = test_data.get("g_x_t_a_t", q)
    pi: NDArray = eps_greedy_policy(q_for_pi)
    # Value is always with respect to the true reward q.
    return float((q * pi).sum(1).mean())
