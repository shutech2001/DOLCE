from __future__ import annotations

from typing import Dict
from numpy.typing import NDArray

from .generate import generate_synthetic_data
from utils.utils import eps_greedy_policy


def calc_true_value(
    num_features: int,
    num_actions: int,
    non_overlap_ratio: float,
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

    The true value is independent of non_overlap_ratio, since overlap affects only logging.
    We keep the argument for API compatibility with existing simulation scripts.
    """
    test_data = generate_synthetic_data(
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
    q = test_data["q"]
    # Target policy uses only current-context component when available.
    q_for_pi = test_data.get("g_x_t_a_t", q)
    pi = eps_greedy_policy(q_for_pi)
    # Value is always with respect to the true reward q.
    return float((q * pi).sum(1).mean())
