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
    *,
    env_random_state: int | None = None,
    data_random_state: int | None = None,
    eps: float = 0.1,
) -> float:
    """Calculate the true value of the policy value using synthetic data

    Args:
        num_features (int): number of features
        num_actions (int): number of actions
        non_overlap_ratio (float): non-overlap ratio
        lambda_ (float): lambda
        eta (float, optional): eta. Defaults to 0.0.

    Returns:
        float: true value of the policy value
    """
    test_data: Dict = generate_synthetic_data(
        num_data=1000000,
        num_features=num_features,
        num_actions=num_actions,
        non_overlap_ratio=non_overlap_ratio,
        lambda_=lambda_,
        eta=eta,
        env_random_state=env_random_state,
        data_random_state=data_random_state,
    )
    q: NDArray = test_data["q"]
    pi: NDArray = eps_greedy_policy(q, eps=eps)
    return (q * pi).sum(1).mean()
