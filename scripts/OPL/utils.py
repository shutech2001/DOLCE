from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata  # type: ignore


def eps_greedy_policy(
    q_func: np.ndarray,
    k: int = 1,
    eps: float = 0.1,
) -> NDArray:
    """Define the policy using epsilon-greedy"""
    is_topk = rankdata(-q_func, method="ordinal", axis=1) <= k
    pi = ((1.0 - eps) / k) * is_topk + eps / q_func.shape[1]

    return pi / pi.sum(1)[:, np.newaxis]


def sample_action(pi: NDArray, random_state: int = 42) -> NDArray:
    """Sample action from the policy"""
    np.random.seed(random_state)
    uniform_rvs: NDArray = np.random.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi: NDArray = pi.cumsum(axis=1)
    flg: NDArray = (
        cum_pi > uniform_rvs
    )  # if cumulated probability is greater than random value
    actions: NDArray = flg.argmax(axis=1)
    return actions
