from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata  # type: ignore


def parse_comma_separated_list(raw: str) -> List:
    """Parse a comma-separated list of floats.

    Args:
        raw (str): A string containing a comma-separated list of floats.

    Raises:
        ValueError: If the list of values is empty.

    Returns:
        List[float]: A list of floats parsed from the input string.
    """
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        values.append(value)
    if not values:
        raise ValueError("values are empty")
    return values


def eps_greedy_policy(
    q_func: NDArray,
    k: int = 1,
    eps: float = 0.1,
) -> NDArray:
    """Define an epsilon-greedy policy over actions.

    Args:
        q_func (NDArray): A 2D array of shape (n_samples, n_actions) representing the Q-values.
        k (int, optional): The number of top actions to consider. Defaults to 1.
        eps (float, optional): The exploration parameter. Defaults to 0.1.

    Returns:
        NDArray: A 2D array of shape (n_samples, n_actions) representing the policy.
    """
    is_topk: NDArray = rankdata(-q_func, method="ordinal", axis=1) <= k
    pi: NDArray = ((1.0 - eps) / k) * is_topk + eps / q_func.shape[1]
    return pi / pi.sum(1)[:, np.newaxis]


def sample_action(pi: NDArray, random_state: int = 42) -> NDArray:
    """Sample an action from a categorical policy π for each row.

    Args:
        pi (NDArray): A 2D array of shape (n_samples, n_actions) representing the policy.
        random_state (int, optional): The random state. Defaults to 42.

    Returns:
        NDArray: A 1D array of shape (n_samples,) representing the sampled actions.
    """
    rng = np.random.default_rng(int(random_state))
    uniform_rvs: NDArray = rng.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi: NDArray = pi.cumsum(axis=1)
    flg: NDArray = cum_pi > uniform_rvs
    actions: NDArray = flg.argmax(axis=1)
    return actions
