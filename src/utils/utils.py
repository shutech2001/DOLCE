from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import rankdata  # type: ignore


def parse_comma_separated_list(raw: str) -> List:
    """Parse a comma-separated list of floats."""
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
    """Define an epsilon-greedy policy over actions."""
    is_topk: NDArray = rankdata(-q_func, method="ordinal", axis=1) <= k
    pi: NDArray = ((1.0 - eps) / k) * is_topk + eps / q_func.shape[1]
    return pi / pi.sum(1)[:, np.newaxis]


def sample_action(pi: NDArray, random_state: int = 42) -> NDArray:
    """
    Sample an action from a categorical policy π for each row.

    IMPORTANT: this function must NOT call np.random.seed(), because that mutates global RNG state
    and silently changes randomness elsewhere (e.g., reward noise), which breaks reproducibility
    and can create artificial dependencies.
    """
    rng = np.random.default_rng(int(random_state))
    uniform_rvs: NDArray = rng.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi: NDArray = pi.cumsum(axis=1)
    flg: NDArray = cum_pi > uniform_rvs
    actions: NDArray = flg.argmax(axis=1)
    return actions


def aggregate_simulation_results(
    estimated_policy_value_list: List[float],
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> pd.DataFrame:
    """Aggregate simulation results across Monte-Carlo runs."""
    result_df: pd.DataFrame = (
        pd.DataFrame(pd.DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0.0
    result_df["variance"] = 0.0
    result_df["true_value"] = policy_value

    sample_mean: pd.DataFrame = pd.DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates: NDArray = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates: NDArray = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (policy_value - mean_estimates) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (estimates - mean_estimates) ** 2

    return result_df
