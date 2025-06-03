from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import rankdata  # type: ignore


def eps_greedy_policy(
    q_func: NDArray,
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


def aggregate_simulation_results(
    estimated_policy_value_list: list,
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> pd.DataFrame:
    """aggregate experimental results such as mean squared error, squared bias, and variance—based"""
    result_df = (
        pd.DataFrame(pd.DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value
    sample_mean = pd.DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (
            policy_value - mean_estimates
        ) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (
            estimates - mean_estimates
        ) ** 2

    return result_df
