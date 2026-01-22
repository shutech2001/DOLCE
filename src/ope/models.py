from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import numpy as np
from numpy.typing import NDArray

from estimating.lag_policy import LaggedPolicyEstimatorNumpy
from estimating.reward import train_reward_model_mtri_crossfit, _make_folds


def calc_dm(
    pi: NDArray,
    q_hat: NDArray,
    return_contributions: bool = False,
) -> NDArray | float:
    """Direct Method.

    Args:
        pi (NDArray): policy
        q_hat (NDArray): estimated reward
        return_contributions (bool, optional): whether to return contributions. Defaults to False.

    Returns:
        NDArray | float: estimated policy value
    """
    contributions: NDArray = (q_hat * pi).sum(1)
    if return_contributions:
        return contributions
    return float(contributions.mean())


def calc_ips(
    dataset: Dict,
    pi: NDArray,
    max_value: float | None = 100,
    return_contributions: bool = False,
) -> NDArray | float:
    """Inverse Propensity Score.

    Args:
        dataset (Dict): dataset
        pi (NDArray): policy
        max_value (float, optional): maximum value. Defaults to 100.
        return_contributions (bool, optional): whether to return contributions. Defaults to False.

    Returns:
        NDArray | float: estimated policy value
    """
    num_data: int = dataset["num_data"]
    actions: NDArray = dataset["a_t"]
    rewards: NDArray = dataset["r"]
    pi_0: NDArray = dataset["pi_0"]

    idx = np.arange(num_data)
    w: NDArray = pi[idx, actions] / pi_0[idx, actions]

    contributions: NDArray = w * rewards
    if return_contributions:
        return contributions
    estimate: float = float(contributions.mean())
    if max_value is None:
        return estimate
    return float(np.minimum(estimate, max_value))


def calc_dr(
    dataset: Dict,
    pi: NDArray,
    q_hat: NDArray,
    max_value: float | None = 100,
    return_contributions: bool = False,
) -> NDArray | float:
    """Doubly Robust.

    Args:
        dataset (Dict): dataset
        pi (NDArray): policy
        q_hat (NDArray): estimated reward
        max_value (float, optional): maximum value. Defaults to 100.
        return_contributions (bool, optional): whether to return contributions. Defaults to False.

    Returns:
        NDArray | float: estimated policy value
    """
    num_data: int = dataset["num_data"]
    actions: NDArray = dataset["a_t"]
    rewards: NDArray = dataset["r"]
    pi_0: NDArray = dataset["pi_0"]

    idx: NDArray = np.arange(num_data)
    w: NDArray = pi[idx, actions] / pi_0[idx, actions]

    contributions: NDArray = (q_hat * pi).sum(1)
    contributions += w * (rewards - q_hat[idx, actions])

    if return_contributions:
        return contributions
    estimate: float = float(contributions.mean())
    if max_value is None:
        return estimate
    return float(np.minimum(estimate, max_value))


def calc_dolce(
    dataset: Dict,
    pi: NDArray,
    max_value: float | None = 100,
    num_folds: int = 2,
    tau: float = 0.1,
    bandwidth: float = 1.0,
    lambda_mtri: float = 1.0,
    weight_clip: float = 100.0,
    random_state: int = 42,
    eps: float = 1e-6,
    return_contributions: bool = False,
) -> NDArray | float:
    """DOLCE: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects

    Args:
        dataset (Dict): dataset
        pi (NDArray): policy
        max_value (float, optional): maximum value. Defaults to 100.
        num_folds (int, optional): number of folds. Defaults to 2.
        tau (float, optional): tau. Defaults to 0.1.
        bandwidth (float, optional): bandwidth. Defaults to 1.0.
        lambda_mtri (float, optional): lambda for MTRI. Defaults to 1.0.
        weight_clip (float, optional): weight clip. Defaults to 100.0.
        random_state (int, optional): random state. Defaults to 42.
        eps (float, optional): epsilon. Defaults to 1e-6.
        return_contributions (bool, optional): whether to return contributions. Defaults to False.

    Returns:
        NDArray | float: estimated policy value
    """
    num_data: int = dataset["num_data"]
    actions: NDArray = dataset["a_t"]
    rewards: NDArray = dataset["r"]

    lag_features: NDArray | list[NDArray] = dataset.get("x_t_ls")
    if lag_features is None:
        lag_features_list: list[NDArray] = [dataset["x_t_l"]]
    elif isinstance(lag_features, list):
        lag_features_list: list[NDArray] = lag_features
    else:
        lag_features_list: list[NDArray] = [lag_features[i] for i in range(lag_features.shape[0])]

    folds: list[NDArray] = _make_folds(num_data=num_data, num_folds=num_folds, random_state=random_state)
    estimator = LaggedPolicyEstimatorNumpy(bandwidth=bandwidth, eps=eps)

    q_hat_list: Deque[NDArray] = deque()
    lag_alc: Deque[float] = deque()
    bar_pi_0_list: Deque[NDArray] = deque()

    for lag_features in lag_features_list:
        dataset_lag: Dict = dict(dataset)
        dataset_lag["x_t_l"] = lag_features
        q_hat, alc, _ = train_reward_model_mtri_crossfit(
            dataset=dataset_lag,
            num_folds=num_folds,
            lambda_mtri=lambda_mtri,
            random_state=random_state,
            folds=folds,
        )
        q_hat_list.append(q_hat)
        lag_alc.append(alc)

        bar_pi_0_hat = np.zeros((num_data, dataset["num_actions"]))
        for fold_idx in folds:
            if num_folds <= 1:
                train_idx = fold_idx
            else:
                mask = np.ones(num_data, dtype=bool)
                mask[fold_idx] = False
                train_idx = np.where(mask)[0]

            bar_pi_0_hat[fold_idx] = estimator.estimate_bar_pi_from_actions(
                lag_features[fold_idx],
                lag_features[train_idx],
                actions[train_idx],
                dataset["num_actions"],
            )
        bar_pi_0_hat = np.clip(bar_pi_0_hat, eps, None)
        bar_pi_0_hat = bar_pi_0_hat / bar_pi_0_hat.sum(axis=1, keepdims=True)
        bar_pi_0_list.append(bar_pi_0_hat)

    lag_contributions: Deque[NDArray] = deque()
    for lag_idx, lag_features in enumerate(lag_features_list):
        lag_contrib = np.zeros(num_data)
        q_hat = q_hat_list[lag_idx]
        bar_pi_0_hat = bar_pi_0_list[lag_idx]
        for fold_idx in folds:
            if num_folds <= 1:
                train_idx = fold_idx
            else:
                mask = np.ones(num_data, dtype=bool)
                mask[fold_idx] = False
                train_idx = np.where(mask)[0]

            bar_pi_theta = estimator.estimate_bar_pi(
                lag_features[fold_idx],
                lag_features[train_idx],
                pi[train_idx],
            )
            w = np.clip(bar_pi_theta / bar_pi_0_hat[fold_idx], 0.0, weight_clip)

            idx = np.arange(fold_idx.shape[0])
            q_hat_test = q_hat[fold_idx]
            q_hat_factual = q_hat_test[idx, actions[fold_idx]]
            w_factual = w[idx, actions[fold_idx]]

            fold_contrib = (q_hat_test * pi[fold_idx]).sum(1)
            fold_contrib += w_factual * (rewards[fold_idx] - q_hat_factual)
            lag_contrib[fold_idx] = fold_contrib
        lag_contributions.append(lag_contrib)

    alc_arr = np.array(lag_alc)
    weights = np.exp(-alc_arr / max(tau, eps))
    weights /= weights.sum()
    contributions = np.zeros(num_data)
    for weight, lag_contrib in zip(weights, lag_contributions):
        contributions += weight * lag_contrib

    if return_contributions:
        return contributions
    estimate = float(contributions.mean())
    if max_value is None:
        return estimate
    return float(np.minimum(estimate, max_value))
