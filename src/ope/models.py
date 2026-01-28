from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import numpy as np
from numpy.typing import NDArray

from estimating.lag_policy import LaggedPolicyEstimator
from estimating.reward import (
    RIAdditiveRewardConfig,
    estimate_alc_knn,
    fit_predict_by_MLP_actionwise,
    fit_predict_by_MLP_for_all_actions,
    train_predict_ri_additive_crossfit,
    train_reward_model_mtri_crossfit,
)


def _make_folds(n: int, n_folds: int, seed: int) -> List[NDArray[np.int64]]:
    if n_folds <= 1:
        return [np.arange(n, dtype=np.int64)]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return [np.sort(block).astype(np.int64) for block in np.array_split(perm, n_folds)]


def _get_q_hat(
    dataset: dict,
    reward_model: str,
    q_hat: Optional[NDArray],
    reward_cfg: Optional[RIAdditiveRewardConfig],
    random_state: int,
) -> Tuple[NDArray, Dict]:
    if q_hat is not None:
        return q_hat, {"reward_model": "provided"}

    if reward_model == "mlp":
        q_hat = fit_predict_by_MLP_for_all_actions(
            features=dataset["x_t"],
            actions=dataset["a_t"],
            rewards=dataset["r"],
            num_actions=dataset["num_actions"],
            random_state=random_state,
        )
        return q_hat, {"reward_model": "mlp"}

    if reward_model == "actionwise_mlp":
        q_hat = fit_predict_by_MLP_actionwise(
            features=dataset["x_t"],
            actions=dataset["a_t"],
            rewards=dataset["r"],
            num_actions=dataset["num_actions"],
            random_state=random_state,
        )
        return q_hat, {"reward_model": "actionwise_mlp"}

    if reward_model == "ri_additive":
        if reward_cfg is None:
            reward_cfg = RIAdditiveRewardConfig()
        q_hat, info_q = train_predict_ri_additive_crossfit(
            x=dataset["x_t"],
            x_lag=dataset["x_t_l"],
            a=dataset["a_t"],
            r=dataset["r"],
            num_actions=dataset["num_actions"],
            cfg=reward_cfg,
        )
        return q_hat, {"reward_model": "ri_additive", **info_q}

    if reward_model == "mtri":
        q_hat, alc, _ = train_reward_model_mtri_crossfit(
            dataset=dataset,
            num_folds=2,
            lambda_mtri=1.0,
            random_state=random_state,
        )
        return q_hat, {"reward_model": "mtri", "alc": float(alc)}

    raise ValueError(f"Unknown reward_model: {reward_model}")


def calc_dm(
    dataset: dict,
    pi: NDArray,
    q_hat: Optional[NDArray] = None,
    reward_model: str = "mlp",
    reward_cfg: Optional[RIAdditiveRewardConfig] = None,
    random_state: int = 42,
) -> Tuple[NDArray, Dict]:
    """Direct Method (DM)."""
    q_hat, info_q = _get_q_hat(dataset, reward_model, q_hat, reward_cfg, random_state)
    contrib = (pi * q_hat).sum(axis=1)

    q_hat_fact = q_hat[np.arange(dataset["x_t"].shape[0]), dataset["a_t"]]
    residual = dataset["r"] - q_hat_fact
    alc_proxy = estimate_alc_knn(residual=residual, x_lag=dataset["x_t_l"], a=dataset["a_t"])

    info = {**info_q, "alc_proxy": alc_proxy}
    return contrib, info


def calc_ips(dataset: dict, pi: NDArray) -> Tuple[NDArray, Dict]:
    """Inverse Propensity Scoring (IPS)."""
    a = dataset["a_t"]
    r = dataset["r"]
    pi0 = dataset["pi_0"]
    iw = pi[np.arange(a.shape[0]), a] / pi0[np.arange(a.shape[0]), a]
    contrib = iw * r
    info = {"iw_min": float(np.min(iw)), "iw_max": float(np.max(iw)), "iw_mean": float(np.mean(iw))}
    return contrib, info


def calc_dr(
    dataset: dict,
    pi: NDArray,
    q_hat: Optional[NDArray] = None,
    reward_model: str = "mlp",
    reward_cfg: Optional[RIAdditiveRewardConfig] = None,
    random_state: int = 42,
) -> Tuple[NDArray, Dict]:
    """Doubly Robust (DR)."""
    q_hat, info_q = _get_q_hat(dataset, reward_model, q_hat, reward_cfg, random_state)

    a = dataset["a_t"]
    r = dataset["r"]
    pi0 = dataset["pi_0"]

    iw = pi[np.arange(a.shape[0]), a] / pi0[np.arange(a.shape[0]), a]
    q_hat_fact = q_hat[np.arange(a.shape[0]), a]
    contrib = iw * (r - q_hat_fact) + (pi * q_hat).sum(axis=1)

    residual = r - q_hat_fact
    alc_proxy = estimate_alc_knn(residual=residual, x_lag=dataset["x_t_l"], a=a)

    info = {
        **info_q,
        "alc_proxy": alc_proxy,
        "iw_min": float(np.min(iw)),
        "iw_max": float(np.max(iw)),
        "iw_mean": float(np.mean(iw)),
    }
    return contrib, info


def calc_dolce(
    dataset: dict,
    pi: NDArray,
    num_folds: int = 2,
    tau: float = 0.1,
    bandwidth: float = 1.0,
    lambda_mtri: float = 1.0,
    weight_clip: Optional[float] = None,
    random_state: int = 42,
    eps: float = 1e-6,
    *,
    reward_model: str = "ri_additive",
    reward_cfg: Optional[RIAdditiveRewardConfig] = None,
) -> Tuple[NDArray, Dict]:
    """DOLCE (single-lag) with kernel lag policies and residual-invariance reward model."""
    num_data = dataset["num_data"]
    actions = dataset["a_t"]
    rewards = dataset["r"]
    pi_0 = dataset.get("pi_0")
    lag_features = dataset.get("x_t_ls")
    if lag_features is None:
        lag_features_list = [dataset["x_t_l"]]
    elif isinstance(lag_features, list):
        lag_features_list = lag_features
    else:
        lag_features_list = [lag_features[i] for i in range(lag_features.shape[0])]

    folds = _make_folds(n=num_data, n_folds=num_folds, seed=random_state)
    estimator = LaggedPolicyEstimator(bandwidth=bandwidth, eps=eps)

    q_hat_list = []
    lag_alc = []
    bar_pi_0_list = []

    for lag in lag_features_list:
        if reward_model == "ri_additive":
            if reward_cfg is None:
                cfg = RIAdditiveRewardConfig(n_folds=num_folds, seed=random_state)
            else:
                cfg_dict = {**reward_cfg.__dict__}
                cfg_dict["n_folds"] = num_folds
                cfg_dict["seed"] = random_state
                cfg = RIAdditiveRewardConfig(**cfg_dict)
            q_hat, _ = train_predict_ri_additive_crossfit(
                x=dataset["x_t"],
                x_lag=lag,
                a=actions,
                r=rewards,
                num_actions=dataset["num_actions"],
                cfg=cfg,
                folds=folds,
            )
            q_hat_fact = q_hat[np.arange(num_data), actions]
            alc = estimate_alc_knn(residual=rewards - q_hat_fact, x_lag=lag, a=actions)
        elif reward_model == "mtri":
            dataset_lag = dict(dataset)
            dataset_lag["x_t_l"] = lag
            q_hat, alc, _ = train_reward_model_mtri_crossfit(
                dataset=dataset_lag,
                num_folds=num_folds,
                lambda_mtri=lambda_mtri,
                random_state=random_state,
                folds=folds,
            )
        else:
            raise ValueError(f"Unknown reward_model for DOLCE: {reward_model}")

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

            if pi_0 is not None:
                bar_pi_0_hat[fold_idx] = estimator.estimate_bar_pi(
                    lag[fold_idx],
                    lag[train_idx],
                    pi_0[train_idx],
                )
            else:
                bar_pi_0_hat[fold_idx] = estimator.estimate_bar_pi_from_actions(
                    lag[fold_idx],
                    lag[train_idx],
                    actions[train_idx],
                    dataset["num_actions"],
                )
        bar_pi_0_hat = np.clip(bar_pi_0_hat, eps, None)
        bar_pi_0_hat = bar_pi_0_hat / bar_pi_0_hat.sum(axis=1, keepdims=True)
        bar_pi_0_list.append(bar_pi_0_hat)

    lag_contributions = []
    for lag_idx, lag in enumerate(lag_features_list):
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
                lag[fold_idx],
                lag[train_idx],
                pi[train_idx],
            )
            w = bar_pi_theta / bar_pi_0_hat[fold_idx]
            if weight_clip is not None:
                w = np.clip(w, 0.0, weight_clip)

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
    weights = weights / weights.sum()
    contributions = np.zeros(num_data)
    for weight, lag_contrib in zip(weights, lag_contributions):
        contributions += weight * lag_contrib

    info = {
        "alc_values": alc_arr,
        "lag_weights": weights,
        "lag_weight_min": float(np.min(weights)),
        "lag_weight_max": float(np.max(weights)),
        "reward_model": reward_model,
    }
    return contributions, info
