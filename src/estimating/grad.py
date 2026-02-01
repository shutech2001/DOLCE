from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch
from torch.types import Tensor

from estimating.reward import _make_folds
from .lag_policy import LaggedPolicyEstimatorTorch
from .reward import RIAdditiveRewardConfig, train_predict_ri_additive_crossfit, estimate_alc_knn
from opl.models import DOLCE
from .overlap_adaptive import adaptive_clip, unsupported_mass
from utils.utils import flatten_grads


def estimate_ips_gradient(
    model: torch.nn.Module,
    x_t: NDArray,
    a_t: NDArray,
    r: NDArray,
    pi_0: NDArray,
    log_eps: float,
) -> NDArray:
    """Estimate the IPS gradient.

    Args:
        model (torch.nn.Module): The model.
        x_t (NDArray): The current features.
        a_t (NDArray): The actions.
        r (NDArray): The rewards.
        pi_0 (NDArray): The reference policy.
        log_eps (float): The log epsilon.

    Returns:
        NDArray: The gradient of the IPS estimator.
    """
    model.zero_grad(set_to_none=True)
    x_t_tensor: Tensor = torch.from_numpy(x_t).float()
    a_t_tensor: Tensor = torch.from_numpy(a_t).long()
    r_tensor: Tensor = torch.from_numpy(r).float()
    pi_0_tensor: Tensor = torch.from_numpy(pi_0).float()
    pi: Tensor = model(x_t_tensor)
    log_prob: Tensor = torch.log(pi + log_eps)
    idx = torch.arange(a_t_tensor.shape[0], dtype=torch.long)
    w: Tensor = (pi[idx, a_t_tensor] / pi_0_tensor[idx, a_t_tensor]).detach()
    objective: Tensor = (w * r_tensor * log_prob[idx, a_t_tensor]).mean()
    objective.backward()
    return flatten_grads(model)


def estimate_dr_gradient(
    model: torch.nn.Module,
    x_t: NDArray,
    a_t: NDArray,
    r: NDArray,
    pi_0: NDArray,
    q_hat: NDArray,
    log_eps: float,
) -> NDArray:
    """Estimate the DR gradient.

    Args:
        model (torch.nn.Module): The model.
        x_t (NDArray): The current features.
        a_t (NDArray): The actions.
        r (NDArray): The rewards.
        pi_0 (NDArray): The reference policy.
        q_hat (NDArray): The estimated rewards.
        log_eps (float): The log epsilon.

    Returns:
        NDArray: The gradient of the DR estimator.
    """
    model.zero_grad(set_to_none=True)
    x_t_tensor: Tensor = torch.from_numpy(x_t).float()
    a_t_tensor: Tensor = torch.from_numpy(a_t).long()
    r_tensor: Tensor = torch.from_numpy(r).float()
    pi_0_tensor: Tensor = torch.from_numpy(pi_0).float()
    q_hat_tensor: Tensor = torch.from_numpy(q_hat).float()
    pi = model(x_t_tensor)
    log_prob: Tensor = torch.log(pi + log_eps)
    idx: Tensor = torch.arange(a_t_tensor.shape[0], dtype=torch.long)
    w: Tensor = (pi[idx, a_t_tensor] / pi_0_tensor[idx, a_t_tensor]).detach()
    q_hat_factual: Tensor = q_hat_tensor[idx, a_t_tensor]
    term1: Tensor = (w * (r_tensor - q_hat_factual) * log_prob[idx, a_t_tensor]).mean()
    term2: Tensor = torch.sum(q_hat_tensor * pi.detach() * log_prob, dim=1).mean()
    objective: Tensor = term1 + term2
    objective.backward()
    return flatten_grads(model)


def estimate_dolce_gradient(
    dolce: DOLCE,
    logged_data: dict,
) -> NDArray:
    """Estimate the DOLCE gradient.

    Args:
        dolce (DOLCE): The DOLCE model.
        logged_data (dict): The logged data.

    Returns:
        NDArray: The gradient of the DOLCE estimator.
    """
    model = dolce.nn_model
    model.zero_grad(set_to_none=True)
    x_t: NDArray = logged_data["x_t"]
    a_t: NDArray = logged_data["a_t"]
    r: NDArray = logged_data["r"]
    num_actions: int = logged_data["num_actions"]

    lag_features: NDArray | list[NDArray] = logged_data.get("x_t_ls")
    if lag_features is None:
        lag_features_list: list[NDArray] = [logged_data["x_t_l"]]
    elif isinstance(lag_features, list):
        lag_features_list = lag_features
    else:
        lag_features_list = [lag_features[i] for i in range(lag_features.shape[0])]

    folds = getattr(dolce, "folds_", None)
    if folds is None:
        folds = _make_folds(x_t.shape[0], dolce.num_folds, dolce.random_state)

    # reward models / ALC
    q_hat_list = getattr(dolce, "q_hat_list_", None)
    alc_values = getattr(dolce, "alc_values_", None)
    if q_hat_list is None or alc_values is None:
        q_hat_list = []
        alc_values = []
        for lag in lag_features_list:
            cfg = RIAdditiveRewardConfig(n_folds=dolce.num_folds, seed=dolce.random_state)
            q_hat_lag, _ = train_predict_ri_additive_crossfit(
                x=logged_data["x_t"],
                x_lag=lag,
                a=logged_data["a_t"],
                r=logged_data["r"],
                num_actions=logged_data["num_actions"],
                cfg=cfg,
                folds=folds,
            )
            q_hat_factual = q_hat_lag[np.arange(logged_data["x_t"].shape[0]), logged_data["a_t"]]
            alc_value = estimate_alc_knn(residual=logged_data["r"] - q_hat_factual, x_lag=lag, a=logged_data["a_t"])
            q_hat_list.append(q_hat_lag)
            alc_values.append(alc_value)

    # bar_pi_0
    bar_pi_0_list = getattr(dolce, "bar_pi_0_list_", None)
    if bar_pi_0_list is None:
        estimator = LaggedPolicyEstimatorTorch(bandwidth=dolce.bandwidth, eps=dolce.log_eps)
        bar_pi_0_list = []
        actions_tensor = torch.from_numpy(a_t).long()
        pi_0 = logged_data.get("pi_0")
        pi_0_tensor = torch.from_numpy(pi_0).float() if pi_0 is not None else None
        for lag in lag_features_list:
            lag_tensor = torch.from_numpy(lag).float()
            bar_pi_0_hat = np.zeros((x_t.shape[0], num_actions))
            for fold_idx in folds:
                if dolce.num_folds <= 1:
                    train_idx = fold_idx
                else:
                    mask = np.ones(x_t.shape[0], dtype=bool)
                    mask[fold_idx] = False
                    train_idx = np.where(mask)[0]
                fold_idx_t = torch.from_numpy(fold_idx).long()
                train_idx_t = torch.from_numpy(train_idx).long()
                if pi_0_tensor is not None:
                    bar_pi_0_hat[fold_idx] = (
                        estimator.estimate_bar_pi(
                            lag_tensor[fold_idx_t],
                            lag_tensor[train_idx_t],
                            pi_0_tensor[train_idx_t],
                        )
                        .detach()
                        .numpy()
                    )
                else:
                    bar_pi_0_hat[fold_idx] = (
                        estimator.estimate_bar_pi_from_actions(
                            lag_tensor[fold_idx_t],
                            lag_tensor[train_idx_t],
                            actions_tensor[train_idx_t],
                            num_actions,
                        )
                        .detach()
                        .numpy()
                    )
            bar_pi_0_hat = np.clip(bar_pi_0_hat, dolce.log_eps, None)
            bar_pi_0_hat = bar_pi_0_hat / bar_pi_0_hat.sum(axis=1, keepdims=True)
            bar_pi_0_list.append(bar_pi_0_hat)

    alc_arr = np.array(alc_values)
    lag_weights = np.exp(-alc_arr / max(dolce.tau, dolce.log_eps))
    lag_weights = lag_weights / lag_weights.sum()

    estimator = LaggedPolicyEstimatorTorch(bandwidth=dolce.bandwidth, eps=dolce.log_eps)
    x_t_tensor = torch.from_numpy(x_t).float()
    a_t_tensor = torch.from_numpy(a_t).long()
    r_tensor = torch.from_numpy(r).float()
    pi_all = model(x_t_tensor)
    log_prob_all = torch.log(pi_all + dolce.log_eps)
    current_policy = pi_all.detach()
    clip_value = dolce.weight_clip
    blend_weight = 1.0
    pi_0 = logged_data.get("pi_0")
    if pi_0 is not None:
        u_mass = unsupported_mass(pi_all.detach().cpu().numpy(), pi_0, eps=dolce.log_eps)
        if getattr(dolce, "blend_dolce", False):
            blend_weight = min(1.0, u_mass / max(getattr(dolce, "blend_threshold", 0.0), dolce.log_eps))
        if clip_value is None:
            clip_value = adaptive_clip(u_mass)

    total = torch.zeros((), dtype=pi_all.dtype)
    n = x_t.shape[0]
    for lag_idx, lag in enumerate(lag_features_list):
        lag_tensor = torch.from_numpy(lag).float()
        q_hat_tensor = torch.from_numpy(q_hat_list[lag_idx]).float()
        bar_pi_0_tensor = torch.from_numpy(bar_pi_0_list[lag_idx]).float()
        lag_contrib = torch.zeros(n, dtype=pi_all.dtype)

        for fold_idx in folds:
            if dolce.num_folds <= 1:
                train_idx = fold_idx
            else:
                mask = np.ones(n, dtype=bool)
                mask[fold_idx] = False
                train_idx = np.where(mask)[0]

            test_idx_t = torch.from_numpy(fold_idx).long()
            train_idx_t = torch.from_numpy(train_idx).long()

            pi_train = pi_all[train_idx_t]
            log_prob_test = log_prob_all[test_idx_t]

            a_test = a_t_tensor[test_idx_t]
            r_test = r_tensor[test_idx_t]

            q_hat_test = q_hat_tensor[test_idx_t]
            q_hat_factual = q_hat_test[torch.arange(test_idx_t.shape[0]), a_test]
            pi_test = current_policy[test_idx_t]
            term = torch.sum(q_hat_test * pi_test * log_prob_test, dim=1)

            bar_pi_theta = estimator.estimate_bar_pi(lag_tensor[test_idx_t], lag_tensor[train_idx_t], pi_train)
            bar_pi_0 = bar_pi_0_tensor[test_idx_t]
            w = (bar_pi_theta / bar_pi_0).detach()
            if clip_value is not None:
                w = torch.clamp(w, max=clip_value)
            log_bar_pi = torch.log(bar_pi_theta + dolce.log_eps)
            log_bar_pi_factual = log_bar_pi[torch.arange(test_idx_t.shape[0]), a_test]

            dolce_term = (
                term + w[torch.arange(test_idx_t.shape[0]), a_test] * (r_test - q_hat_factual) * log_bar_pi_factual
            )
            term = dolce_term
            if (blend_weight < 1.0) and (pi_0 is not None):
                pi_0_test = torch.from_numpy(pi_0[fold_idx]).float().to(pi_test.device)
                pi_0_test = torch.clamp(pi_0_test, min=dolce.log_eps)
                iw = pi_test / pi_0_test
                if clip_value is not None:
                    iw = torch.clamp(iw, max=clip_value)
                iw_factual = iw[torch.arange(test_idx_t.shape[0]), a_test]
                dr_term = (
                    iw_factual * (r_test - q_hat_factual) * log_prob_test[torch.arange(test_idx_t.shape[0]), a_test]
                )
                dr_term += torch.sum(q_hat_test * pi_test * log_prob_test, dim=1)
                term = blend_weight * dolce_term + (1.0 - blend_weight) * dr_term
            lag_contrib[test_idx_t] = term

        total = total + float(lag_weights[lag_idx]) * lag_contrib.mean()

    total.backward()
    return flatten_grads(model)
