from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
from numpy.typing import NDArray

import torch
from torch.types import Tensor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


@dataclass
class LagPolicyMLPConfig:
    """Config for lag-marginal policy estimation.

    We intentionally use scikit-learn MLPs here because they are:
      - fast to fit for moderate n,
      - stable (built-in early stopping),
      - easy to cross-fit.

    This is adequate for the synthetic experiments, where the aim is to
    estimate conditional expectations:
        bar_pi0(a|x_lag)      = P(A=a | X_lag=x_lag)
        bar_pi_theta(a|x_lag)= E[pi_theta(a|X) | X_lag=x_lag]
    """

    n_folds: int = 2
    hidden_layer_sizes: Tuple[int, ...] = (64, 64)
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.2
    alpha: float = 1e-4  # L2 penalty
    random_state: int = 0
    prob_floor: float = 1e-6


def _make_folds(n: int, n_folds: int, seed: int) -> List[NDArray[np.int64]]:
    if n_folds <= 1:
        return [np.arange(n, dtype=np.int64)]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return [np.sort(block).astype(np.int64) for block in np.array_split(perm, n_folds)]


def _normalize_folds(
    n: int,
    n_folds: int,
    seed: int,
    folds: Optional[NDArray[np.int64] | List[NDArray[np.int64]]],
) -> List[NDArray[np.int64]]:
    if folds is None:
        return _make_folds(n, n_folds, seed)
    if isinstance(folds, list):
        return folds
    # treat as fold-id vector of shape (n,)
    fold_list: List[NDArray[np.int64]] = []
    for k in range(int(np.max(folds)) + 1):
        fold_list.append(np.where(folds == k)[0].astype(np.int64))
    return fold_list


class LaggedPolicyEstimator:
    """Kernel regression estimator for lag-marginalized policies (NumPy)."""

    def __init__(self, bandwidth: float = 1.0, eps: float = 1e-6):
        self.bandwidth = bandwidth
        self.eps = eps

    def _kernel_weights(self, x_query: NDArray, x_ref: NDArray) -> NDArray:
        dist_sq = cdist(x_query, x_ref, metric="sqeuclidean")
        return np.exp(-dist_sq / (2.0 * self.bandwidth**2))

    def estimate_bar_pi(self, x_query: NDArray, x_ref: NDArray, pi_ref: NDArray) -> NDArray:
        weights = self._kernel_weights(x_query, x_ref)
        weights_sum = weights.sum(axis=1, keepdims=True) + self.eps
        bar_pi = weights @ pi_ref / weights_sum
        bar_pi = np.clip(bar_pi, self.eps, None)
        return bar_pi / bar_pi.sum(axis=1, keepdims=True)

    def estimate_bar_pi_from_actions(
        self, x_query: NDArray, x_ref: NDArray, actions_ref: NDArray, num_actions: int
    ) -> NDArray:
        one_hot = np.eye(num_actions)[actions_ref]
        return self.estimate_bar_pi(x_query, x_ref, one_hot)


class LaggedPolicyEstimatorNumpy(LaggedPolicyEstimator):
    """Alias for the NumPy kernel estimator (kept for API compatibility)."""


class LaggedPolicyEstimatorTorch:
    """Kernel regression estimator for lag-marginalized policies (Torch)."""

    def __init__(self, bandwidth: float = 1.0, eps: float = 1e-6):
        self.bandwidth = bandwidth
        self.eps = eps

    def _kernel_weights(self, x_query: Tensor, x_ref: Tensor) -> Tensor:
        dist = torch.cdist(x_query, x_ref) ** 2
        return torch.exp(-dist / (2.0 * self.bandwidth**2))

    def estimate_bar_pi(self, x_query: Tensor, x_ref: Tensor, pi_ref: Tensor) -> Tensor:
        weights = self._kernel_weights(x_query, x_ref)
        weights_sum = weights.sum(dim=1, keepdim=True) + self.eps
        bar_pi = weights @ pi_ref / weights_sum
        bar_pi = torch.clamp(bar_pi, min=self.eps)
        return bar_pi / bar_pi.sum(dim=1, keepdim=True)

    def estimate_bar_pi_from_actions(
        self, x_query: Tensor, x_ref: Tensor, actions_ref: Tensor, num_actions: int
    ) -> Tensor:
        one_hot = torch.nn.functional.one_hot(actions_ref, num_classes=num_actions).float()
        return self.estimate_bar_pi(x_query, x_ref, one_hot)


def crossfit_bar_pi0(
    x_lag: NDArray,
    a: NDArray,
    num_actions: int,
    cfg: LagPolicyMLPConfig,
    folds: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray, Dict]:
    """Cross-fitted estimate of bar{pi}_0(a|x_lag)=P(A=a|X_lag=x_lag)."""
    n = x_lag.shape[0]
    fold_list = _normalize_folds(n, cfg.n_folds, cfg.random_state, folds)

    preds = np.zeros((n, num_actions), dtype=np.float64)

    num_folds = len(fold_list)
    for fold_idx in fold_list:
        if fold_idx.size == 0:
            continue
        if num_folds <= 1:
            train_idx = fold_idx
        else:
            mask = np.ones(n, dtype=bool)
            mask[fold_idx] = False
            train_idx = np.where(mask)[0]

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_lag[train_idx])
        x_te = scaler.transform(x_lag[fold_idx])

        clf = MLPClassifier(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            max_iter=cfg.max_iter,
            early_stopping=cfg.early_stopping,
            validation_fraction=cfg.validation_fraction,
            alpha=cfg.alpha,
            random_state=cfg.random_state + 1000 + k,
        )
        clf.fit(x_tr, a[train_idx])
        prob = clf.predict_proba(x_te)

        # ensure all actions exist in columns
        # sklearn drops missing classes; align to [0..K-1]
        full = np.zeros((x_te.shape[0], num_actions), dtype=np.float64)
        for cls_idx, cls in enumerate(clf.classes_):
            full[:, int(cls)] = prob[:, cls_idx]

        full = np.clip(full, cfg.prob_floor, 1.0)
        full = full / full.sum(axis=1, keepdims=True)
        preds[fold_idx] = full

    info = {
        "bar_pi0_min": float(np.min(preds)),
        "bar_pi0_max": float(np.max(preds)),
        "bar_pi0_mean": float(np.mean(preds)),
    }
    return preds, info


def crossfit_bar_pi_theta(
    x_lag: NDArray,
    pi_theta_x: NDArray,
    num_actions: int,
    cfg: LagPolicyMLPConfig,
    folds: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray, Dict]:
    """Cross-fitted estimate of bar{pi}_theta(a|x_lag)=E[pi_theta(a|X)|X_lag=x_lag].

    We fit a multi-output regressor to map x_lag -> pi_theta(x),
    then project predictions back to the simplex.
    """
    n = x_lag.shape[0]
    fold_list = _normalize_folds(n, cfg.n_folds, cfg.random_state, folds)

    preds = np.zeros((n, num_actions), dtype=np.float64)

    num_folds = len(fold_list)
    for fold_idx in fold_list:
        if fold_idx.size == 0:
            continue
        if num_folds <= 1:
            train_idx = fold_idx
        else:
            mask = np.ones(n, dtype=bool)
            mask[fold_idx] = False
            train_idx = np.where(mask)[0]

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_lag[train_idx])
        x_te = scaler.transform(x_lag[fold_idx])

        reg = MLPRegressor(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            max_iter=cfg.max_iter,
            early_stopping=cfg.early_stopping,
            validation_fraction=cfg.validation_fraction,
            alpha=cfg.alpha,
            random_state=cfg.random_state + 2000 + k,
        )
        reg.fit(x_tr, pi_theta_x[train_idx])
        y_hat = reg.predict(x_te)

        y_hat = np.clip(y_hat, cfg.prob_floor, None)
        y_hat = y_hat / y_hat.sum(axis=1, keepdims=True)
        preds[fold_idx] = y_hat

    info = {
        "bar_pi_theta_min": float(np.min(preds)),
        "bar_pi_theta_max": float(np.max(preds)),
        "bar_pi_theta_mean": float(np.mean(preds)),
    }
    return preds, info


def compute_lag_weights(bar_pi_theta: NDArray, bar_pi0: NDArray, eps: float = 1e-6) -> NDArray:
    denom = np.clip(bar_pi0, eps, None)
    return bar_pi_theta / denom
