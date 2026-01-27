from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial import cKDTree
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _one_hot(a: NDArray, num_actions: int) -> NDArray:
    out = np.zeros((a.shape[0], num_actions), dtype=np.float32)
    out[np.arange(a.shape[0]), a.astype(int)] = 1.0
    return out


class MLPScalar(nn.Module):
    """Small scalar MLP (torch) used for g(x,a)."""

    def __init__(self, in_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class RIAdditiveRewardConfig:
    """Config for the residual-invariance-friendly reward model.

    Model class:
        q_hat(x, x_lag, a) = g_hat(x, a) + h_hat(x_lag)

    Key point:
      * h_hat is intentionally action-independent (same for all actions),
        mirroring the construction in your 'old' simulation code.
      * This makes the remaining reward-model error approximately
        delta(x_lag, a), i.e., residual-invariant w.r.t. current X once (x_lag,a) is fixed.
    """

    n_folds: int = 2

    # g-model (torch, pairwise training)
    g_hidden: Tuple[int, ...] = (64, 64)
    g_lr: float = 1e-2
    g_weight_decay: float = 1e-4
    g_batch_size: int = 256
    g_epochs: int = 120
    g_knn_k: int = 3

    # h-model (sklearn MLPRegressor, fast)
    h_hidden: Tuple[int, ...] = (50, 50)
    h_max_iter: int = 500
    h_alpha: float = 1e-4
    h_early_stopping: bool = True
    h_validation_fraction: float = 0.2
    h_action_specific: bool = True
    h_min_samples: int = 20

    dropout: float = 0.0
    seed: int = 0
    device: str = "cpu"


def _make_folds(n: int, n_folds: int, seed: int) -> List[NDArray[np.int64]]:
    if n_folds <= 1:
        return [np.arange(n, dtype=np.int64)]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return [np.sort(block).astype(np.int64) for block in np.array_split(perm, n_folds)]


def _build_pairs_nn_by_action(
    x_lag: NDArray, a: NDArray, k: int = 1
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """For each sample, pick k-NN within the same action group (in x_lag space)."""
    i_list = []
    j_list = []
    for act in np.unique(a):
        idx = np.where(a == act)[0]
        if idx.size < 2:
            continue
        tree = cKDTree(x_lag[idx])
        k_eff = min(k + 1, idx.size)
        _, nn = tree.query(x_lag[idx], k=k_eff)
        if k_eff <= 1:
            continue
        nn = nn[:, 1:]
        for col in range(nn.shape[1]):
            i_list.append(idx)
            j_list.append(idx[nn[:, col]])
    if len(i_list) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(i_list).astype(np.int64), np.concatenate(j_list).astype(np.int64)


def _train_g_pairwise(
    x: NDArray,
    x_lag: NDArray,
    a: NDArray,
    r: NDArray,
    num_actions: int,
    cfg: RIAdditiveRewardConfig,
) -> MLPScalar:
    """Train g(x,a) using pairwise differences within same action and similar lag."""
    _set_torch_seed(cfg.seed)

    i_idx, j_idx = _build_pairs_nn_by_action(x_lag, a, k=cfg.g_knn_k)
    if i_idx.size == 0:
        # Degenerate fallback
        model = MLPScalar(in_dim=x.shape[1] + num_actions, hidden_sizes=cfg.g_hidden, dropout=cfg.dropout).to(
            cfg.device
        )
        for p in model.parameters():
            nn.init.zeros_(p)
        return model

    x_i = x[i_idx].astype(np.float32)
    x_j = x[j_idx].astype(np.float32)
    a_i = a[i_idx]
    a_j = a[j_idx]
    y = (r[i_idx] - r[j_idx]).astype(np.float32)

    feat_i = np.concatenate([x_i, _one_hot(a_i, num_actions)], axis=1)
    feat_j = np.concatenate([x_j, _one_hot(a_j, num_actions)], axis=1)

    X_i = torch.tensor(feat_i, dtype=torch.float32, device=cfg.device)
    X_j = torch.tensor(feat_j, dtype=torch.float32, device=cfg.device)
    y_t = torch.tensor(y, dtype=torch.float32, device=cfg.device)

    model = MLPScalar(in_dim=X_i.shape[1], hidden_sizes=cfg.g_hidden, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.g_lr, weight_decay=cfg.g_weight_decay)

    n_pairs = X_i.shape[0]
    bs = cfg.g_batch_size

    for ep in range(cfg.g_epochs):
        perm = torch.randperm(n_pairs, device=cfg.device)
        for s in range(0, n_pairs, bs):
            b = perm[s : s + bs]
            pred = model(X_i[b]) - model(X_j[b])
            loss = F.mse_loss(pred, y_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model


def _predict_g_all_actions(
    g_model: MLPScalar,
    x: NDArray,
    num_actions: int,
    device: str,
) -> NDArray:
    """Predict g(x,a) for all actions (n, num_actions)."""
    n, d = x.shape
    x_rep = np.repeat(x.astype(np.float32), repeats=num_actions, axis=0)
    a_rep = np.tile(np.arange(num_actions, dtype=np.int64), reps=n)
    feat = np.concatenate([x_rep, _one_hot(a_rep, num_actions)], axis=1)

    X = torch.tensor(feat, dtype=torch.float32, device=device)
    g_model.eval()
    with torch.no_grad():
        pred = g_model(X).cpu().numpy().astype(np.float64)
    return pred.reshape(n, num_actions)


def _fit_h_sklearn(
    x_lag: NDArray,
    residual: NDArray,
    cfg: RIAdditiveRewardConfig,
) -> Tuple[StandardScaler, MLPRegressor]:
    scaler = StandardScaler()
    X = scaler.fit_transform(x_lag)
    reg = MLPRegressor(
        hidden_layer_sizes=cfg.h_hidden,
        max_iter=cfg.h_max_iter,
        alpha=cfg.h_alpha,
        early_stopping=cfg.h_early_stopping,
        validation_fraction=cfg.h_validation_fraction,
        random_state=cfg.seed + 999,
    )
    reg.fit(X, residual)
    return scaler, reg


def _predict_h_sklearn(
    scaler: StandardScaler,
    reg: MLPRegressor,
    x_lag: NDArray,
) -> NDArray:
    X = scaler.transform(x_lag)
    return reg.predict(X).astype(np.float64)


def _fit_h_sklearn_actionwise(
    x_lag: NDArray,
    residual: NDArray,
    a: NDArray,
    num_actions: int,
    cfg: RIAdditiveRewardConfig,
) -> Tuple[List[StandardScaler], List[MLPRegressor]]:
    scalers: List[StandardScaler] = []
    regs: List[MLPRegressor] = []
    for action in range(num_actions):
        idx = np.where(a == action)[0]
        scaler = StandardScaler()
        if idx.size < cfg.h_min_samples:
            # Fallback to a constant predictor to avoid early_stopping split on tiny samples.
            scaler.fit(x_lag[: min(x_lag.shape[0], 1)])
            const = float(residual[idx].mean()) if idx.size > 0 else float(residual.mean())
            reg = DummyRegressor(strategy="constant", constant=const)
            reg.fit(scaler.transform(x_lag[: min(x_lag.shape[0], 1)]), np.array([const]))
        else:
            reg = MLPRegressor(
                hidden_layer_sizes=cfg.h_hidden,
                max_iter=cfg.h_max_iter,
                alpha=cfg.h_alpha,
                early_stopping=cfg.h_early_stopping,
                validation_fraction=cfg.h_validation_fraction,
                random_state=cfg.seed + 999 + action,
            )
            X = scaler.fit_transform(x_lag[idx])
            reg.fit(X, residual[idx])
        scalers.append(scaler)
        regs.append(reg)
    return scalers, regs


def _predict_h_sklearn_actionwise(
    scalers: List[StandardScaler],
    regs: List[MLPRegressor],
    x_lag: NDArray,
) -> NDArray:
    num_actions = len(regs)
    n = x_lag.shape[0]
    out = np.zeros((n, num_actions), dtype=np.float64)
    for action, (scaler, reg) in enumerate(zip(scalers, regs)):
        X = scaler.transform(x_lag)
        out[:, action] = reg.predict(X).astype(np.float64)
    return out


def train_predict_ri_additive_crossfit(
    x: NDArray,
    x_lag: NDArray,
    a: NDArray,
    r: NDArray,
    num_actions: int,
    cfg: RIAdditiveRewardConfig,
    folds: Optional[List[NDArray[np.int64]]] = None,
) -> Tuple[NDArray, Dict[str, float]]:
    """Cross-fitted RI-additive reward model: q_hat(x,x_lag,a)=g_hat(x,a)+h_hat(x_lag)."""
    n = x.shape[0]
    if folds is None:
        folds = _make_folds(n, cfg.n_folds, cfg.seed)

    q_hat = np.zeros((n, num_actions), dtype=np.float64)

    num_folds = len(folds)
    for k, fold_idx in enumerate(folds):
        if fold_idx.size == 0:
            continue
        if num_folds <= 1:
            train_idx = fold_idx
        else:
            mask = np.ones(n, dtype=bool)
            mask[fold_idx] = False
            train_idx = np.where(mask)[0]

        cfg_k = RIAdditiveRewardConfig(**{**cfg.__dict__, "seed": cfg.seed + 100 * k})

        g_model = _train_g_pairwise(
            x=x[train_idx],
            x_lag=x_lag[train_idx],
            a=a[train_idx],
            r=r[train_idx],
            num_actions=num_actions,
            cfg=cfg_k,
        )

        g_tr_all = _predict_g_all_actions(g_model, x[train_idx], num_actions, device=cfg_k.device)
        g_tr_fact = g_tr_all[np.arange(train_idx.shape[0]), a[train_idx]]
        residual_tr = r[train_idx] - g_tr_fact

        g_te_all = _predict_g_all_actions(g_model, x[fold_idx], num_actions, device=cfg_k.device)

        if cfg_k.h_action_specific:
            scalers_h, regs_h = _fit_h_sklearn_actionwise(
                x_lag=x_lag[train_idx],
                residual=residual_tr,
                a=a[train_idx],
                num_actions=num_actions,
                cfg=cfg_k,
            )
            h_te_all = _predict_h_sklearn_actionwise(scalers_h, regs_h, x_lag[fold_idx])
            q_hat[fold_idx] = g_te_all + h_te_all
        else:
            scaler_h, h_reg = _fit_h_sklearn(x_lag=x_lag[train_idx], residual=residual_tr, cfg=cfg_k)
            h_te = _predict_h_sklearn(scaler_h, h_reg, x_lag[fold_idx])
            q_hat[fold_idx] = g_te_all + h_te[:, None]

    info = {
        "q_hat_mean": float(np.mean(q_hat)),
        "q_hat_std": float(np.std(q_hat)),
    }
    return q_hat, info


def estimate_alc_knn(
    residual: NDArray,
    x_lag: NDArray,
    a: NDArray,
) -> float:
    """Cheap ALC proxy using 1-NN residual differences within action."""
    i_idx, j_idx = _build_pairs_nn_by_action(x_lag, a)
    if i_idx.size == 0:
        return 0.0
    diff = residual[i_idx] - residual[j_idx]
    return float(0.5 * np.mean(diff**2))


class MTRIRewardModel(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x)


class MTRICritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class MTRICentering(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def _subset_dataset(dataset: dict, indices: NDArray[np.int64]) -> dict:
    return dict(
        num_data=indices.shape[0],
        num_features=dataset["num_features"],
        num_actions=dataset["num_actions"],
        x_t=dataset["x_t"][indices],
        x_t_l=dataset["x_t_l"][indices],
        a_t=dataset["a_t"][indices],
        r=dataset["r"][indices],
    )


def train_reward_model_mtri_models(
    dataset: dict,
    lambda_mtri: float = 1.0,
    hidden_dim: int = 64,
    critic_hidden_dim: int = 64,
    centering_hidden_dim: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_epochs: int = 100,
    weight_decay: float = 1e-4,
    critic_reg: float = 1e-3,
    random_state: int = 42,
) -> Tuple[MTRIRewardModel, MTRICritic, MTRICentering]:
    torch.manual_seed(random_state)
    x_t = torch.from_numpy(dataset["x_t"]).float()
    x_t_l = torch.from_numpy(dataset["x_t_l"]).float()
    a_t = torch.from_numpy(dataset["a_t"]).long()
    r = torch.from_numpy(dataset["r"]).float()

    num_actions = dataset["num_actions"]
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_t, x_t_l, a_t, r),
        batch_size=batch_size,
        shuffle=True,
    )

    reward_model = MTRIRewardModel(
        input_dim=x_t.shape[1] + x_t_l.shape[1],
        num_actions=num_actions,
        hidden_dim=hidden_dim,
    )
    critic_model = MTRICritic(
        input_dim=x_t.shape[1] + x_t_l.shape[1] + num_actions,
        hidden_dim=critic_hidden_dim,
    )
    centering_model = MTRICentering(
        input_dim=x_t_l.shape[1] + num_actions,
        hidden_dim=centering_hidden_dim,
    )

    reward_opt = torch.optim.AdamW(
        reward_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    critic_opt = torch.optim.AdamW(
        critic_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    centering_opt = torch.optim.AdamW(
        centering_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    reward_model.train()
    critic_model.train()
    centering_model.train()
    for _ in range(num_epochs):
        for x_b, x_l_b, a_b, r_b in data_loader:
            idx = torch.arange(a_b.shape[0], dtype=torch.long)
            a_onehot = F.one_hot(a_b, num_classes=num_actions).float()
            reward_inputs = torch.cat([x_b, x_l_b], dim=1)
            critic_inputs = torch.cat([x_b, x_l_b, a_onehot], dim=1)
            centering_inputs = torch.cat([x_l_b, a_onehot], dim=1)

            q_hat = reward_model(reward_inputs)
            q_hat_factual = q_hat[idx, a_b]
            residual = r_b - q_hat_factual

            f_val = critic_model(critic_inputs)
            c_val = centering_model(centering_inputs)

            centering_loss = (f_val.detach() - c_val).pow(2).mean()
            centering_opt.zero_grad()
            centering_loss.backward()
            centering_opt.step()

            c_val = centering_model(centering_inputs)
            f_tilde = f_val - c_val
            f_norm = f_tilde / (f_tilde.pow(2).mean().sqrt() + 1e-8)
            moment = (residual.detach() * f_norm).mean()
            critic_loss = -(moment**2) + critic_reg * (f_tilde.pow(2).mean())
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            f_tilde_det = (critic_model(critic_inputs) - centering_model(centering_inputs)).detach()
            f_norm_det = f_tilde_det / (f_tilde_det.pow(2).mean().sqrt() + 1e-8)
            moment_q = (residual * f_norm_det).mean()
            mse_loss = residual.pow(2).mean()
            reward_loss = mse_loss + lambda_mtri * (moment_q**2)
            reward_opt.zero_grad()
            reward_loss.backward()
            reward_opt.step()

    return reward_model, critic_model, centering_model


def predict_reward_model_mtri(
    reward_model: MTRIRewardModel,
    x_t: NDArray,
    x_t_l: NDArray,
) -> NDArray:
    reward_model.eval()
    inputs = torch.from_numpy(np.hstack([x_t, x_t_l])).float()
    with torch.no_grad():
        q_hat = reward_model(inputs).detach().numpy()
    return q_hat


def estimate_mtri_moment(
    critic_model: MTRICritic,
    centering_model: MTRICentering,
    x_t: NDArray,
    x_t_l: NDArray,
    a_t: NDArray,
    residual: NDArray,
    num_actions: int,
) -> float:
    critic_model.eval()
    centering_model.eval()
    x_t_t = torch.from_numpy(x_t).float()
    x_t_l_t = torch.from_numpy(x_t_l).float()
    a_t_t = torch.from_numpy(a_t).long()
    a_onehot = F.one_hot(a_t_t, num_classes=num_actions).float()
    critic_inputs = torch.cat([x_t_t, x_t_l_t, a_onehot], dim=1)
    centering_inputs = torch.cat([x_t_l_t, a_onehot], dim=1)
    with torch.no_grad():
        f_val = critic_model(critic_inputs)
        c_val = centering_model(centering_inputs)
        f_tilde = (f_val - c_val).numpy()
    norm = np.sqrt(np.mean(f_tilde**2)) + 1e-8
    f_norm = f_tilde / norm
    return float(np.mean(residual * f_norm))


def train_reward_model_mtri_crossfit(
    dataset: dict,
    num_folds: int = 2,
    lambda_mtri: float = 1.0,
    hidden_dim: int = 64,
    critic_hidden_dim: int = 64,
    centering_hidden_dim: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_epochs: int = 100,
    weight_decay: float = 1e-4,
    critic_reg: float = 1e-3,
    random_state: int = 42,
    folds: Optional[List[NDArray[np.int64]]] = None,
) -> Tuple[NDArray, float, List[NDArray[np.int64]]]:
    num_data = dataset["num_data"]
    if folds is None:
        folds = _make_folds(num_data, num_folds, random_state)
    q_hat = np.zeros((num_data, dataset["num_actions"]))
    alc_values: List[Tuple[int, float]] = []

    for fold_idx in folds:
        if fold_idx.size == 0:
            continue
        if num_folds <= 1:
            train_idx = fold_idx
        else:
            mask = np.ones(num_data, dtype=bool)
            mask[fold_idx] = False
            train_idx = np.where(mask)[0]

        train_dataset = _subset_dataset(dataset, train_idx)
        reward_model, critic_model, centering_model = train_reward_model_mtri_models(
            dataset=train_dataset,
            lambda_mtri=lambda_mtri,
            hidden_dim=hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            centering_hidden_dim=centering_hidden_dim,
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            weight_decay=weight_decay,
            critic_reg=critic_reg,
            random_state=random_state,
        )

        q_hat_fold = predict_reward_model_mtri(
            reward_model,
            dataset["x_t"][fold_idx],
            dataset["x_t_l"][fold_idx],
        )
        q_hat[fold_idx] = q_hat_fold

        idx = np.arange(fold_idx.shape[0])
        residual = dataset["r"][fold_idx] - q_hat_fold[idx, dataset["a_t"][fold_idx]]
        moment = estimate_mtri_moment(
            critic_model,
            centering_model,
            dataset["x_t"][fold_idx],
            dataset["x_t_l"][fold_idx],
            dataset["a_t"][fold_idx],
            residual,
            dataset["num_actions"],
        )
        alc_values.append((fold_idx.shape[0], moment**2))

    if len(alc_values) == 0:
        alc = 0.0
    else:
        alc = float(np.average([val for _, val in alc_values], weights=[count for count, _ in alc_values]))
    return q_hat, alc, folds


def train_reward_model(
    dataset: dict,
    lambda_mtri: float = 1.0,
    num_epochs: int = 100,
    hidden_dim: int = 64,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    random_state: int = 42,
) -> NDArray:
    reward_model, _, _ = train_reward_model_mtri_models(
        dataset=dataset,
        lambda_mtri=lambda_mtri,
        hidden_dim=hidden_dim,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        random_state=random_state,
    )
    return predict_reward_model_mtri(
        reward_model,
        dataset["x_t"],
        dataset["x_t_l"],
    )


def fit_predict_by_MLP(
    features: NDArray,
    actions: NDArray,
    rewards: NDArray,
    hidden_layer_sizes: Tuple[int, ...] = (50, 50, 50),
    random_state: int = 42,
) -> NDArray:
    """Estimate the reward function using an MLP on (x, a)."""
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    X = np.hstack((features, actions[:, None]))
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X)
    model.fit(X_, rewards)
    return model.predict(X_)


def fit_predict_by_MLP_for_all_actions(
    features: NDArray,
    actions: NDArray,
    rewards: NDArray,
    num_actions: int,
    hidden_layer_sizes: Tuple[int, ...] = (50, 50, 50),
    random_state: int = 42,
) -> NDArray:
    """Estimate rewards using a single MLP on (x,a) and predict all actions."""
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    a_onehot = np.eye(num_actions)[actions]
    X = np.hstack((features, a_onehot))
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X)
    model.fit(X_, rewards)

    num_data = features.shape[0]
    actions_all = np.tile(np.arange(num_actions), num_data)
    features_rep = np.repeat(features, num_actions, axis=0)
    a_all_onehot = np.eye(num_actions)[actions_all]
    X_all = np.hstack((features_rep, a_all_onehot))
    X_all_scaled = scaler.transform(X_all)
    q_hat = model.predict(X_all_scaled)
    return q_hat.reshape(num_data, num_actions)


def fit_predict_by_MLP_actionwise(
    features: NDArray,
    actions: NDArray,
    rewards: NDArray,
    num_actions: int,
    hidden_layer_sizes: Tuple[int, ...] = (30, 30),
    random_state: int = 42,
    min_samples: int = 10,
) -> NDArray:
    """Fit separate MLPs per action using only current features (misspecified by design)."""
    num_data = features.shape[0]
    q_hat = np.zeros((num_data, num_actions), dtype=np.float64)

    for a in range(num_actions):
        idx = np.where(actions == a)[0]
        if idx.size < min_samples:
            q_hat[:, a] = float(np.mean(rewards)) if rewards.size > 0 else 0.0
            continue
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state + a)
        scaler = StandardScaler()
        X = scaler.fit_transform(features[idx])
        model.fit(X, rewards[idx])
        q_hat[:, a] = model.predict(scaler.transform(features))

    return q_hat


def fit_predict_by_MLP_actionwise_crossfit(
    features: NDArray,
    actions: NDArray,
    rewards: NDArray,
    num_actions: int,
    n_folds: int = 2,
    hidden_layer_sizes: Tuple[int, ...] = (30, 30),
    random_state: int = 42,
    min_samples: int = 10,
    folds: Optional[List[NDArray[np.int64]]] = None,
) -> NDArray:
    """Cross-fitted actionwise MLP to avoid in-sample optimism."""
    n = features.shape[0]
    if folds is None:
        folds = _make_folds(n, n_folds, random_state)
    q_hat = np.zeros((n, num_actions), dtype=np.float64)

    for k, fold_idx in enumerate(folds):
        if fold_idx.size == 0:
            continue
        if n_folds <= 1:
            train_idx = fold_idx
        else:
            mask = np.ones(n, dtype=bool)
            mask[fold_idx] = False
            train_idx = np.where(mask)[0]

        q_hat_fold = np.zeros((fold_idx.shape[0], num_actions), dtype=np.float64)
        for a in range(num_actions):
            idx = train_idx[actions[train_idx] == a]
            if idx.size < min_samples:
                const = float(np.mean(rewards[train_idx])) if train_idx.size > 0 else 0.0
                q_hat_fold[:, a] = const
                continue
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state + 100 * k + a)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(features[idx])
            model.fit(X_tr, rewards[idx])
            q_hat_fold[:, a] = model.predict(scaler.transform(features[fold_idx]))

        q_hat[fold_idx] = q_hat_fold

    return q_hat
