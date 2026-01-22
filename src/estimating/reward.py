from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import torch
from torch.types import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MTRIRewardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x)


class MTRICritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class MTRICentering(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


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
) -> tuple[MTRIRewardModel, MTRICritic, MTRICentering]:
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


def _make_folds(num_data: int, num_folds: int, random_state: int) -> list[NDArray]:
    if num_folds <= 1:
        return [np.arange(num_data)]
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(num_data)
    return [np.sort(fold) for fold in np.array_split(perm, num_folds)]


def _subset_dataset(dataset: dict, indices: NDArray) -> dict:
    return dict(
        num_data=indices.shape[0],
        num_features=dataset["num_features"],
        num_actions=dataset["num_actions"],
        x_t=dataset["x_t"][indices],
        x_t_l=dataset["x_t_l"][indices],
        a_t=dataset["a_t"][indices],
        r=dataset["r"][indices],
    )


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
    folds: Optional[list[NDArray]] = None,
) -> tuple[NDArray, float, list[NDArray]]:
    num_data = dataset["num_data"]
    if folds is None:
        folds = _make_folds(num_data, num_folds, random_state)
    q_hat = np.zeros((num_data, dataset["num_actions"]))
    alc_values = []

    for fold_idx in folds:
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

    alc = np.average(
        [val for _, val in alc_values],
        weights=[count for count, _ in alc_values],
    )
    return q_hat, float(alc), folds


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
    num_actions: int,
    hidden_layer_sizes: tuple = (50, 50, 50),
    random_state: int = 42,
) -> NDArray:
    """Fit and predict the reward function using a single MLP on (x, a) and predict all actions.

    Args:
        features (NDArray): features
        actions (NDArray): actions
        rewards (NDArray): rewards
        num_actions (int): number of actions
        hidden_layer_sizes (tuple, optional): hidden layer sizes. Defaults to (50, 50, 50).
        random_state (int, optional): random state. Defaults to 42.

    Returns:
        NDArray: predicted rewards
    """
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    X = np.hstack((features, actions[:, None]))
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X)
    model.fit(X_, rewards)

    num_data = features.shape[0]
    actions_all = np.tile(np.arange(num_actions), num_data)
    features_rep = np.repeat(features, num_actions, axis=0)
    X_all = np.hstack((features_rep, actions_all[:, None]))
    X_all_scaled = scaler.transform(X_all)
    q_hat = model.predict(X_all_scaled)
    return q_hat.reshape(num_data, num_actions)
