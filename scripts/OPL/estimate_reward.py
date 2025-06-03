from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
import torch
from torch.types import Tensor
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairData4CurrentRewardEstimate:
    x1: Tensor
    x2: Tensor
    a1: Tensor
    a2: Tensor
    r1: Tensor
    r2: Tensor

    def __getitem__(self, index):
        return (
            self.x1[index],
            self.x2[index],
            self.a1[index],
            self.a2[index],
            self.r1[index],
            self.r2[index],
        )

    def __len__(self):
        return self.x1.shape[0]


@dataclass
class Data4LaggedRewardEstimate:
    x_t_l: Tensor
    a_t: Tensor
    r_t: Tensor

    def __getitem__(self, index):
        return (
            self.x_t_l[index],
            self.a_t[index],
            self.r_t[index],
        )

    def __len__(self):
        return self.x_t_l.shape[0]


class CurrentRewardEstimator(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_features: int,
        hidden_dim: int = 30,
    ):
        super(CurrentRewardEstimator, self).__init__()
        self.num_actions: int = num_actions
        self.num_features: int = num_features

        # current reward network
        self.fc1 = nn.Linear(self.num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.num_actions)

    def current_reward_pred(self, x):
        g_hat = F.elu(self.fc1(x))
        g_hat = F.elu(self.fc2(g_hat))
        g_hat = F.elu(self.fc3(g_hat))
        g_hat = self.fc4(g_hat)
        return g_hat

    def forward(self, x1, x2, a1, a2, r1, r2):
        g_hat1 = self.current_reward_pred(x1)
        g_hat2 = self.current_reward_pred(x2)
        g_hat1_action, g_hat2_action = g_hat1[:, a1], g_hat2[:, a2]
        loss = ((r1 - r2) - (g_hat1_action - g_hat2_action)) ** 2
        return loss.mean()


class LaggedRewardEstimator(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_features: int,
        hidden_dim: int = 30,
    ):
        super(LaggedRewardEstimator, self).__init__()
        self.num_actions: int = num_actions
        self.num_features: int = num_features

        # current reward network
        self.fc1 = nn.Linear(self.num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.num_actions)

    def lagged_reward_pred(self, x):
        h_hat = F.elu(self.fc1(x))
        h_hat = F.elu(self.fc2(h_hat))
        h_hat = F.elu(self.fc3(h_hat))
        h_hat = self.fc4(h_hat)
        return h_hat

    def forward(self, x_t_l, a_t, r):
        h_hat = self.lagged_reward_pred(x_t_l)
        h_hat_action = h_hat[:, a_t]
        loss = (r - h_hat_action) ** 2
        return loss.mean()


def make_pair_data(dataset: dict):
    """Make pair data for training residual reward model"""
    num_data = dataset["num_data"]
    x_t_l = dataset["x_t_l"]
    x_t = dataset["x_t"]
    a_t = dataset["a_t"]
    r = dataset["r"]
    lag_features_n_actions = np.hstack((x_t_l, a_t[:, None]))
    dist_matrix = cdist(lag_features_n_actions, lag_features_n_actions)
    pairs = []
    for i in range(num_data):
        min_dist_idx = np.argmin(np.delete(dist_matrix[i], i))
        if i != min_dist_idx:
            pairs.append((i, min_dist_idx))

    x1 = []
    x2 = []
    a1 = []
    a2 = []
    r1 = []
    r2 = []
    for p1, p2 in pairs:
        x1.append(x_t[p1])
        x2.append(x_t[p2])
        a1.append(a_t[p1])
        a2.append(a_t[p2])
        r1.append(r[p1])
        r2.append(r[p2])

    return PairData4CurrentRewardEstimate(
        torch.from_numpy(np.array(x1)).float(),
        torch.from_numpy(np.array(x2)).float(),
        torch.from_numpy(np.array(a1)).long(),
        torch.from_numpy(np.array(a2)).long(),
        torch.from_numpy(np.array(r1)).float(),
        torch.from_numpy(np.array(r2)).float(),
    )


def train_lagged_effect_model(
    lag_features: NDArray,
    actions: NDArray,
    rewards: NDArray,
    num_actions: int,
    num_features: int,
    lr: float = 1e-2,
    batch_size: int = 32,
    num_epochs: int = 50,
    gamma: float = 0.95,
    weight_decay: float = 1e-4,
    verbose: bool = False,
) -> None:
    """Train the lagged effect of reward model"""
    train_data = Data4LaggedRewardEstimate(
        torch.from_numpy(np.array(lag_features)).float(),
        torch.from_numpy(np.array(actions)).long(),
        torch.from_numpy(np.array(rewards)).float(),
    )
    data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    model = LaggedRewardEstimator(
        num_actions=num_actions,
        num_features=num_features,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    model.train()
    loss_list = []
    for _ in range(num_epochs):
        losses = []
        for x_t_l, a_t, r in data_loader:
            loss = model(x_t_l, a_t, r)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        if verbose:
            print(_, np.average(losses))
        loss_list.append(np.average(losses))
        scheduler.step()
    x_t_l = torch.from_numpy(lag_features).float()
    h_hat = model.lagged_reward_pred(x_t_l).detach().numpy()

    return h_hat


def train_pair_model(
    dataset: dict,
    lr: float = 1e-2,
    batch_size: int = 32,
    num_epochs: int = 50,
    gamma: float = 0.95,
    weight_decay: float = 1e-4,
    verbose: bool = False,
) -> None:
    """Train the residual reward model"""
    pair_data = make_pair_data(dataset)
    data_loader = torch.utils.data.DataLoader(
        pair_data,
        batch_size=batch_size,
        shuffle=True,
    )
    model = CurrentRewardEstimator(
        num_actions=dataset["num_actions"],
        num_features=dataset["num_features"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    model.train()
    loss_list = []
    for _ in range(num_epochs):
        losses = []
        for x1, x2, a1, a2, r1, r2 in data_loader:
            loss = model(x1, x2, a1, a2, r1, r2)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        if verbose:
            print(_, np.average(losses))
        loss_list.append(np.average(losses))
        scheduler.step()
    x_t = torch.from_numpy(dataset["x_t"]).float()
    g_hat = model.current_reward_pred(x_t).detach().numpy()

    return g_hat


def train_reward_model(dataset: dict) -> NDArray:
    """Train the reward model"""
    g_hat = train_pair_model(dataset=dataset)
    reward = dataset["r"].astype(float)
    reward_lag = reward - g_hat[np.arange(dataset["x_t"].shape[0]), dataset["a_t"]]
    h_hat = train_lagged_effect_model(
        lag_features=dataset["x_t_l"],
        actions=dataset["a_t"],
        rewards=reward_lag,
        num_actions=dataset["num_actions"],
        num_features=dataset["num_features"],
    )

    f_hat = g_hat + h_hat

    return f_hat
