from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
import torch
from torch.types import Tensor
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax
from sklearn.utils import check_random_state

from estimate_policy_value_given_lag_features import PolicyValueGivenLagFeaturesEstimator


@dataclass
class RegressionBasedPolicyDataset(torch.utils.data.Dataset):
    features: NDArray
    actions: NDArray
    rewards: NDArray

    def __post_init__(self) -> None:
        assert self.features.shape[0] == self.actions.shape[0] == self.rewards.shape[0]

    def __getitem__(self, index):
        return (
            self.features[index],
            self.actions[index],
            self.rewards[index],
        )

    def __len__(self):
        return self.features.shape[0]


@dataclass
class GradientBasedPolicyDataset(torch.utils.data.Dataset):
    features: NDArray
    actions: NDArray
    rewards: NDArray
    q_hat: NDArray
    pi_0: NDArray

    def __post_init__(self) -> None:
        assert (
            self.features.shape[0]
            == self.actions.shape[0]
            == self.rewards.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.features[index],
            self.actions[index],
            self.rewards[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.features.shape[0]


@dataclass
class DOLCEDataset(torch.utils.data.Dataset):
    features: NDArray
    lag_features: NDArray
    actions: NDArray
    rewards: NDArray
    q_hat: NDArray
    pi_0: NDArray

    def __post_init__(self) -> None:
        assert (
            self.features.shape[0]
            == self.lag_features.shape[0]
            == self.actions.shape[0]
            == self.rewards.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.features[index],
            self.lag_features[index],
            self.actions[index],
            self.rewards[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.features.shape[0]


@dataclass
class RegressionBasedPolicyLearner:
    num_features: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.005
    gamma: float = 0.98
    alpha: float = 1e-6
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 42

    def __post_init__(self) -> None:
        layer_list = []
        input_size = self.num_features

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise NotImplementedError(
                "`activation` must be one of 'tanh', 'relu', or 'elu'"
            )

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, dataset: dict, dataset_test: dict) -> None:
        x_t = dataset["x_t"]
        a_t = dataset["a_t"]
        r = dataset["r"]

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(x_t, a_t, r)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        q_train, q_test = dataset["q"], dataset_test["q"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for x_t_, a_t_, r_ in training_data_loader:
                optimizer.zero_grad()
                q_hat = self.nn_model(x_t_)
                idx = torch.arange(a_t_.shape[0], dtype=torch.long)
                loss = ((r_ - q_hat[idx, a_t_]) ** 2).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            pi_train = self.predict(dataset)
            scheduler.step()
            self.train_value.append((q_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_test * pi_test).sum(1).mean())
            self.train_loss.append(loss_epoch)

    def _create_train_data_for_opl(
        self,
        x_t: NDArray,
        a_t: NDArray,
        r: NDArray,
    ) -> tuple:
        dataset = RegressionBasedPolicyDataset(
            torch.from_numpy(x_t).float(),
            torch.from_numpy(a_t).long(),
            torch.from_numpy(r).float(),
        )

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        return data_loader

    def predict(self, dataset_test: dict, beta: float = 0.3) -> NDArray:
        q_hat = self.predict_q(dataset_test)

        return softmax(beta * q_hat, axis=1)

    def predict_q(self, dataset_test: dict) -> NDArray:
        self.nn_model.eval()
        x_t = torch.from_numpy(dataset_test["x_t"]).float()

        return self.nn_model(x_t).detach().numpy()


@dataclass
class GradientBasedPolicyLearner:
    num_features: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.005
    gamma: float = 0.98
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 42

    def __post_init__(self) -> None:
        layer_list = []
        input_size = self.num_features

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise NotImplementedError(
                "`activation` must be one of 'tanh', 'relu', or 'elu'"
            )

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random = check_random_state(self.random_state)
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(
        self, dataset: dict, dataset_test: dict, q_hat: Optional[NDArray] = None
    ) -> None:
        x_t = dataset["x_t"]
        a_t = dataset["a_t"]
        r = dataset["r"]
        pi_0 = dataset["pi_0"]
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x_t,
            a_t,
            r,
            q_hat,
            pi_0,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        q_train, q_test = dataset["q"], dataset_test["q"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for x_t_, a_t_, r_, q_hat_, pi_0_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_t_)
                loss = -self._estimate_policy_gradient(
                    a_t=a_t_,
                    r=r_,
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset)
            self.train_value.append((q_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_test * pi_test).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x_t: NDArray,
        a_t: NDArray,
        r: NDArray,
        q_hat: NDArray,
        pi_0: NDArray,
    ) -> tuple:
        dataset = GradientBasedPolicyDataset(
            torch.from_numpy(x_t).float(),
            torch.from_numpy(a_t).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        a_t: Tensor,
        r: Tensor,
        q_hat: Tensor,
        pi: Tensor,
        pi_0: Tensor,
    ) -> Tensor:
        current_policy = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a_t.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a_t]
        w = current_policy[idx, a_t] / pi_0[idx, a_t]
        estimated_policy_grad_arr = w * (r - q_hat_factual) * log_prob[idx, a_t]
        estimated_policy_grad_arr += torch.sum(q_hat * current_policy * log_prob, dim=1)

        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a_t]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: NDArray) -> NDArray:
        self.nn_model.eval()
        x_t = torch.from_numpy(dataset_test["x_t"]).float()
        return self.nn_model(x_t).detach().numpy()


@dataclass
class DOLCE:
    num_features: int
    num_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 16
    learning_rate_init: float = 0.005
    gamma: float = 0.98
    alpha: float = 1e-6
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 42

    def __post_init__(self) -> None:
        layer_list = []
        input_size = self.num_features

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise NotImplementedError(
                "`activation` must be one of 'tanh', 'relu', or 'elu'"
            )

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random = check_random_state(self.random_state)
        self.train_loss: List[float] = []
        self.train_value: List[float] = []
        self.test_value: List[float] = []

    def fit(
        self, dataset: dict, dataset_test: dict, q_hat: Optional[NDArray] = None
    ) -> None:
        x_t = dataset["x_t"]
        x_t_l = dataset["x_t_l"]
        a_t = dataset["a_t"]
        r = dataset["r"]
        pi_0 = dataset["pi_0"]
        if q_hat is None:
            q_hat = np.zeros((r.shape[0], self.num_actions))

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x_t,
            x_t_l,
            a_t,
            r,
            q_hat,
            pi_0,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        q_train, q_test = dataset["q"], dataset_test["q"]
        for _ in range(self.max_iter):
            loss_epoch = 0.0
            self.nn_model.train()
            for x_t_, x_t_l_, a_t_, r_, q_hat_, pi_0_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x_t_)
                loss = -self._estimate_policy_gradient(
                    x_t=x_t_,
                    x_t_l=x_t_l_,
                    a_t=a_t_,
                    r=r_,
                    q_hat=q_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
            scheduler.step()
            pi_train = self.predict(dataset)
            self.train_value.append((q_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_test * pi_test).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        x_t: NDArray,
        x_t_l: NDArray,
        a_t: NDArray,
        r: NDArray,
        q_hat: NDArray,
        pi_0: NDArray,
    ) -> tuple:
        dataset = DOLCEDataset(
            torch.from_numpy(x_t).float(),
            torch.from_numpy(x_t_l).float(),
            torch.from_numpy(a_t).long(),
            torch.from_numpy(r).float(),
            torch.from_numpy(q_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        x_t: Tensor,
        x_t_l: Tensor,
        a_t: Tensor,
        r: Tensor,
        q_hat: Tensor,
        pi: Tensor,
        pi_0: Tensor,
    ) -> Tensor:
        current_policy = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a_t.shape[0], dtype=torch.long)

        estimator = PolicyValueGivenLagFeaturesEstimator()
        integral_value_current = estimator.mc_int_cond_prob_times_pi(
            x_t_l=x_t_l, x_t=x_t, pi=current_policy, action_indices=a_t
        )
        integral_value_0 = estimator.mc_int_cond_prob_times_pi(
            x_t_l=x_t_l, x_t=x_t, pi=pi_0, action_indices=a_t
        )

        integral_value_pi = estimator.mc_int_cond_prob_times_pi(
            x_t_l=x_t_l, x_t=x_t, pi=pi, action_indices=a_t
        )
        log_prob_pi = torch.log(integral_value_pi + self.log_eps)

        q_hat_factual = q_hat[idx, a_t]
        w = integral_value_current / integral_value_0
        estimated_policy_grad_arr = w * (r - q_hat_factual) * log_prob_pi
        estimated_policy_grad_arr += torch.sum(q_hat * current_policy * log_prob, dim=1)

        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a_t]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: NDArray) -> NDArray:
        self.nn_model.eval()
        x_t = torch.from_numpy(dataset_test["x_t"]).float()
        return self.nn_model(x_t).detach().numpy()
