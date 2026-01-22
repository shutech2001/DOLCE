from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch.types import Tensor
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax
from sklearn.utils import check_random_state

from estimating.lag_policy import LaggedPolicyEstimatorTorch
from estimating.reward import train_reward_model_mtri_crossfit, _make_folds


@dataclass
class RegressionBasedPolicyDataset(torch.utils.data.Dataset):
    """Dataset for regression-based policy learner.

    Args:
        features (NDArray): features
        actions (NDArray): actions
        rewards (NDArray): rewards
    """

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
    """Dataset for gradient-based policy learner.

    Args:
        features (NDArray): features
        actions (NDArray): actions
        rewards (NDArray): rewards
        q_hat (NDArray): estimated rewards
        pi_0 (NDArray): initial policy
    """

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
    """Dataset for DOLCE.

    Args:
        features (NDArray): features
        lag_features (NDArray): lag features
        actions (NDArray): actions
        rewards (NDArray): rewards
        q_hat (NDArray): estimated rewards
        pi_0 (NDArray): initial policy
    """

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
            raise NotImplementedError("`activation` must be one of 'tanh', 'relu', or 'elu'")

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
        """Fit the regression-based policy learner.

        Args:
            dataset (dict): dataset
            dataset_test (dict): test dataset
        """
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Create train data for OPL.

        Args:
            x_t (NDArray): features
            a_t (NDArray): actions
            r (NDArray): rewards

        Returns:
            Tuple[Tensor, Tensor, Tensor]: train data
        """
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
    """Gradient-based policy learner.

    Args:
        num_features (int): number of features
        num_actions (int): number of actions
        hidden_layer_size (tuple, optional): hidden layer size. Defaults to (30, 30, 30).
        activation (str, optional): activation function. Defaults to "elu".
        batch_size (int, optional): batch size. Defaults to 16.
        learning_rate_init (float, optional): learning rate. Defaults to 0.005.
        gamma (float, optional): gamma. Defaults to 0.98.
        alpha (float, optional): alpha. Defaults to 1e-6.
        imit_reg (float, optional): imitation regularization. Defaults to 0.0.
        log_eps (float, optional): log epsilon. Defaults to 1e-10.
        bandwidth (float, optional): bandwidth. Defaults to 1.0.
        weight_clip (float, optional): weight clip. Defaults to 100.0.
        solver (str, optional): solver. Defaults to "adagrad".
        max_iter (int, optional): maximum iterations. Defaults to 30.
        random_state (int, optional): random state. Defaults to 42.
    """

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
    bandwidth: float = 1.0
    weight_clip: float = 100.0
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
            raise NotImplementedError("`activation` must be one of 'tanh', 'relu', or 'elu'")

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

    def fit(self, dataset: dict, dataset_test: dict, q_hat: Optional[NDArray] = None) -> None:
        """Fit the gradient-based policy learner.

        Args:
            dataset (dict): dataset
            dataset_test (dict): test dataset
            q_hat (Optional[NDArray], optional): estimated rewards. Defaults to None.
        """
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Create train data for OPL.

        Args:
            x_t (NDArray): features
            a_t (NDArray): actions
            r (NDArray): rewards
            q_hat (NDArray): estimated rewards
            pi_0 (NDArray): initial policy

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: train data
        """
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
        """Estimate policy gradient.

        Args:
            a_t (Tensor): actions
            r (Tensor): rewards
            q_hat (Tensor): estimated rewards
            pi (Tensor): current policy
            pi_0 (Tensor): initial policy
        """
        current_policy: Tensor = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx: Tensor = torch.arange(a_t.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a_t]
        w: Tensor = current_policy[idx, a_t] / pi_0[idx, a_t]
        estimated_policy_grad_arr: Tensor = w * (r - q_hat_factual) * log_prob[idx, a_t]
        estimated_policy_grad_arr += torch.sum(q_hat * current_policy * log_prob, dim=1)

        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a_t]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: NDArray) -> NDArray:
        """Predict the policy.

        Args:
            dataset_test (NDArray): test dataset

        Returns:
            NDArray: predicted policy
        """
        self.nn_model.eval()
        x_t: Tensor = torch.from_numpy(dataset_test["x_t"]).float()
        return self.nn_model(x_t).detach().numpy()


@dataclass
class DOLCE:
    """DOLCE: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects

    Args:
        num_features (int): number of features
        num_actions (int): number of actions
        hidden_layer_size (tuple, optional): hidden layer size. Defaults to (30, 30, 30).
        activation (str, optional): activation function. Defaults to "elu".
        batch_size (int, optional): batch size. Defaults to 16.
        learning_rate_init (float, optional): learning rate. Defaults to 0.005.
        gamma (float, optional): gamma. Defaults to 0.98.
        alpha (float, optional): alpha. Defaults to 1e-6.
        imit_reg (float, optional): imitation regularization. Defaults to 0.0.
        log_eps (float, optional): log epsilon. Defaults to 1e-10.
        bandwidth (float, optional): bandwidth. Defaults to 1.0.
        weight_clip (float, optional): weight clip. Defaults to 100.0.
        tau (float, optional): tau. Defaults to 0.1.
        num_folds (int, optional): number of folds. Defaults to 2.
        solver (str, optional): solver. Defaults to "adagrad".
        max_iter (int, optional): maximum iterations. Defaults to 30.
        random_state (int, optional): random state. Defaults to 42.
    """

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
    bandwidth: float = 1.0
    weight_clip: float = 100.0
    tau: float = 0.1
    num_folds: int = 2
    solver: str = "adagrad"
    max_iter: int = 30
    random_state: int = 42

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        layer_list: List[Tuple[str, nn.Module]] = []
        input_size = self.num_features

        if self.activation == "tanh":
            activation_layer: nn.Module = nn.Tanh
        elif self.activation == "relu":
            activation_layer: nn.Module = nn.ReLU
        elif self.activation == "elu":
            activation_layer: nn.Module = nn.ELU
        else:
            raise NotImplementedError("`activation` must be one of 'tanh', 'relu', or 'elu'")

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{:d}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{:d}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.num_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random = check_random_state(self.random_state)
        self.train_loss: List[float] = []
        self.train_value: List[float] = []
        self.test_value: List[float] = []

    def fit(
        self,
        dataset: dict,
        dataset_test: dict,
        q_hat: Optional[NDArray | list[NDArray]] = None,
        alc_values: Optional[list[float]] = None,
        lambda_mtri: float = 1.0,
        mtri_epochs: int = 100,
        mtri_hidden_dim: int = 64,
        mtri_batch_size: int = 64,
        mtri_lr: float = 1e-3,
        mtri_weight_decay: float = 1e-4,
        tau: Optional[float] = None,
    ) -> None:
        """Fit the DOLCE.

        Args:
            dataset (dict): dataset
            dataset_test (dict): test dataset
            q_hat (Optional[NDArray | list[NDArray]], optional): estimated rewards. Defaults to None.
            alc_values (Optional[list[float]], optional): lag-marginalized values. Defaults to None.
            lambda_mtri (float, optional): lambda for MTRI. Defaults to 1.0.
            mtri_epochs (int, optional): number of epochs for MTRI. Defaults to 100.
            mtri_hidden_dim (int, optional): hidden dimension for MTRI. Defaults to 64.
            mtri_batch_size (int, optional): batch size for MTRI. Defaults to 64.
            mtri_lr (float, optional): learning rate for MTRI. Defaults to 1e-3.
            mtri_weight_decay (float, optional): weight decay for MTRI. Defaults to 1e-4.
            tau (Optional[float], optional): tau. Defaults to None.

        Raises:
            ValueError: q_hat must match the number of lag feature sets.
            ValueError: alc_values must match the number of lag feature sets.
            NotImplementedError: `solver` must be one of 'adam' or 'adagrad'
        """
        x_t: NDArray = dataset["x_t"]
        a_t: NDArray = dataset["a_t"]
        r: NDArray = dataset["r"]
        lag_features: NDArray | list[NDArray] = dataset.get("x_t_ls")
        if lag_features is None:
            lag_features_list: list[NDArray] = [dataset["x_t_l"]]
        elif isinstance(lag_features, list):
            lag_features_list: list[NDArray] = lag_features
        else:
            lag_features_list: list[NDArray] = [lag_features[i] for i in range(lag_features.shape[0])]

        folds: list[NDArray] = _make_folds(
            num_data=x_t.shape[0],
            num_folds=self.num_folds,
            random_state=self.random_state,
        )

        if q_hat is None:
            q_hat_list: list[NDArray] = []
            alc_list: list[float] = []
            for lag_features in lag_features_list:
                dataset_lag = dict(dataset)
                dataset_lag["x_t_l"] = lag_features
                q_hat_lag, alc_value, _ = train_reward_model_mtri_crossfit(
                    dataset=dataset_lag,
                    num_folds=self.num_folds,
                    lambda_mtri=lambda_mtri,
                    hidden_dim=mtri_hidden_dim,
                    lr=mtri_lr,
                    batch_size=mtri_batch_size,
                    num_epochs=mtri_epochs,
                    weight_decay=mtri_weight_decay,
                    random_state=self.random_state,
                    folds=folds,
                )
                q_hat_list.append(q_hat_lag)
                alc_list.append(alc_value)
        else:
            if isinstance(q_hat, list):
                q_hat_list = q_hat
            else:
                q_hat_list = [q_hat]
            if len(q_hat_list) != len(lag_features_list):
                raise ValueError("q_hat must match the number of lag feature sets.")
            if alc_values is None:
                alc_list = [0.0 for _ in lag_features_list]
            else:
                alc_list = alc_values
            if len(alc_list) != len(lag_features_list):
                raise ValueError("alc_values must match the number of lag feature sets.")

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

        tau_value = self.tau if tau is None else tau
        alc_arr: NDArray = np.array(alc_list)
        lag_weights: NDArray = np.exp(-alc_arr / max(tau_value, self.log_eps))
        lag_weights: NDArray = lag_weights / lag_weights.sum()
        self.alc_values_ = alc_arr
        self.lag_weights_ = lag_weights

        estimator = LaggedPolicyEstimatorTorch(bandwidth=self.bandwidth, eps=self.log_eps)
        bar_pi_0_list: Deque[NDArray] = deque()
        for lag_features in lag_features_list:
            bar_pi_0_hat = np.zeros((x_t.shape[0], self.num_actions))
            lag_features_tensor = torch.from_numpy(lag_features).float()
            actions_tensor = torch.from_numpy(a_t).long()
            for fold_idx in folds:
                if self.num_folds <= 1:
                    train_idx = fold_idx
                else:
                    mask = np.ones(x_t.shape[0], dtype=bool)
                    mask[fold_idx] = False
                    train_idx = np.where(mask)[0]
                fold_idx_t = torch.from_numpy(fold_idx).long()
                train_idx_t = torch.from_numpy(train_idx).long()
                bar_pi_0_hat[fold_idx] = (
                    estimator.estimate_bar_pi_from_actions(
                        lag_features_tensor[fold_idx_t],
                        lag_features_tensor[train_idx_t],
                        actions_tensor[train_idx_t],
                        self.num_actions,
                    )
                    .detach()
                    .numpy()
                )
            bar_pi_0_hat = np.clip(bar_pi_0_hat, self.log_eps, None)
            bar_pi_0_hat = bar_pi_0_hat / bar_pi_0_hat.sum(axis=1, keepdims=True)
            bar_pi_0_list.append(bar_pi_0_hat)

        self.q_hat_list_ = q_hat_list
        self.bar_pi_0_list_ = bar_pi_0_list
        self.lag_features_list_ = lag_features_list
        self.folds_ = folds

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        q_train, q_test = dataset["q"], dataset_test["q"]
        n = x_t.shape[0]
        x_t_tensor = torch.from_numpy(x_t).float()
        a_t_tensor = torch.from_numpy(a_t).long()
        r_tensor = torch.from_numpy(r).float()
        lag_tensors = [torch.from_numpy(lag).float() for lag in lag_features_list]
        q_hat_tensors = [torch.from_numpy(q_hat_lag).float() for q_hat_lag in q_hat_list]
        bar_pi_0_tensors = [torch.from_numpy(bar_pi_0_hat).float() for bar_pi_0_hat in bar_pi_0_list]
        for _ in range(self.max_iter):
            self.nn_model.train()
            pi_all = self.nn_model(x_t_tensor)
            log_prob_all = torch.log(pi_all + self.log_eps)
            total_term = 0.0
            for fold_idx in folds:
                if self.num_folds <= 1:
                    train_idx = fold_idx
                else:
                    mask = np.ones(n, dtype=bool)
                    mask[fold_idx] = False
                    train_idx = np.where(mask)[0]

                test_idx_t = torch.from_numpy(fold_idx).long()
                train_idx_t = torch.from_numpy(train_idx).long()

                pi_train = pi_all[train_idx_t]
                pi_test = pi_all[test_idx_t]
                log_prob_test = log_prob_all[test_idx_t]

                a_test = a_t_tensor[test_idx_t]
                r_test = r_tensor[test_idx_t]

                fold_term = 0.0
                for lag_idx, lag_tensor in enumerate(lag_tensors):
                    lag_train = lag_tensor[train_idx_t]
                    lag_test = lag_tensor[test_idx_t]
                    bar_pi_theta = estimator.estimate_bar_pi(
                        x_query=lag_test,
                        x_ref=lag_train,
                        pi_ref=pi_train,
                    )
                    bar_pi_0 = bar_pi_0_tensors[lag_idx][test_idx_t]
                    bar_pi_0 = torch.clamp(bar_pi_0, min=self.log_eps)
                    w = torch.clamp((bar_pi_theta / bar_pi_0).detach(), max=self.weight_clip)
                    log_bar_pi = torch.log(bar_pi_theta + self.log_eps)

                    q_hat_test = q_hat_tensors[lag_idx][test_idx_t]
                    idx_local = torch.arange(a_test.shape[0], dtype=torch.long)
                    q_hat_factual = q_hat_test[idx_local, a_test]
                    w_factual = w[idx_local, a_test]
                    log_bar_pi_factual = log_bar_pi[idx_local, a_test]

                    term = w_factual * (r_test - q_hat_factual) * log_bar_pi_factual
                    term += torch.sum(q_hat_test * pi_test * log_prob_test, dim=1)
                    fold_term = fold_term + float(lag_weights[lag_idx]) * term.sum()

                total_term += fold_term

            loss = -total_term / n
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.train_loss.append(float(loss.item()))
            scheduler.step()
            pi_train = self.predict(dataset)
            self.train_value.append((q_train * pi_train).sum(1).mean())
            pi_test = self.predict(dataset_test)
            self.test_value.append((q_test * pi_test).sum(1).mean())

    def _estimate_policy_gradient(
        self,
        x_t: Tensor,
        x_t_l: Tensor,
        a_t: Tensor,
        r: Tensor,
        q_hat: Tensor,
        pi: Tensor,
        lag_ref: Tensor,
        actions_ref: Tensor,
        pi_ref: Tensor,
    ) -> Tensor:
        """Estimate policy gradient.

        Args:
            x_t (Tensor): features
            x_t_l (Tensor): lag features
            a_t (Tensor): actions
            r (Tensor): rewards
            q_hat (Tensor): estimated rewards
            pi (Tensor): current policy
            lag_ref (Tensor): lag reference features
            actions_ref (Tensor): reference actions
            pi_ref (Tensor): reference policy

        Returns:
            Tensor: estimated policy gradient
        """
        current_policy: Tensor = pi.detach()
        log_prob: Tensor = torch.log(pi + self.log_eps)
        idx: Tensor = torch.arange(a_t.shape[0], dtype=torch.long)

        estimator = LaggedPolicyEstimatorTorch(bandwidth=self.bandwidth, eps=self.log_eps)
        bar_pi_theta: Tensor = estimator.estimate_bar_pi(x_query=x_t_l, x_ref=lag_ref, pi_ref=pi_ref)
        bar_pi_0: Tensor = estimator.estimate_bar_pi_from_actions(
            x_query=x_t_l,
            x_ref=lag_ref,
            actions_ref=actions_ref,
            num_actions=self.num_actions,
        )
        bar_pi_0: Tensor = torch.clamp(bar_pi_0, min=self.log_eps)
        w: Tensor = torch.clamp(bar_pi_theta.detach() / bar_pi_0, max=self.weight_clip)
        w_factual: Tensor = w[idx, a_t]

        log_bar_pi = torch.log(bar_pi_theta + self.log_eps)
        log_bar_pi_factual: Tensor = log_bar_pi[idx, a_t]

        q_hat_factual: Tensor = q_hat[idx, a_t]
        estimated_policy_grad: Tensor = w_factual * (r - q_hat_factual) * log_bar_pi_factual
        estimated_policy_grad += torch.sum(q_hat * current_policy * log_prob, dim=1)
        estimated_policy_grad += self.imit_reg * log_prob[idx, a_t]

        return estimated_policy_grad

    def predict(self, dataset_test: NDArray) -> NDArray:
        """Predict the policy.

        Args:
            dataset_test (NDArray): test dataset

        Returns:
            NDArray: predicted policy
        """
        self.nn_model.eval()
        x_t: Tensor = torch.from_numpy(dataset_test["x_t"]).float()
        pi: NDArray = self.nn_model(x_t).detach().numpy()
        return pi
