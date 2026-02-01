from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata  # type: ignore
import torch
from torch.types import Tensor


def parse_comma_separated_list(raw: str) -> List:
    """Parse a comma-separated list of floats.

    Args:
        raw (str): A string containing a comma-separated list of floats.

    Raises:
        ValueError: If the list of values is empty.

    Returns:
        List[float]: A list of floats parsed from the input string.
    """
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        values.append(value)
    if not values:
        raise ValueError("values are empty")
    return values


def flatten_grads(model: torch.nn.Module) -> NDArray:
    """Flatten the gradients of a model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        NDArray: The flattened gradients.
    """
    grads: List[Tensor] = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).view(-1))
        else:
            grads.append(param.grad.detach().view(-1))
    return torch.cat(grads).cpu().numpy()


def eps_greedy_policy(
    q_func: NDArray,
    k: int = 1,
    eps: float = 0.1,
) -> NDArray:
    """Define an epsilon-greedy policy over actions.

    Args:
        q_func (NDArray): A 2D array of shape (n_samples, n_actions) representing the Q-values.
        k (int, optional): The number of top actions to consider. Defaults to 1.
        eps (float, optional): The exploration parameter. Defaults to 0.1.

    Returns:
        NDArray: A 2D array of shape (n_samples, n_actions) representing the policy.
    """
    is_topk: NDArray = rankdata(-q_func, method="ordinal", axis=1) <= k
    pi: NDArray = ((1.0 - eps) / k) * is_topk + eps / q_func.shape[1]
    return pi / pi.sum(1)[:, np.newaxis]


def sample_action(pi: NDArray, random_state: int = 42) -> NDArray:
    """Sample an action from a categorical policy π for each row.

    Args:
        pi (NDArray): A 2D array of shape (n_samples, n_actions) representing the policy.
        random_state (int, optional): The random state. Defaults to 42.

    Returns:
        NDArray: A 1D array of shape (n_samples,) representing the sampled actions.
    """
    rng = np.random.default_rng(int(random_state))
    uniform_rvs: NDArray = rng.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi: NDArray = pi.cumsum(axis=1)
    flg: NDArray = cum_pi > uniform_rvs
    actions: NDArray = flg.argmax(axis=1)
    return actions


def apply_flat_grad(model: torch.nn.Module, flat_grad: NDArray, step_size: float) -> None:
    """Apply a flat gradient to a model.

    Args:
        model (torch.nn.Module): The model.
        flat_grad (NDArray): The flat gradient.
        step_size (float): The step size.
    """
    offset: int = 0
    for param in model.parameters():
        numel = param.numel()
        grad_slice = flat_grad[offset : offset + numel].reshape(param.shape)  # noqa: E203
        param.data.add_(torch.from_numpy(grad_slice).to(param.data) * step_size)
        offset += numel
