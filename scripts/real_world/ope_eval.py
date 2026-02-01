from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from estimating import fit_predict_by_MLP_actionwise_crossfit
from ope import calc_dm, calc_ips, calc_dr, calc_dolce


@dataclass(frozen=True)
class LoggedDataBundle:
    """Bundle of logged data for training and evaluation."""

    df_train: pd.DataFrame
    df_eval: pd.DataFrame
    logged_train: Dict[str, NDArray[np.float32]]
    logged_eval: Dict[str, NDArray[np.float32]]
    a_train: NDArray[np.int32]
    a_eval: NDArray[np.int32]
    cur_cols: List[str]
    lag_cols_list: List[List[str]]


def split_train_eval(bandit: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the bandit data into training and evaluation sets.

    Args:
        bandit (pd.DataFrame): The bandit data.
        test_size (float): The size of the evaluation set.
        seed (int): The random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and evaluation sets.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, eval_idx = next(gss.split(bandit, groups=bandit["stay_id"]))
    df_train = bandit.iloc[train_idx].copy().reset_index(drop=True)
    df_eval = bandit.iloc[eval_idx].copy().reset_index(drop=True)
    return df_train, df_eval


def _feature_columns(lags: Sequence[int]) -> Tuple[List[str], List[List[str]]]:
    """Get the feature columns.

    Args:
        lags (Sequence[int]): The lags.

    Returns:
        Tuple[List[str], List[List[str]]]: The feature columns.
    """
    cur_cols = ["hr", "spo2", "sbp", "dbp", "map"]
    lag_cols_list = [[f"{c}_lag{lag}h" for c in cur_cols] for lag in lags]
    return cur_cols, lag_cols_list


def _drop_na_rows(df: pd.DataFrame, cur_cols: Sequence[str], lag_cols_list: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Drop rows with missing values.

    Args:
        df (pd.DataFrame): The dataframe.
        cur_cols (Sequence[str]): The current columns.
        lag_cols_list (Sequence[Sequence[str]]): The lag columns.

    Returns:
        pd.DataFrame: The dataframe with missing rows dropped.
    """
    cols = list(cur_cols) + [c for cols in lag_cols_list for c in cols] + ["a", "r"]
    return df.dropna(subset=cols).copy()


def _to_base_names(df_part: pd.DataFrame, cur_cols: Sequence[str]) -> pd.DataFrame:
    """Convert the column names to base names.

    Args:
        df_part (pd.DataFrame): The dataframe.
        cur_cols (Sequence[str]): The current columns.

    Returns:
        pd.DataFrame: The dataframe with the column names converted to base names.
    """
    out = df_part.copy()
    out.columns = [c.split("_lag")[0] if "_lag" in c else c for c in out.columns]
    out = out.reindex(columns=list(cur_cols))
    return out


def make_X(
    df_: pd.DataFrame,
    cur_cols: Sequence[str],
    lag_cols_list: Sequence[Sequence[str]],
    scaler: StandardScaler,
) -> Tuple[NDArray[np.float32], List[NDArray[np.float32]]]:
    """Make the X matrix.

    Args:
        df_ (pd.DataFrame): The dataframe.
        cur_cols (Sequence[str]): The current columns.
        lag_cols_list (Sequence[Sequence[str]]): The lag columns.
        scaler (StandardScaler): The scaler.

    Returns:
        Tuple[NDArray[np.float32], List[NDArray[np.float32]]]: The X matrix.
    """
    X_cur = scaler.transform(df_[list(cur_cols)].astype(float))

    X_lags = []
    for cols in lag_cols_list:
        lag_df = _to_base_names(df_[list(cols)].astype(float), cur_cols)
        X_lag = scaler.transform(lag_df)
        X_lags.append(X_lag)

    X_cur = np.nan_to_num(X_cur, nan=0.0).astype(np.float32)
    X_lags = [np.nan_to_num(X, nan=0.0).astype(np.float32) for X in X_lags]
    return X_cur, X_lags


def crossfit_pi0_binary(
    X: NDArray[np.float32],
    a: NDArray[np.int32],
    groups: NDArray[np.int32],
    n_splits: int = 5,
    eps: float = 1e-3,
) -> NDArray[np.float32]:
    """Crossfit the pi0 matrix.

    Args:
        X (NDArray[np.float32]): The X matrix.
        a (NDArray[np.int32]): The actions.
        groups (NDArray[np.int32]): The groups.
        n_splits (int, optional): The number of splits. Defaults to 5.
        eps (float, optional): The epsilon. Defaults to 1e-3.

    Returns:
        NDArray[np.float32]: The pi0 matrix.
    """
    gkf = GroupKFold(n_splits=n_splits)
    base = LogisticRegression(max_iter=2000, solver="lbfgs")

    p = np.zeros(len(a), dtype=float)
    for tr, te in gkf.split(X, a, groups=groups):
        m = clone(base)
        m.fit(X[tr], a[tr])
        p[te] = m.predict_proba(X[te])[:, 1]

    p = np.clip(p, eps, 1 - eps)
    pi0 = np.column_stack([1 - p, p]).astype(np.float32)
    return pi0


def make_logged_dict(
    X_cur: NDArray[np.float32],
    X_lags: List[NDArray[np.float32]],
    a: NDArray[np.int32],
    r: NDArray[np.float32],
    pi0: NDArray[np.float32],
    q_dummy: NDArray[np.float32] | None = None,
) -> Dict[str, NDArray[np.float32]]:
    """Make the logged dictionary.

    Args:
        X_cur (NDArray[np.float32]): The current X matrix.
        X_lags (List[NDArray[np.float32]]): The lag X matrices.
        a (NDArray[np.int32]): The actions.
        r (NDArray[np.float32]): The rewards.
        pi0 (NDArray[np.float32]): The pi0 matrix.
        q_dummy (NDArray[np.float32] | None): The dummy q matrix.

    Returns:
        Dict[str, NDArray[np.float32]]: The logged dictionary.
    """
    d = {
        "x_t": X_cur,
        "x_t_l": X_lags[0] if X_lags else X_cur,
        "x_t_ls": X_lags,
        "a_t": a.astype(int),
        "r": r.astype(np.float32),
        "pi_0": pi0.astype(np.float32),
        "num_data": len(a),
        "num_actions": pi0.shape[1],
    }
    if q_dummy is not None:
        d["q"] = q_dummy.astype(np.float32)
    return d


def prepare_logged_data(
    bandit: pd.DataFrame,
    lags: Sequence[int],
    test_size: float = 0.3,
    seed: int = 0,
) -> LoggedDataBundle:
    """Prepare the logged data.

    Args:
        bandit (pd.DataFrame): The bandit data.
        lags (Sequence[int]): The lags.
        test_size (float, optional): The size of the evaluation set. Defaults to 0.3.
        seed (int, optional): The random seed. Defaults to 0.

    Returns:
        LoggedDataBundle: The logged data bundle.
    """
    bandit = bandit.sort_values(["stay_id", "t"]).reset_index(drop=True)
    df_train, df_eval = split_train_eval(bandit, test_size=test_size, seed=seed)

    cur_cols, lag_cols_list = _feature_columns(lags)
    need_cols = ["stay_id", "t", "a", "r"] + cur_cols + [c for cols in lag_cols_list for c in cols]
    missing = [c for c in need_cols if c not in bandit.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_train = _drop_na_rows(df_train, cur_cols, lag_cols_list).reset_index(drop=True)
    df_eval = _drop_na_rows(df_eval, cur_cols, lag_cols_list).reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(df_train[cur_cols].astype(float))

    X_cur_train, X_lags_train = make_X(df_train, cur_cols, lag_cols_list, scaler)
    X_cur_eval, X_lags_eval = make_X(df_eval, cur_cols, lag_cols_list, scaler)

    a_train = df_train["a"].astype(int).to_numpy()
    r_train = df_train["r"].astype(float).to_numpy().astype(np.float32)

    a_eval = df_eval["a"].astype(int).to_numpy()
    r_eval = df_eval["r"].astype(float).to_numpy().astype(np.float32)

    num_actions = int(np.max(bandit["a"])) + 1
    if num_actions != 2:
        raise ValueError("Expected binary action (0/1) for this pipeline.")

    pi0_train = crossfit_pi0_binary(
        X_cur_train,
        a_train,
        groups=df_train["stay_id"].to_numpy(),
        n_splits=5,
        eps=1e-3,
    )

    prop_model = LogisticRegression(max_iter=2000, solver="lbfgs")
    prop_model.fit(X_cur_train, a_train)
    p_eval = prop_model.predict_proba(X_cur_eval)[:, 1]
    p_eval = np.clip(p_eval, 1e-3, 1 - 1e-3)
    pi0_eval = np.column_stack([1 - p_eval, p_eval]).astype(np.float32)

    q_dummy_train = np.zeros((len(a_train), num_actions), dtype=np.float32)
    q_dummy_eval = np.zeros((len(a_eval), num_actions), dtype=np.float32)

    logged_train = make_logged_dict(X_cur_train, X_lags_train, a_train, r_train, pi0_train, q_dummy=q_dummy_train)
    logged_eval = make_logged_dict(X_cur_eval, X_lags_eval, a_eval, r_eval, pi0_eval, q_dummy=q_dummy_eval)

    return LoggedDataBundle(
        df_train=df_train,
        df_eval=df_eval,
        logged_train=logged_train,
        logged_eval=logged_eval,
        a_train=a_train,
        a_eval=a_eval,
        cur_cols=cur_cols,
        lag_cols_list=lag_cols_list,
    )


def _sigmoid(z: NDArray[np.float32]) -> NDArray[np.float32]:
    """Sigmoid function.

    Args:
        z (NDArray[np.float32]): The input.

    Returns:
        NDArray[np.float32]: The output.
    """
    return 1.0 / (1.0 + np.exp(-z))


def make_target_pi_from_rule(df_raw: pd.DataFrame, eps: float = 1e-3) -> NDArray[np.float32]:
    """Make the target policy from the rule.

    Args:
        df_raw (pd.DataFrame): The raw dataframe.
        eps (float, optional): The epsilon. Defaults to 1e-3.

    Returns:
        NDArray[np.float32]: The target policy.
    """
    map_ = df_raw["map"].astype(float).to_numpy()
    hr = df_raw["hr"].astype(float).to_numpy()

    score = (65.0 - map_) / 5.0 + (hr - 110.0) / 30.0
    p1 = _sigmoid(score)
    p1 = np.clip(p1, eps, 1 - eps)
    pi = np.column_stack([1 - p1, p1]).astype(np.float32)
    return pi


def ess_from_weights(w: NDArray[np.float32]) -> float:
    """Calculate the effective sample size from the weights.

    Args:
        w (NDArray[np.float32]): The weights.

    Returns:
        float: The effective sample size.
    """
    w = np.asarray(w, float)
    return float((w.sum() ** 2) / (np.sum(w**2) + 1e-12))


def summarize_estimator(est_vec: NDArray[np.float32], alpha: float = 0.05) -> Tuple[float, float, Tuple[float, float]]:
    """Summarize the estimator.

    Args:
        est_vec (NDArray[np.float32]): The estimator vector.
        alpha (float, optional): The alpha. Defaults to 0.05.

    Returns:
        Tuple[float, float, Tuple[float, float]]: The summary.
    """
    n = len(est_vec)
    mu = float(np.mean(est_vec))
    se = float(np.std(est_vec, ddof=1) / np.sqrt(n))
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci = (mu - tcrit * se, mu + tcrit * se)
    return mu, se, ci


def run_ope_estimators(
    logged_eval: Dict[str, NDArray[np.float32]],
    pi_target_eval: NDArray[np.float32],
    a_eval: NDArray[np.int32],
    random_state: int = 0,
) -> pd.DataFrame:
    """Run the OPE estimators.

    Args:
        logged_eval (Dict[str, NDArray[np.float32]]): The logged data.
        pi_target_eval (NDArray[np.float32]): The target policy.
        a_eval (NDArray[np.int32]): The actions.
        random_state (int, optional): The random state. Defaults to 0.

    Returns:
        pd.DataFrame: The OPE results.
    """
    q_hat_eval = fit_predict_by_MLP_actionwise_crossfit(
        features=logged_eval["x_t"],
        actions=logged_eval["a_t"],
        rewards=logged_eval["r"],
        num_actions=logged_eval["num_actions"],
        random_state=random_state,
    )

    est_dm, info_dm = calc_dm(logged_eval, pi_target_eval, q_hat=q_hat_eval)
    est_ips, info_ips = calc_ips(logged_eval, pi_target_eval)
    est_dr, info_dr = calc_dr(logged_eval, pi_target_eval, q_hat=q_hat_eval)
    est_dol, info_dol = calc_dolce(logged_eval, pi_target_eval, random_state=random_state)

    out = []
    for name, est, info in [
        ("DM", est_dm, info_dm),
        ("IPS", est_ips, info_ips),
        ("DR", est_dr, info_dr),
        ("DOLCE", est_dol, info_dol),
    ]:
        vhat, se, (lo, hi) = summarize_estimator(est)

        if name in ["IPS", "DR"]:
            w = pi_target_eval[np.arange(len(a_eval)), a_eval] / logged_eval["pi_0"][np.arange(len(a_eval)), a_eval]
            ess = ess_from_weights(w)
        elif name == "DOLCE" and isinstance(info, dict) and ("w" in info) and ("alpha" in info):
            w_eff = 0.0
            for k, wk in enumerate(info["w"]):
                w_eff = w_eff + float(info["alpha"][k]) * wk[np.arange(len(a_eval)), a_eval]
            ess = ess_from_weights(w_eff)
        else:
            ess = float(len(est))

        out.append([name, vhat, se, lo, hi, ess])

    res_ope = pd.DataFrame(out, columns=["estimator", "V_hat", "SE", "CI_low", "CI_high", "ESS"])
    return res_ope
