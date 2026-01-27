from __future__ import annotations

import argparse
import pickle
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_random_state
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from estimating import LaggedPolicyEstimatorTorch  # noqa: E402
from estimating import train_reward_model_mtri_crossfit  # noqa: E402
from synthetic import generate_synthetic_data  # noqa: E402
from opl import RegressionBasedPolicyLearner, GradientBasedPolicyLearner, DOLCE  # noqa: E402

warnings.filterwarnings("ignore")


def _parse_ratio_list(raw: str) -> list[float]:
    vals = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(float(p) / 100.0)
    if not vals:
        raise ValueError("ratios are empty")
    return vals


def _make_folds(num_data: int, num_folds: int, random_state: int) -> list[np.ndarray]:
    if num_folds <= 1:
        return [np.arange(num_data)]
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(num_data)
    return [np.sort(fold) for fold in np.array_split(perm, num_folds)]


def _flatten_grads(model: torch.nn.Module) -> np.ndarray:
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).view(-1))
        else:
            grads.append(param.grad.detach().view(-1))
    return torch.cat(grads).cpu().numpy()


def _estimate_true_gradient(model: torch.nn.Module, x_t: np.ndarray, q: np.ndarray) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    x_t_tensor = torch.from_numpy(x_t).float()
    q_tensor = torch.from_numpy(q).float()
    pi = model(x_t_tensor)
    value = (pi * q_tensor).sum(1).mean()
    value.backward()
    return _flatten_grads(model)


def _estimate_ips_gradient(
    model: torch.nn.Module,
    x_t: np.ndarray,
    a_t: np.ndarray,
    r: np.ndarray,
    pi_0: np.ndarray,
    log_eps: float,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    x_t_tensor = torch.from_numpy(x_t).float()
    a_t_tensor = torch.from_numpy(a_t).long()
    r_tensor = torch.from_numpy(r).float()
    pi_0_tensor = torch.from_numpy(pi_0).float()
    pi = model(x_t_tensor)
    log_prob = torch.log(pi + log_eps)
    idx = torch.arange(a_t_tensor.shape[0], dtype=torch.long)
    w = (pi[idx, a_t_tensor] / pi_0_tensor[idx, a_t_tensor]).detach()
    objective = (w * r_tensor * log_prob[idx, a_t_tensor]).mean()
    objective.backward()
    return _flatten_grads(model)


def _estimate_dr_gradient(
    model: torch.nn.Module,
    x_t: np.ndarray,
    a_t: np.ndarray,
    r: np.ndarray,
    pi_0: np.ndarray,
    q_hat: np.ndarray,
    log_eps: float,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    x_t_tensor = torch.from_numpy(x_t).float()
    a_t_tensor = torch.from_numpy(a_t).long()
    r_tensor = torch.from_numpy(r).float()
    pi_0_tensor = torch.from_numpy(pi_0).float()
    q_hat_tensor = torch.from_numpy(q_hat).float()
    pi = model(x_t_tensor)
    log_prob = torch.log(pi + log_eps)
    idx = torch.arange(a_t_tensor.shape[0], dtype=torch.long)
    w = (pi[idx, a_t_tensor] / pi_0_tensor[idx, a_t_tensor]).detach()
    q_hat_factual = q_hat_tensor[idx, a_t_tensor]
    term1 = (w * (r_tensor - q_hat_factual) * log_prob[idx, a_t_tensor]).mean()
    term2 = torch.sum(q_hat_tensor * pi.detach() * log_prob, dim=1).mean()
    objective = term1 + term2
    objective.backward()
    return _flatten_grads(model)


def _estimate_dolce_gradient(
    dolce: DOLCE,
    logged_data: dict,
    lambda_mtri: float = 1.0,
) -> np.ndarray:
    """
    Re-construct the DOLCE gradient estimator on the logged data, using the same cross-fitting structure.

    Note: This is only used for diagnostic "grad MSE" plots.
    """
    model = dolce.nn_model
    model.zero_grad(set_to_none=True)
    x_t = logged_data["x_t"]
    a_t = logged_data["a_t"]
    r = logged_data["r"]
    num_actions = logged_data["num_actions"]

    lag_features = logged_data.get("x_t_ls")
    if lag_features is None:
        lag_features_list = [logged_data["x_t_l"]]
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
            dataset_lag = dict(logged_data)
            dataset_lag["x_t_l"] = lag
            q_hat_lag, alc_value, _ = train_reward_model_mtri_crossfit(
                dataset=dataset_lag,
                num_folds=dolce.num_folds,
                lambda_mtri=lambda_mtri,
                random_state=dolce.random_state,
                folds=folds,
            )
            q_hat_list.append(q_hat_lag)
            alc_values.append(alc_value)

    # bar_pi_0
    bar_pi_0_list = getattr(dolce, "bar_pi_0_list_", None)
    if bar_pi_0_list is None:
        estimator = LaggedPolicyEstimatorTorch(bandwidth=dolce.bandwidth, eps=dolce.log_eps)
        bar_pi_0_list = []
        actions_tensor = torch.from_numpy(a_t).long()
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
            w = torch.clamp((bar_pi_theta / bar_pi_0).detach(), max=dolce.weight_clip)
            log_bar_pi = torch.log(bar_pi_theta + dolce.log_eps)
            log_bar_pi_factual = log_bar_pi[torch.arange(test_idx_t.shape[0]), a_test]

            term = term + w[torch.arange(test_idx_t.shape[0]), a_test] * (r_test - q_hat_factual) * log_bar_pi_factual
            lag_contrib[test_idx_t] = term

        total = total + float(lag_weights[lag_idx]) * lag_contrib.mean()

    total.backward()
    return _flatten_grads(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="OPL simulation for support violation ratios.")
    parser.add_argument("--num-sim", type=int, default=30)
    parser.add_argument("--num-data", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--test-data-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42, help="data seed base (contexts + reward noise)")
    parser.add_argument("--env-seed", type=int, default=7, help="environment seed (reward function)")
    parser.add_argument("--test-seed", type=int, default=999, help="test context seed")
    parser.add_argument("--logging-eps", type=float, default=0.2, help="exploration floor in logging policy")
    parser.add_argument(
        "--x-dep",
        type=float,
        default=0.0,
        help="dependence strength of x_t on x_{t-l} (0 = independent)",
    )

    parser.add_argument(
        "--ratios",
        type=str,
        default="0,30,60,90",
        help="comma-separated percent values",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result_df_support_violation_opl.pkl",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default="",
        help="optional path for raw per-simulation results",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="optional directory to save plots",
    )
    parser.add_argument("--show-summary", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    _ = check_random_state(args.seed)

    ratio_list = _parse_ratio_list(args.ratios)
    raw_rows = []

    # Use the same test contexts across ratios and sims; only pi_0 changes with ratio.
    for non_overlap_ratio in ratio_list:
        test_data = generate_synthetic_data(
            num_data=args.test_data_size,
            num_features=args.num_features,
            num_actions=args.num_actions,
            non_overlap_ratio=non_overlap_ratio,
            lambda_=args.lambda_,
            eta=args.eta,
            random_state=args.test_seed,
            env_random_state=args.env_seed,
            logging_eps=args.logging_eps,
            x_t_dep=args.x_dep,
        )
        v_pi_0 = float((test_data["q"] * test_data["pi_0"]).sum(1).mean())
        v_star = float(test_data["q"].max(axis=1).mean())
        denom = v_star - v_pi_0

        for sim in tqdm(range(args.num_sim), desc=f"support_violation={int(non_overlap_ratio*100)}"):
            data_seed = args.seed + sim * 100
            logged_data = generate_synthetic_data(
                num_data=args.num_data,
                num_features=args.num_features,
                num_actions=args.num_actions,
                non_overlap_ratio=non_overlap_ratio,
                lambda_=args.lambda_,
                eta=args.eta,
                random_state=data_seed,
                env_random_state=args.env_seed,
                logging_eps=args.logging_eps,
                x_t_dep=args.x_dep,
            )

            # DM (regression-based)
            reg = RegressionBasedPolicyLearner(
                num_features=args.num_features,
                num_actions=args.num_actions,
                max_iter=args.num_epochs,
                random_state=data_seed,
            )
            reg.fit(logged_data, test_data)
            pi_reg = reg.predict(test_data)
            value_dm = float((test_data["q"] * pi_reg).sum(1).mean())

            # IPS
            ips = GradientBasedPolicyLearner(
                num_features=args.num_features,
                num_actions=args.num_actions,
                max_iter=args.num_epochs,
                random_state=data_seed,
            )
            ips.fit(logged_data, test_data)
            pi_ips = ips.predict(test_data)
            value_ips = float((test_data["q"] * pi_ips).sum(1).mean())

            # DR
            dr = GradientBasedPolicyLearner(
                num_features=args.num_features,
                num_actions=args.num_actions,
                max_iter=args.num_epochs,
                random_state=data_seed,
            )
            reg_dr = RegressionBasedPolicyLearner(
                num_features=args.num_features,
                num_actions=args.num_actions,
                max_iter=args.num_epochs,
                random_state=data_seed,
            )
            reg_dr.fit(logged_data, test_data)
            q_hat = reg_dr.predict_q(logged_data)
            dr.fit(logged_data, test_data, q_hat=q_hat)
            pi_dr = dr.predict(test_data)
            value_dr = float((test_data["q"] * pi_dr).sum(1).mean())

            # DOLCE
            dolce = DOLCE(
                num_features=args.num_features,
                num_actions=args.num_actions,
                max_iter=args.num_epochs,
                random_state=data_seed,
            )
            dolce.fit(logged_data, test_data)
            pi_dolce = dolce.predict(test_data)
            value_dolce = float((test_data["q"] * pi_dolce).sum(1).mean())

            metrics = {
                "DM": value_dm,
                "IPS": value_ips,
                "DR": value_dr,
                "DOLCE": value_dolce,
            }

            x_test = test_data["x_t"]
            q_test = test_data["q"]
            x_logged = logged_data["x_t"]
            a_logged = logged_data["a_t"]
            r_logged = logged_data["r"]
            pi_0_logged = logged_data["pi_0"]

            grad_true_ips = _estimate_true_gradient(ips.nn_model, x_test, q_test)
            grad_hat_ips = _estimate_ips_gradient(
                ips.nn_model,
                x_logged,
                a_logged,
                r_logged,
                pi_0_logged,
                ips.log_eps,
            )
            grad_mse_ips = float(np.mean((grad_hat_ips - grad_true_ips) ** 2))

            grad_true_dr = _estimate_true_gradient(dr.nn_model, x_test, q_test)
            grad_hat_dr = _estimate_dr_gradient(
                dr.nn_model,
                x_logged,
                a_logged,
                r_logged,
                pi_0_logged,
                q_hat,
                dr.log_eps,
            )
            grad_mse_dr = float(np.mean((grad_hat_dr - grad_true_dr) ** 2))

            grad_true_dolce = _estimate_true_gradient(dolce.nn_model, x_test, q_test)
            grad_hat_dolce = _estimate_dolce_gradient(dolce, logged_data)
            grad_mse_dolce = float(np.mean((grad_hat_dolce - grad_true_dolce) ** 2))

            grad_mse_map = {
                "IPS": grad_mse_ips,
                "DR": grad_mse_dr,
                "DOLCE": grad_mse_dolce,
                "DM": np.nan,
            }

            for method, value in metrics.items():
                ni = np.nan if denom == 0 else (value - v_pi_0) / denom
                raw_rows.append(
                    dict(
                        support_violation_ratio=int(non_overlap_ratio * 100),
                        method=method,
                        value=value,
                        ni=ni,
                        regret=v_star - value,
                        win_rate=value > v_pi_0,
                        grad_mse=grad_mse_map[method],
                        v_star=v_star,
                        v_pi_0=v_pi_0,
                        num_data=args.num_data,
                        num_actions=args.num_actions,
                        lambda_=args.lambda_,
                        eta=args.eta,
                        env_seed=args.env_seed,
                        logging_eps=args.logging_eps,
                    )
                )

    raw_df = pd.DataFrame(raw_rows)
    grouped = raw_df.groupby(["support_violation_ratio", "method"])
    summary_df = grouped.agg(
        ni_median=("ni", "median"),
        ni_q1=("ni", lambda x: float(np.quantile(x, 0.25))),
        ni_q3=("ni", lambda x: float(np.quantile(x, 0.75))),
        regret_mean=("regret", "mean"),
        win_rate=("win_rate", "mean"),
        value_mean=("value", "mean"),
        grad_mse_mean=("grad_mse", "mean"),
        v_star=("v_star", "first"),
        v_pi_0=("v_pi_0", "first"),
        num_data=("num_data", "first"),
        num_actions=("num_actions", "first"),
        lambda_=("lambda_", "first"),
        eta=("eta", "first"),
        env_seed=("env_seed", "first"),
        logging_eps=("logging_eps", "first"),
    ).reset_index()

    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(summary_df, f)

    if args.output_raw:
        raw_output_path = Path(args.output_raw)
        with open(raw_output_path, "wb") as f:
            pickle.dump(raw_df, f)

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        for method in summary_df["method"].unique():
            method_df = summary_df[summary_df["method"] == method].sort_values("support_violation_ratio")
            ax.plot(method_df["support_violation_ratio"], method_df["ni_median"], marker="o", label=method)
            ax.fill_between(method_df["support_violation_ratio"], method_df["ni_q1"], method_df["ni_q3"], alpha=0.2)
        ax.set_xlabel("support violation ratio (%)")
        ax.set_ylabel("Normalized Improvement (median)")
        ax.set_title("OPL: NI vs support violation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "opl_ni.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        for method in summary_df["method"].unique():
            method_df = summary_df[summary_df["method"] == method].sort_values("support_violation_ratio")
            ax.plot(method_df["support_violation_ratio"], method_df["regret_mean"], marker="o", label=method)
        ax.set_xlabel("support violation ratio (%)")
        ax.set_ylabel("Regret (mean)")
        ax.set_title("OPL: regret vs support violation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "opl_regret.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        for method in summary_df["method"].unique():
            method_df = summary_df[summary_df["method"] == method].sort_values("support_violation_ratio")
            ax.plot(method_df["support_violation_ratio"], method_df["grad_mse_mean"], marker="o", label=method)
        ax.set_xlabel("support violation ratio (%)")
        ax.set_ylabel("Gradient MSE (mean)")
        ax.set_title("OPL: gradient MSE vs support violation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "opl_grad_mse.png", dpi=150)
        plt.close(fig)

    if args.show_summary:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
