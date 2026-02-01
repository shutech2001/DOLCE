from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import sys
from typing import List, Tuple
import warnings

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t  # type: ignore
from tqdm import tqdm  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic import generate_synthetic_data, calc_true_value  # noqa: E402
from estimating import fit_predict_by_MLP_actionwise_crossfit  # noqa: E402
from ope import calc_dm, calc_ips, calc_dr, calc_dolce  # noqa: E402
from utils import eps_greedy_policy, parse_comma_separated_list  # noqa: E402

warnings.filterwarnings("ignore")

SWEEP_DEFAULT_VALUES = {
    "support_violation": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    "num_data": "500,1000,3000,5000,7000,10000",
    "num_actions": "2,5,10,30,50,100",
    "lambda": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    "eta": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
}
SWEEP_X_LABELS = {
    "support_violation": "support violation ratio",
    "num_data": "logged data size",
    "num_actions": "number of actions",
    "lambda": r"$\lambda$",
    "eta": r"$\eta$",
}
SWEEP_X_COLUMNS = {
    "support_violation": "support_violation_ratio",
    "num_data": "num_data",
    "num_actions": "num_actions",
    "lambda": "lambda_",
    "eta": "eta",
}
SWEEP_VALUE_CAST = {
    "support_violation": float,
    "num_data": int,
    "num_actions": int,
    "lambda": float,
    "eta": float,
}


def _resolve_sweep_values(raw: str | None, sweep: str) -> List[int | float]:
    """Resolve the sweep values.

    Args:
        raw (str | None): The raw values.
        sweep (str): The sweep.

    Returns:
        List[int | float]: The resolved values.
    """
    if raw is None:
        raw = SWEEP_DEFAULT_VALUES[sweep]
    values = parse_comma_separated_list(raw)
    cast = SWEEP_VALUE_CAST[sweep]
    if cast is int:
        return [int(round(v)) for v in values]
    return [float(np.round(v, 10)) for v in values]


def _resolve_config(args: argparse.Namespace, sweep_value: int | float) -> Tuple[int, int, float, float, float]:
    """Resolve the config.

    Args:
        args (argparse.Namespace): The arguments.
        sweep_value (int | float): The sweep value.

    Returns:
        Tuple[int, int, float, float, float]: The resolved config.
    """
    num_data = args.num_data
    num_actions = args.num_actions
    lambda_ = args.lambda_
    eta = args.eta
    non_overlap_ratio = args.support_violation

    if args.sweep == "support_violation":
        non_overlap_ratio = float(sweep_value)
    elif args.sweep == "num_data":
        num_data = int(sweep_value)
    elif args.sweep == "num_actions":
        num_actions = int(sweep_value)
    elif args.sweep == "lambda":
        lambda_ = float(sweep_value)
    elif args.sweep == "eta":
        eta = float(sweep_value)

    return num_data, num_actions, lambda_, eta, non_overlap_ratio


def _x_value(sweep: str, value: int | float) -> int | float:
    """Convert the value to an x value.

    Args:
        sweep (str): The sweep.
        value (int | float): The value.

    Returns:
        int | float: The x value.
    """
    if sweep == "support_violation":
        return int(round(float(value) * 100))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="OPE simulation for support violation and parameter sweeps.")
    parser.add_argument(
        "--sweep",
        type=str,
        default="support_violation",
        choices=sorted(SWEEP_DEFAULT_VALUES.keys()),
        help="parameter to sweep",
    )
    parser.add_argument("--num-sim", type=int, default=100, help="number of simulations")
    parser.add_argument("--num-data", type=int, default=1000, help="logged data size")
    parser.add_argument("--num-features", type=int, default=5, help="number of features")
    parser.add_argument("--num-actions", type=int, default=5, help="number of actions")
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.5,
        help="mixture weight for reward as function of current vs. lagged features",
    )
    parser.add_argument("--eta", type=float, default=0.0, help="weight for u(x_t, x_t_l, a) interaction term in reward")
    parser.add_argument("--seed", type=int, default=42, help="data seed base (contexts + reward noise)")
    parser.add_argument(
        "--support-violation",
        type=float,
        default=0.0,
        help="support violation ratio when sweep != support_violation",
    )
    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="comma-separated values for sweep (default depends on --sweep)",
    )
    parser.add_argument("--num-test-data", type=int, default=200000, help="test data size")
    parser.add_argument("--test-seed", type=int, default=999, help="test context seed")
    parser.add_argument("--env-seed", type=int, default=7, help="environment seed (reward function)")
    parser.add_argument(
        "--x-dep", type=float, default=1.0, help="dependence strength of x_t on x_{t-l} (0 = independent)"
    )
    parser.add_argument("--lag-scale", type=float, default=2.0, help="scale factor for lag reward component (h)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output path for summary results",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default=None,
        help="optional path for raw per-simulation results",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="results/plots",
        help="optional directory to save plots",
    )
    parser.add_argument("--show-summary", action="store_true", help="show summary of results")
    args = parser.parse_args()

    sweep_values = _resolve_sweep_values(args.values, args.sweep)
    x_col = SWEEP_X_COLUMNS[args.sweep]
    x_label = SWEEP_X_LABELS[args.sweep]

    if args.output is None:
        args.output = f"results/result_df_{args.sweep}_ope.pkl"
    if args.output_raw is None:
        args.output_raw = f"results/result_df_{args.sweep}_ope_raw.pkl"

    true_value_cache: dict = {}

    def _get_true_value(num_actions: int, lambda_: float, eta: float) -> float:
        key = (num_actions, lambda_, eta)
        if key not in true_value_cache:
            true_value_cache[key] = calc_true_value(
                num_features=args.num_features,
                num_actions=num_actions,
                lambda_=lambda_,
                eta=eta,
                num_mc=args.num_test_data,
                random_state=args.test_seed,
                env_random_state=args.env_seed,
                x_t_dep=args.x_dep,
                lag_scale=args.lag_scale,
            )
        return float(true_value_cache[key])

    raw_rows = []
    for value in sweep_values:
        num_data, num_actions, lambda_, eta, non_overlap_ratio = _resolve_config(args, value)
        support_violation_ratio = int(round(non_overlap_ratio * 100))
        true_value = _get_true_value(num_actions, lambda_, eta)
        desc_value = _x_value(args.sweep, value)

        for sim in tqdm(range(args.num_sim), desc=f"{args.sweep}={desc_value}"):
            data_seed = args.seed + sim * 100

            logged_data = generate_synthetic_data(
                num_data=num_data,
                num_features=args.num_features,
                num_actions=num_actions,
                non_overlap_ratio=non_overlap_ratio,
                lambda_=lambda_,
                eta=eta,
                random_state=data_seed,
                env_random_state=args.env_seed,
                x_t_dep=args.x_dep,
                lag_scale=args.lag_scale,
            )

            # Target policy uses only current context (theory assumption).
            q_for_pi = logged_data.get("g_x_t_a_t", logged_data["q"])
            pi = eps_greedy_policy(q_for_pi)

            q_hat = fit_predict_by_MLP_actionwise_crossfit(
                features=logged_data["x_t"],
                actions=logged_data["a_t"],
                rewards=logged_data["r"],
                num_actions=logged_data["num_actions"],
                random_state=data_seed,
                n_folds=2,
            )

            dolce_contrib, dolce_info = calc_dolce(logged_data, pi, random_state=data_seed)
            contributions = {
                "DM": calc_dm(logged_data, pi, q_hat=q_hat)[0],
                "IPS": calc_ips(logged_data, pi)[0],
                "DR": calc_dr(logged_data, pi, q_hat=q_hat)[0],
                "DOLCE": dolce_contrib,
            }

            for method, contrib in contributions.items():
                estimate = float(np.mean(contrib))
                # influence-function-based SE (same as std(contrib)/sqrt(n), but explicit)
                if method == "DOLCE" and isinstance(dolce_info, dict):
                    fold_means = dolce_info.get("fold_means")
                    if fold_means is not None and len(fold_means) > 1:
                        fold_var = float(np.var(fold_means, ddof=1))
                        se = float(np.sqrt(fold_var / len(fold_means)))
                        t_alpha = float(student_t.ppf(0.975, df=len(fold_means) - 1))
                        ci_low = estimate - t_alpha * se
                        ci_high = estimate + t_alpha * se
                    else:
                        phi = contrib - estimate
                        se = float(np.sqrt(np.mean(phi**2) / contrib.shape[0]))
                        ci_low = estimate - 1.96 * se
                        ci_high = estimate + 1.96 * se
                else:
                    phi = contrib - estimate
                    se = float(np.sqrt(np.mean(phi**2) / contrib.shape[0]))
                    ci_low = estimate - 1.96 * se
                    ci_high = estimate + 1.96 * se
                raw_rows.append(
                    dict(
                        method=method,
                        value=estimate,
                        error=estimate - true_value,
                        se=se,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        covered=ci_low <= true_value <= ci_high,
                        true_value=true_value,
                        support_violation_ratio=support_violation_ratio,
                        num_data=num_data,
                        num_actions=num_actions,
                        lambda_=lambda_,
                        eta=eta,
                        lag_scale=args.lag_scale,
                        env_seed=args.env_seed,
                    )
                )

    raw_df = pd.DataFrame(raw_rows)
    group_cols = [x_col, "method"]
    agg = {
        "mean_value": ("value", "mean"),
        "mse": ("error", lambda x: float(np.mean(x**2))),
        "bias": ("error", "mean"),
        "variance": ("value", lambda x: float(np.var(x, ddof=0))),
        "coverage": ("covered", "mean"),
        "true_value": ("true_value", "first"),
    }
    for col in [
        "support_violation_ratio",
        "num_data",
        "num_actions",
        "lambda_",
        "eta",
        "lag_scale",
        "env_seed",
    ]:
        if col not in group_cols:
            agg[col] = (col, "first")
    summary_df = raw_df.groupby(group_cols).agg(**agg).reset_index()
    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(summary_df, f)

    if args.output_raw:
        raw_output_path = Path(args.output_raw)
        with open(raw_output_path, "wb") as f:
            pickle.dump(pd.DataFrame(raw_rows), f)

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        # plot settings
        preferred_font = "Times New Roman"
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        serif_list = (
            [preferred_font]
            if preferred_font in available_fonts
            else [preferred_font, "Times", "Nimbus Roman", "DejaVu Serif"]
        )
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": serif_list,
                "mathtext.fontset": "stix",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "axes.unicode_minus": False,
                "font.size": 11,
                "axes.labelsize": 11,
                "axes.titlesize": 11,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.linewidth": 0.8,
            }
        )

        order = ["DM", "IPS", "DR", "DOLCE"]  # legend order
        linestyles = {"DM": "-", "IPS": "--", "DR": ":", "DOLCE": "-."}
        markers = {"DM": "o", "IPS": "^", "DR": "s", "DOLCE": "D"}
        colors = {"DM": "blue", "IPS": "red", "DR": "purple", "DOLCE": "green"}

        # aggregate to avoid duplicate values
        plot_df = summary_df.groupby(["method", x_col], as_index=False).agg(
            {
                "mse": "mean",
                "bias": "mean",
                "variance": "mean",
                "coverage": "mean",
            }
        )

        # convert to numeric and sort (support_violation_ratio is object)
        plot_df[x_col] = pd.to_numeric(plot_df[x_col])
        plot_df = plot_df.sort_values(["method", x_col])

        # Figure (1x4): (a)=MSE, (b)=bias, (c)=variance, (d)=coverage
        fig, (ax_mse, ax_bias, ax_var, ax_cov) = plt.subplots(1, 4, figsize=(12.0, 3.0), sharex=True)

        # bias: y=0.0
        ax_bias.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=0)

        # coverage: y=0.95
        ax_cov.axhline(y=0.95, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=0)

        # plot each method
        for m in order:
            sub = plot_df[plot_df["method"] == m].sort_values(x_col)
            x = sub[x_col].to_numpy()

            ax_mse.plot(
                x,
                sub["mse"].to_numpy(),
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=5,
            )
            ax_bias.plot(
                x,
                sub["bias"].to_numpy(),
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=5,
            )
            ax_var.plot(
                x,
                sub["variance"].to_numpy(),
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=5,
            )
            ax_cov.plot(
                x,
                sub["coverage"].to_numpy(),
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=5,
            )

        for ax in (ax_mse, ax_bias, ax_var, ax_cov):
            ax.set_xlabel(x_label)
            ax.tick_params(direction="in", length=4, width=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.02)

        ax_mse.set_ylabel("MSE")
        ax_bias.set_ylabel("bias")
        ax_var.set_ylabel("variance")
        ax_cov.set_ylabel("coverage")

        ax_mse.set_title("(a)")
        ax_bias.set_title("(b)")
        ax_var.set_title("(c)")
        ax_cov.set_title("(d)")
        ax_cov.set_ylim(-0.02, 1.02)

        handles = [
            Line2D(
                [0],
                [0],
                label=m,
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=6,
            )
            for m in order
        ]
        fig.legend(
            handles=handles,
            labels=order,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.4,
            handletextpad=0.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(args.plot_dir + f"/ope_{args.sweep}.pdf", bbox_inches="tight")

    if args.show_summary:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
