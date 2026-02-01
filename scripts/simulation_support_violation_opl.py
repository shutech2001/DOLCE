from __future__ import annotations

import argparse
import copy
import pickle
import os
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_random_state  # type: ignore
from tqdm import tqdm  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from estimating import (  # noqa: E402
    estimate_ips_gradient,
    estimate_dr_gradient,
    estimate_dolce_gradient,
)
from synthetic import generate_synthetic_data, estimate_true_gradient  # noqa: E402
from opl import RegressionBasedPolicyLearner, GradientBasedPolicyLearner, DOLCE  # noqa: E402
from utils import parse_comma_separated_list, apply_flat_grad  # noqa: E402

warnings.filterwarnings("ignore")


def _init_policy_model(
    num_features: int,
    num_actions: int,
    hidden_layer_size: tuple,
    activation: str,
    seed: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    base = GradientBasedPolicyLearner(
        num_features=num_features,
        num_actions=num_actions,
        hidden_layer_size=hidden_layer_size,
        activation=activation,
        max_iter=0,
        random_state=seed,
    )
    return copy.deepcopy(base.nn_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="OPL simulation for support violation ratios.")
    parser.add_argument("--num-sim", type=int, default=50)
    parser.add_argument("--num-data", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--test-data-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42, help="data seed base (contexts + reward noise)")
    parser.add_argument("--env-seed", type=int, default=7, help="environment seed (reward function)")
    parser.add_argument("--test-seed", type=int, default=999, help="test context seed")
    parser.add_argument("--logging-eps", type=float, default=0.2, help="exploration floor in logging policy")
    parser.add_argument(
        "--x-dep",
        type=float,
        default=1.0,
        help="dependence strength of x_t on x_{t-l} (0 = independent)",
    )
    parser.add_argument(
        "--lag-scale",
        type=float,
        default=2.0,
        help="scale factor for lag reward component (h)",
    )
    parser.add_argument(
        "--values",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="comma-separated values",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/result_df_support_violation_opl.pkl",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default="results/result_df_support_violation_opl_raw.pkl",
        help="optional path for raw per-simulation results",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="results/plots",
        help="optional directory to save plots",
    )
    parser.add_argument("--show-summary", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    _ = check_random_state(args.seed)

    value_list = parse_comma_separated_list(args.values)
    raw_rows = []

    # Use the same test contexts across ratios and sims; only pi_0 changes with ratio.
    for non_overlap_ratio in value_list:
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
            lag_scale=args.lag_scale,
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
                lag_scale=args.lag_scale,
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

            # Evaluate gradient estimators on a fixed target policy (DOLCE's),
            # so differences reflect estimator bias under support violation rather than
            # policy adaptation across methods.
            grad_model = dolce.nn_model
            grad_true = estimate_true_gradient(grad_model, x_test, q_test)

            grad_hat_ips = estimate_ips_gradient(
                grad_model,
                x_logged,
                a_logged,
                r_logged,
                pi_0_logged,
                ips.log_eps,
            )
            grad_err_ips = grad_hat_ips - grad_true
            grad_mse_ips = float(np.mean(grad_err_ips**2))

            grad_hat_dr = estimate_dr_gradient(
                grad_model,
                x_logged,
                a_logged,
                r_logged,
                pi_0_logged,
                q_hat,
                dr.log_eps,
            )
            grad_err_dr = grad_hat_dr - grad_true
            grad_mse_dr = float(np.mean(grad_err_dr**2))

            grad_hat_dolce = estimate_dolce_gradient(dolce, logged_data)
            grad_err_dolce = grad_hat_dolce - grad_true
            grad_mse_dolce = float(np.mean(grad_err_dolce**2))

            grad_true_norm = float(np.linalg.norm(grad_true))
            grad_rel_ips = float(np.linalg.norm(grad_err_ips) / (grad_true_norm + 1e-12))
            grad_rel_dr = float(np.linalg.norm(grad_err_dr) / (grad_true_norm + 1e-12))
            grad_rel_dolce = float(np.linalg.norm(grad_err_dolce) / (grad_true_norm + 1e-12))

            def _cos_sim(u: np.ndarray, v: np.ndarray) -> float:
                denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12
                return float(np.dot(u, v) / denom)

            grad_cos_ips = _cos_sim(grad_hat_ips, grad_true)
            grad_cos_dr = _cos_sim(grad_hat_dr, grad_true)
            grad_cos_dolce = _cos_sim(grad_hat_dolce, grad_true)

            # One-step improvement from a common initialization (diagnostic for gradient quality)
            base_model = _init_policy_model(
                num_features=args.num_features,
                num_actions=args.num_actions,
                hidden_layer_size=dolce.hidden_layer_size,
                activation=dolce.activation,
                seed=data_seed,
            )
            base_model.eval()
            with torch.no_grad():
                base_pi = base_model(torch.from_numpy(x_test).float()).numpy()
            base_value = float((q_test * base_pi).sum(1).mean())

            step_size = dolce.learning_rate_init

            ips_model = copy.deepcopy(base_model)
            apply_flat_grad(ips_model, grad_hat_ips, step_size)
            with torch.no_grad():
                pi_step = ips_model(torch.from_numpy(x_test).float()).numpy()
            value_step_ips = float((q_test * pi_step).sum(1).mean())

            dr_model = copy.deepcopy(base_model)
            apply_flat_grad(dr_model, grad_hat_dr, step_size)
            with torch.no_grad():
                pi_step = dr_model(torch.from_numpy(x_test).float()).numpy()
            value_step_dr = float((q_test * pi_step).sum(1).mean())

            dolce_model = copy.deepcopy(base_model)
            apply_flat_grad(dolce_model, grad_hat_dolce, step_size)
            with torch.no_grad():
                pi_step = dolce_model(torch.from_numpy(x_test).float()).numpy()
            value_step_dolce = float((q_test * pi_step).sum(1).mean())

            one_step_map = {
                "IPS": value_step_ips - base_value,
                "DR": value_step_dr - base_value,
                "DOLCE": value_step_dolce - base_value,
                "DM": np.nan,
            }

            grad_mse_map = {
                "IPS": grad_mse_ips,
                "DR": grad_mse_dr,
                "DOLCE": grad_mse_dolce,
                "DM": np.nan,
            }
            grad_rel_map = {
                "IPS": grad_rel_ips,
                "DR": grad_rel_dr,
                "DOLCE": grad_rel_dolce,
                "DM": np.nan,
            }
            grad_cos_map = {
                "IPS": grad_cos_ips,
                "DR": grad_cos_dr,
                "DOLCE": grad_cos_dolce,
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
                        grad_rel=grad_rel_map[method],
                        grad_cos=grad_cos_map[method],
                        one_step_improve=one_step_map[method],
                        v_star=v_star,
                        v_pi_0=v_pi_0,
                        num_data=args.num_data,
                        num_actions=args.num_actions,
                        lambda_=args.lambda_,
                        eta=args.eta,
                        env_seed=args.env_seed,
                        logging_eps=args.logging_eps,
                        lag_scale=args.lag_scale,
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
        grad_rel_mean=("grad_rel", "mean"),
        grad_cos_mean=("grad_cos", "mean"),
        one_step_improve_mean=("one_step_improve", "mean"),
        v_star=("v_star", "first"),
        v_pi_0=("v_pi_0", "first"),
        num_data=("num_data", "first"),
        num_actions=("num_actions", "first"),
        lambda_=("lambda_", "first"),
        eta=("eta", "first"),
        env_seed=("env_seed", "first"),
        logging_eps=("logging_eps", "first"),
        lag_scale=("lag_scale", "first"),
    ).reset_index()

    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(summary_df, f)

    if args.output_raw:
        raw_output_path = Path(args.output_raw)
        with open(raw_output_path, "wb") as f:
            pickle.dump(raw_df, f)

    if args.plot_dir:
        preferred_font = "Times New Roman"
        available_fonts = {f.name for f in fm.fontManager.ttflist}

        serif_list = (
            [preferred_font] if preferred_font in available_fonts else [preferred_font, "Nimbus Roman", "DejaVu Serif"]
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

        order = ["DM", "IPS", "DR", "DOLCE"]
        display_name = {"DM": "Regression-based", "IPS": "IPS", "DR": "DR", "DOLCE": "DOLCE"}
        linestyles = {"DM": "-", "IPS": "--", "DR": ":", "DOLCE": "-."}
        markers = {"DM": "o", "IPS": "^", "DR": "s", "DOLCE": "D"}
        colors = {"DM": "blue", "IPS": "red", "DR": "purple", "DOLCE": "green"}

        # aggregate to avoid duplicate values
        plot_df = summary_df.groupby(["method", "support_violation_ratio"], as_index=False).agg(
            {
                "ni_median": "mean",
                "one_step_improve_mean": "mean",
                "regret_mean": "mean",
            }
        )

        plot_df["support_violation_ratio"] = pd.to_numeric(plot_df["support_violation_ratio"])
        plot_df = plot_df.sort_values(["method", "support_violation_ratio"])

        # Figure (1x3): (a)=NI, (b)=One-step, (c)=Regret
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharex=True)
        ax_ni, ax_os, ax_reg = axes

        # (a) normalized improvement & (c) regret: plot all methods
        for m in order:
            sub = plot_df[plot_df["method"] == m].sort_values("support_violation_ratio")
            x = sub["support_violation_ratio"].to_numpy()

            ax_ni.plot(
                x,
                sub["ni_median"].to_numpy(),
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=5,
            )
            ax_reg.plot(
                x,
                sub["regret_mean"].to_numpy(),
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=5,
            )

        # (b) one-step improvement: do not plot Regression-based(DM)
        for m in ["IPS", "DR", "DOLCE"]:
            sub = plot_df[plot_df["method"] == m].sort_values("support_violation_ratio")
            x = sub["support_violation_ratio"].to_numpy()

            ax_os.plot(
                x,
                sub["one_step_improve_mean"].to_numpy(),
                linestyle=linestyles[m],
                marker=markers[m],
                color=colors[m],
                linewidth=1.8,
                markersize=5,
            )

        # axis labels and style
        xticks = sorted(plot_df["support_violation_ratio"].unique())
        for ax in axes:
            ax.set_xlabel("support violation ratio")
            ax.set_xticks(xticks)
            ax.tick_params(direction="in", length=4, width=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.02)

        ax_ni.set_ylabel("normalized improvement")
        ax_os.set_ylabel("one-step improvement")
        ax_reg.set_ylabel("regret")

        ax_ni.set_title("(a)")
        ax_os.set_title("(b)")
        ax_reg.set_title("(c)")

        # one-step is 1e-4 order, so use scientific notation for paper
        ax_os.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

        # legend: line style + marker, outside of figure (top), fixed order
        handles = [
            Line2D(
                [0],
                [0],
                label=display_name[m],
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
            labels=[display_name[m] for m in order],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.4,
            handletextpad=0.5,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(args.plot_dir + "/opl_support_violation.pdf", bbox_inches="tight")

    if args.show_summary:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
