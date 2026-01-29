from __future__ import annotations

import argparse
import pickle
import os
import sys
import warnings
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="OPE simulation for support violation ratios.")
    parser.add_argument("--num-sim", type=int, default=100)
    parser.add_argument("--num-data", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--target-eps",
        dest="target_eps",
        type=float,
        default=0.1,
        help="epsilon for the epsilon-greedy target policy used in OPE",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--values",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="comma-separated values",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/result_df_support_violation_ope.pkl",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default="results/result_df_support_violation_ope_raw.pkl",
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

    value_list = parse_comma_separated_list(args.values)

    # IMPORTANT: compute true value once for a fixed environment (env_seed), independent of support violation.
    true_value = calc_true_value(
        num_features=args.num_features,
        num_actions=args.num_actions,
        lambda_=args.lambda_,
        eta=args.eta,
        num_mc=args.num_test_data,
        random_state=args.test_seed,
        env_random_state=args.env_seed,
        x_t_dep=args.x_dep,
        lag_scale=args.lag_scale,
    )

    summary_list = []
    raw_rows = []
    for value in value_list:
        for sim in tqdm(range(args.num_sim), desc=f"support_violation={int(value*100)}"):
            data_seed = args.seed + sim * 100

            logged_data = generate_synthetic_data(
                num_data=args.num_data,
                num_features=args.num_features,
                num_actions=args.num_actions,
                non_overlap_ratio=value,
                lambda_=args.lambda_,
                eta=args.eta,
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
                        support_violation_ratio=int(value * 100),
                        method=method,
                        value=estimate,
                        error=estimate - true_value,
                        se=se,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        covered=ci_low <= true_value <= ci_high,
                        true_value=true_value,
                        num_data=args.num_data,
                        num_actions=args.num_actions,
                        lambda_=args.lambda_,
                        eta=args.eta,
                        lag_scale=args.lag_scale,
                        env_seed=args.env_seed,
                    )
                )

        ratio_df = pd.DataFrame([r for r in raw_rows if r["support_violation_ratio"] == int(value * 100)])
        grouped = ratio_df.groupby("method")

        summary = grouped["value"].agg(["mean"]).rename(columns={"mean": "mean_value"})
        summary["mse"] = grouped["error"].apply(lambda x: float(np.mean(x**2)))
        summary["bias"] = grouped["error"].mean()
        summary["variance"] = grouped["value"].apply(lambda x: float(np.var(x, ddof=0)))
        summary["coverage"] = grouped["covered"].mean()
        summary = summary.reset_index()

        summary["support_violation_ratio"] = int(value * 100)
        summary["true_value"] = true_value
        summary["num_data"] = args.num_data
        summary["num_actions"] = args.num_actions
        summary["lambda_"] = args.lambda_
        summary["eta"] = args.eta
        summary["lag_scale"] = args.lag_scale
        summary["env_seed"] = args.env_seed
        summary_list.append(summary)

    summary_df = pd.concat(summary_list, ignore_index=True)
    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(summary_df, f)

    if args.output_raw:
        raw_output_path = Path(args.output_raw)
        with open(raw_output_path, "wb") as f:
            pickle.dump(pd.DataFrame(raw_rows), f)

    if args.plot_dir:
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
        plot_df = summary_df.groupby(["method", "support_violation_ratio"], as_index=False).agg(
            {
                "mse": "mean",
                "bias": "mean",
                "variance": "mean",
                "coverage": "mean",
            }
        )

        # convert to numeric and sort (support_violation_ratio is object)
        plot_df["support_violation_ratio"] = pd.to_numeric(plot_df["support_violation_ratio"])
        plot_df = plot_df.sort_values(["method", "support_violation_ratio"])

        # Figure (1x4): (a)=MSE, (b)=bias, (c)=variance, (d)=coverage
        fig, (ax_mse, ax_bias, ax_var, ax_cov) = plt.subplots(1, 4, figsize=(12.0, 3.0), sharex=True)

        # bias: y=0.0
        ax_bias.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=0)

        # coverage: y=0.95
        ax_cov.axhline(y=0.95, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=0)

        # plot each method
        for m in order:
            sub = plot_df[plot_df["method"] == m].sort_values("support_violation_ratio")
            x = sub["support_violation_ratio"].to_numpy()

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
            ax.set_xlabel("support violation ratio")
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
        fig.savefig(args.plot_dir + "/ope_support_violation.pdf", bbox_inches="tight")

    if args.show_summary:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
