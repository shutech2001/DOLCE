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
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic import generate_synthetic_data, calc_true_value  # noqa: E402
from estimating import fit_predict_by_MLP_actionwise  # noqa: E402
from ope import calc_dm, calc_ips, calc_dr, calc_dolce  # noqa: E402
from utils import eps_greedy_policy, parse_comma_separated_list  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="OPE simulation for support violation ratios.")
    parser.add_argument("--num-sim", type=int, default=30)
    parser.add_argument("--num-data", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="data seed base (contexts + reward noise)")
    parser.add_argument("--env-seed", type=int, default=7, help="environment seed (reward function)")
    parser.add_argument(
        "--x-dep",
        type=float,
        default=0.0,
        help="dependence strength of x_t on x_{t-l} (0 = independent)",
    )

    parser.add_argument(
        "--values",
        type=str,
        default="0.0,0.3,0.6,0.9",
        help="comma-separated values",
    )
    parser.add_argument("--num-test-data", type=int, default=200000, help="MC size for true value")
    parser.add_argument("--test-seed", type=int, default=999, help="MC seed for true value")

    parser.add_argument("--output", type=str, default="result_df_support_violation_ope.pkl")
    parser.add_argument("--output-raw", type=str, default="", help="optional path for raw per-simulation results")
    parser.add_argument("--plot-dir", type=str, default="", help="optional directory to save plots")
    parser.add_argument("--show-summary", action="store_true")
    args = parser.parse_args()

    value_list = parse_comma_separated_list(args.values)

    # IMPORTANT: compute true value once for a fixed environment (env_seed), independent of support violation.
    true_value = calc_true_value(
        num_features=args.num_features,
        num_actions=args.num_actions,
        non_overlap_ratio=0.0,
        lambda_=args.lambda_,
        eta=args.eta,
        num_mc=args.num_test_data,
        random_state=args.test_seed,
        env_random_state=args.env_seed,
        x_t_dep=args.x_dep,
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
            )

            pi = eps_greedy_policy(logged_data["q"])

            q_hat = fit_predict_by_MLP_actionwise(
                features=logged_data["x_t"],
                actions=logged_data["a_t"],
                rewards=logged_data["r"],
                num_actions=logged_data["num_actions"],
                random_state=data_seed,
            )

            contributions = {
                "DM": calc_dm(logged_data, pi, q_hat=q_hat)[0],
                "IPS": calc_ips(logged_data, pi)[0],
                "DR": calc_dr(logged_data, pi, q_hat=q_hat)[0],
                "DOLCE": calc_dolce(logged_data, pi, random_state=data_seed)[0],
            }

            for method, contrib in contributions.items():
                estimate = float(np.mean(contrib))
                # influence-function-based SE (same as std(contrib)/sqrt(n), but explicit)
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
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for metric in ["mse", "bias", "variance", "coverage"]:
            fig, ax = plt.subplots()
            for method in summary_df["method"].unique():
                method_df = summary_df[summary_df["method"] == method].sort_values("support_violation_ratio")
                ax.plot(method_df["support_violation_ratio"], method_df[metric], marker="o", label=method)
            ax.set_xlabel("support violation ratio (%)")
            ax.set_ylabel(metric)
            ax.set_title(f"OPE {metric} vs support violation")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"ope_{metric}.png", dpi=150)
            plt.close(fig)

    if args.show_summary:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
