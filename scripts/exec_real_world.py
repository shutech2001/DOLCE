from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Allow importing package modules when running as a script.
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from scripts.real_world.config import DEFAULT_DATA_DIR, RealWorldConfig  # noqa: E402
from scripts.real_world.io import (  # noqa: E402
    filter_chartevents_dictionary,
    itemids_exact,
    load_chartevents,
    load_d_items,
    load_emar,
    load_icustays,
    resolve_chartevents_columns,
)
from scripts.real_world.ope_eval import make_target_pi_from_rule, prepare_logged_data, run_ope_estimators  # noqa: E402
from scripts.real_world.processing import (  # noqa: E402
    build_action_grid,
    build_bandit_dataset,
    build_features,
    cap_per_stay,
    extract_vasopressor_events,
    filter_value_ranges,
    link_emar_to_icu,
    make_time_grid,
    merge_events_with_icu,
    sample_one_per_stay,
)


def _parse_lags(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return ()
    """Parse the lags from a string.

    Args:
        text (str): The string to parse.

    Returns:
        Tuple[int, ...]: The parsed lags.
    """
    return tuple(int(p) for p in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Real-world OPE pipeline for NW ICU data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="The data directory.")
    parser.add_argument("--chunk-size", type=int, default=2000000, help="The chunk size.")
    parser.add_argument("--min-history-hours", type=int, default=2, help="The minimum history hours.")
    parser.add_argument("--horizon-hours", type=int, default=1, help="The horizon hours.")
    parser.add_argument("--tolerance-hours", type=int, default=2, help="The tolerance hours.")
    parser.add_argument("--lags", type=str, default="1,2,4", help="The lags.")
    parser.add_argument("--action-window-hours", type=int, default=1, help="The action window hours.")
    parser.add_argument("--map-threshold", type=float, default=65.0, help="The map threshold.")
    parser.add_argument("--cap-rows-per-stay", type=int, default=3, help="The cap rows per stay.")
    parser.add_argument(
        "--cap-mode", type=str, default="block", choices=["head", "block", "sample"], help="The cap mode."
    )
    parser.add_argument("--no-sample-one-per-stay", action="store_true", help="Whether to sample one per stay.")
    parser.add_argument("--test-size", type=float, default=0.3, help="The test size.")
    parser.add_argument("--seed", type=int, default=0, help="The seed.")
    parser.add_argument(
        "--vasopressor-regex",
        type=str,
        default=r"(norepi|nor-?epinephrine|epinephrine|vasopressin|phenylephrine|dopamine)",
        help="The vasopressor regex.",
    )
    parser.add_argument("--quiet", action="store_true", help="Whether to be quiet.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = RealWorldConfig(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        min_history_hours=args.min_history_hours,
        horizon_hours=args.horizon_hours,
        tolerance_hours=args.tolerance_hours,
        lags=_parse_lags(args.lags),
        action_window_hours=args.action_window_hours,
        map_threshold=args.map_threshold,
        cap_rows_per_stay=args.cap_rows_per_stay,
        cap_mode=args.cap_mode,
        sample_one_per_stay=not args.no_sample_one_per_stay,
        test_size=args.test_size,
        seed=args.seed,
        vasopressor_regex=args.vasopressor_regex,
        verbose=not args.quiet,
    )

    if config.verbose:
        print("data_dir:", config.data_dir)

    d_items = load_d_items(config.data_dir)
    d_ce = filter_chartevents_dictionary(d_items)

    hr_candidates, hr_itemids = itemids_exact(d_ce, "PULSE")
    spo2_candidates, spo2_itemids = itemids_exact(d_ce, "PULSE OXIMETRY")
    sbp_candidates, sbp_itemids = itemids_exact(d_ce, "BP SYSTOLIC")
    dbp_candidates, dbp_itemids = itemids_exact(d_ce, "BP DIASTOLIC")

    if config.verbose:
        print("HR:", hr_candidates)
        print("SpO2:", spo2_candidates)
        print("SBP:", sbp_candidates)
        print("DBP:", dbp_candidates)

    if not (hr_itemids and spo2_itemids and sbp_itemids and dbp_itemids):
        raise ValueError("Missing itemids for vital signs")

    ce_path = config.data_dir / "nw_icu/chartevents.csv"
    ce_cols = resolve_chartevents_columns(ce_path)
    if config.verbose:
        print("Resolved chartevents:", ce_cols)

    need_itemids = set(hr_itemids + spo2_itemids + sbp_itemids + dbp_itemids)
    ce = load_chartevents(
        ce_path,
        itemids=need_itemids,
        columns=ce_cols,
        chunk_size=config.chunk_size,
        verbose=config.verbose,
    )
    ce = filter_value_ranges(ce, hr_itemids, spo2_itemids, sbp_itemids, dbp_itemids)

    icu_path = config.data_dir / "nw_icu/icustays.csv"
    icu = load_icustays(icu_path)

    ce = merge_events_with_icu(ce, icu)

    grid = make_time_grid(icu, config.min_history_hours, config.horizon_hours)
    if config.verbose:
        print("grid:", grid.shape, grid.head())

    X = build_features(
        ce,
        grid,
        hr_itemids,
        spo2_itemids,
        sbp_itemids,
        dbp_itemids,
        config.lags,
        tolerance_hours=config.tolerance_hours,
    )
    if config.verbose:
        print("features:", X.shape)

    emar_path = config.data_dir / "nw_hosp/emar.csv"
    emar = load_emar(emar_path)
    emar_icu = link_emar_to_icu(emar, icu)
    ev = extract_vasopressor_events(emar_icu, config.vasopressor_regex)
    A = build_action_grid(grid, ev, window_hours=config.action_window_hours)
    if config.verbose:
        print("action counts:\n", A["a"].value_counts(dropna=False).head())

    bandit = build_bandit_dataset(X, A, map_threshold=config.map_threshold)
    if bandit.empty:
        raise ValueError("Bandit dataset is empty after preprocessing")

    if config.cap_rows_per_stay > 0:
        bandit = cap_per_stay(bandit, max_rows=config.cap_rows_per_stay, mode=config.cap_mode, seed=config.seed)
        if config.verbose:
            print("after cap rows:", len(bandit))
            print("after cap unique stays:", bandit["stay_id"].nunique())
            print("after cap action rate:", bandit["a"].mean(), "(#a=1:", bandit["a"].sum(), ")")

    if config.sample_one_per_stay:
        bandit = sample_one_per_stay(bandit, seed=config.seed)
        if config.verbose:
            print("after 1-per-stay:", len(bandit), bandit["stay_id"].nunique(), bandit["a"].mean())

    bundle = prepare_logged_data(bandit, config.lags, test_size=config.test_size, seed=config.seed)
    pi_target_eval = make_target_pi_from_rule(bundle.df_eval)

    res_ope = run_ope_estimators(bundle.logged_eval, pi_target_eval, bundle.a_eval, random_state=config.seed)
    print(res_ope)


if __name__ == "__main__":
    main()
