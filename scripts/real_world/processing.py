from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def filter_value_ranges(
    ce: pd.DataFrame,
    hr_ids: Iterable[int],
    spo2_ids: Iterable[int],
    sbp_ids: Iterable[int],
    dbp_ids: Iterable[int],
) -> pd.DataFrame:
    """Filter the chartevents dataframe to only include events within the value ranges.

    Args:
        ce (pd.DataFrame): The chartevents dataframe.
        hr_ids (Iterable[int]): The HR itemids.
        spo2_ids (Iterable[int]): The SpO2 itemids.
        sbp_ids (Iterable[int]): The SBP itemids.
        dbp_ids (Iterable[int]): The DBP itemids.

    Returns:
        pd.DataFrame: The filtered chartevents dataframe.
    """
    if ce.empty:
        return ce.copy()

    itemid = ce["itemid"].astype(int)
    val = ce["valuenum"].astype(float)
    mask = pd.Series(True, index=ce.index)

    hr_ids = set(hr_ids)
    spo2_ids = set(spo2_ids)
    sbp_ids = set(sbp_ids)
    dbp_ids = set(dbp_ids)

    if hr_ids:
        mask &= ~(itemid.isin(hr_ids) & ((val < 20) | (val > 250)))
    if spo2_ids:
        mask &= ~(itemid.isin(spo2_ids) & ((val < 50) | (val > 100)))
    if sbp_ids:
        mask &= ~(itemid.isin(sbp_ids) & ((val < 40) | (val > 300)))
    if dbp_ids:
        mask &= ~(itemid.isin(dbp_ids) & ((val < 20) | (val > 200)))

    return ce.loc[mask].copy()


def merge_events_with_icu(ce: pd.DataFrame, icu: pd.DataFrame) -> pd.DataFrame:
    """Merge the chartevents dataframe with the icustays dataframe.

    Args:
        ce (pd.DataFrame): The chartevents dataframe.
        icu (pd.DataFrame): The icustays dataframe.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    if ce.empty:
        return ce.copy()

    merged = ce.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    merged = merged[(merged["charttime"] >= merged["intime"]) & (merged["charttime"] <= merged["outtime"])].copy()
    return merged


def make_time_grid(
    icu: pd.DataFrame,
    min_history_hours: int,
    horizon_hours: int,
    freq: str = "1h",
) -> pd.DataFrame:
    """Make the time grid.

    Args:
        icu (pd.DataFrame): The icustays dataframe.
        min_history_hours (int): The minimum history hours.
        horizon_hours (int): The horizon hours.
        freq (str, optional): The frequency. Defaults to "1h".

    Returns:
        pd.DataFrame: The time grid.
    """
    grid_list: list[pd.DataFrame] = []
    min_history = pd.Timedelta(hours=min_history_hours)
    horizon = pd.Timedelta(hours=horizon_hours)

    for stay_id, g in icu.groupby("stay_id"):
        t0 = g["intime"].min()
        t1 = g["outtime"].max()
        if pd.isna(t0) or pd.isna(t1):
            continue
        start = t0 + min_history
        end = t1 - horizon
        if end <= start:
            continue
        times = pd.date_range(start=start.floor("h"), end=end.floor("h"), freq=freq)
        if len(times) == 0:
            continue
        grid_list.append(pd.DataFrame({"stay_id": stay_id, "t": times}))

    if not grid_list:
        return pd.DataFrame(columns=["stay_id", "t"])

    return pd.concat(grid_list, ignore_index=True)


def last_before(
    grid_df: pd.DataFrame,
    ev_df: pd.DataFrame,
    colname: str,
    tolerance_hours: int = 2,
) -> pd.DataFrame:
    """Last before.

    Args:
        grid_df (pd.DataFrame): The grid dataframe.
        ev_df (pd.DataFrame): The event dataframe.
        colname (str): The column name.
        tolerance_hours (int, optional): The tolerance hours. Defaults to 2.

    Returns:
        pd.DataFrame: The last before dataframe.
    """
    if grid_df.empty:
        return grid_df.copy()
    if ev_df.empty:
        out = grid_df.copy()
        out[colname] = np.nan
        return out

    left = grid_df.copy()
    right = ev_df.copy()

    left["t"] = pd.to_datetime(left["t"], errors="coerce")
    right["charttime"] = pd.to_datetime(right["charttime"], errors="coerce")

    left = left.dropna(subset=["stay_id", "t"])
    right = right.dropna(subset=["stay_id", "charttime", "valuenum"])

    left = left.sort_values(["t", "stay_id"]).reset_index(drop=True)
    right = right.rename(columns={"charttime": "t", "valuenum": colname})
    right = right.sort_values(["t", "stay_id"]).reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right[["stay_id", "t", colname]],
        on="t",
        by="stay_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_hours),
        allow_exact_matches=True,
    )
    return out


def build_features(
    ce: pd.DataFrame,
    grid: pd.DataFrame,
    hr_ids: Iterable[int],
    spo2_ids: Iterable[int],
    sbp_ids: Iterable[int],
    dbp_ids: Iterable[int],
    lags: Sequence[int],
    tolerance_hours: int = 2,
) -> pd.DataFrame:
    """Build the features.

    Args:
        ce (pd.DataFrame): The chartevents dataframe.
        grid (pd.DataFrame): The time grid.
        hr_ids (Iterable[int]): The HR itemids.
        spo2_ids (Iterable[int]): The SpO2 itemids.
        sbp_ids (Iterable[int]): The SBP itemids.
        dbp_ids (Iterable[int]): The DBP itemids.
        lags (Sequence[int]): The lags.
        tolerance_hours (int, optional): The tolerance hours. Defaults to 2.

    Returns:
        pd.DataFrame: The features dataframe.
    """
    if grid.empty:
        return pd.DataFrame()

    def _select_events(itemids: Iterable[int]) -> pd.DataFrame:
        return ce[ce["itemid"].isin(itemids)][["stay_id", "charttime", "valuenum"]].sort_values(
            ["stay_id", "charttime"]
        )

    ce_hr = _select_events(hr_ids)
    ce_spo2 = _select_events(spo2_ids)
    ce_sbp = _select_events(sbp_ids)
    ce_dbp = _select_events(dbp_ids)

    X = grid.copy()
    X = last_before(X, ce_hr, "hr", tolerance_hours=tolerance_hours)
    X = last_before(X, ce_spo2, "spo2", tolerance_hours=tolerance_hours)
    X = last_before(X, ce_sbp, "sbp", tolerance_hours=tolerance_hours)
    X = last_before(X, ce_dbp, "dbp", tolerance_hours=tolerance_hours)

    X["map"] = (X["sbp"] + 2 * X["dbp"]) / 3.0

    X = X.sort_values(["stay_id", "t"]).copy()
    feat_cols = ["hr", "spo2", "sbp", "dbp", "map"]

    for lag in lags:
        for c in feat_cols:
            X[f"{c}_lag{lag}h"] = X.groupby("stay_id")[c].shift(lag)

    need_cols = feat_cols + [f"{c}_lag{lag}h" for lag in lags for c in feat_cols]
    X = X.dropna(subset=need_cols).copy()
    return X


def link_emar_to_icu(emar: pd.DataFrame, icu: pd.DataFrame) -> pd.DataFrame:
    """Link the emar dataframe to the icustays dataframe.

    Args:
        emar (pd.DataFrame): The emar dataframe.
        icu (pd.DataFrame): The icustays dataframe.

    Returns:
        pd.DataFrame: The linked dataframe.
    """
    emar_icu = emar.merge(icu[["stay_id", "hadm_id", "intime", "outtime"]], on="hadm_id", how="inner")
    emar_icu = emar_icu[
        (emar_icu["charttime"] >= emar_icu["intime"]) & (emar_icu["charttime"] <= emar_icu["outtime"])
    ].copy()
    emar_icu = emar_icu.sort_values(["emar_id", "intime"]).groupby("emar_id", as_index=False).tail(1)
    return emar_icu


def extract_vasopressor_events(emar_icu: pd.DataFrame, regex: str) -> pd.DataFrame:
    """Extract the vasopressor events.

    Args:
        emar_icu (pd.DataFrame): The emar icustays dataframe.
        regex (str): The regex.

    Returns:
        pd.DataFrame: The vasopressor events dataframe.
    """
    if emar_icu.empty:
        return pd.DataFrame(columns=["stay_id", "charttime"])

    emar_vaso = emar_icu[emar_icu["med_l"].str.contains(regex, na=False, regex=True)].copy()
    return emar_vaso[["stay_id", "charttime"]].dropna().copy()


def build_action_grid(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    window_hours: int = 1,
) -> pd.DataFrame:
    """Build the action grid.

    Args:
        grid (pd.DataFrame): The time grid.
        events (pd.DataFrame): The events dataframe.
        window_hours (int, optional): The window hours. Defaults to 1.

    Returns:
        pd.DataFrame: The action grid.
    """
    grid2 = grid[["stay_id", "t"]].copy()
    grid2["t"] = pd.to_datetime(grid2["t"], errors="coerce")
    grid2 = grid2.dropna(subset=["stay_id", "t"]).copy()

    if grid2.empty:
        out = grid2.copy()
        out["a"] = 0
        return out

    if events.empty:
        out = grid2.copy()
        out["a"] = 0
        return out

    grid2 = grid2.sort_values(["t", "stay_id"]).reset_index(drop=True)
    ev = events.sort_values(["charttime", "stay_id"]).reset_index(drop=True)

    assigned = pd.merge_asof(
        ev,
        grid2,
        left_on="charttime",
        right_on="t",
        by="stay_id",
        direction="backward",
        allow_exact_matches=True,
    )
    assigned = assigned.dropna(subset=["t"])
    assigned = assigned[assigned["charttime"] < assigned["t"] + pd.Timedelta(hours=window_hours)].copy()

    A_events = assigned.drop_duplicates(["stay_id", "t"]).assign(a=1)[["stay_id", "t", "a"]]

    A = grid2.merge(A_events, on=["stay_id", "t"], how="left")
    A["a"] = A["a"].fillna(0).astype(int)
    return A


def build_bandit_dataset(features: pd.DataFrame, actions: pd.DataFrame, map_threshold: float = 65.0) -> pd.DataFrame:
    """Build the bandit dataset.

    Args:
        features (pd.DataFrame): The features dataframe.
        actions (pd.DataFrame): The actions dataframe.
        map_threshold (float, optional): The map threshold. Defaults to 65.0.

    Returns:
        pd.DataFrame: The bandit dataset.
    """
    if features.empty or actions.empty:
        return pd.DataFrame()

    X2 = features.sort_values(["stay_id", "t"]).copy()
    X2["map_next"] = X2.groupby("stay_id")["map"].shift(-1)
    X2["r"] = (X2["map_next"] >= map_threshold).astype(float)
    X2 = X2.dropna(subset=["map_next"]).copy()

    bandit = X2.merge(actions, on=["stay_id", "t"], how="inner")
    return bandit


def cap_per_stay(df: pd.DataFrame, max_rows: int, mode: str = "block", seed: int = 0) -> pd.DataFrame:
    """Cap the dataframe per stay.

    Args:
        df (pd.DataFrame): The dataframe.
        max_rows (int): The maximum rows.
        mode (str, optional): The mode. Defaults to "block".
        seed (int, optional): The seed. Defaults to 0.

    Returns:
        pd.DataFrame: The capped dataframe.
    """
    df = df.sort_values(["stay_id", "t"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if max_rows <= 0:
        return df

    if mode == "head":
        out = df.groupby("stay_id", group_keys=False).head(max_rows).copy()
        return out.reset_index(drop=True)

    if mode == "sample":
        out = df.groupby("stay_id", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_rows), random_state=seed)
        )
        return out.reset_index(drop=True)

    if mode == "block":

        def _take_block(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) <= max_rows:
                return g
            start = int(rng.integers(0, len(g) - max_rows + 1))
            return g.iloc[start : start + max_rows]  # noqa: E203

        out = df.groupby("stay_id", group_keys=False).apply(_take_block)
        return out.reset_index(drop=True)

    raise ValueError("mode must be one of {'head','block','sample'}")


def sample_one_per_stay(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Sample one per stay.

    Args:
        df (pd.DataFrame): The dataframe.
        seed (int, optional): The seed. Defaults to 0.

    Returns:
        pd.DataFrame: The sampled dataframe.
    """
    if df.empty:
        return df

    out = (
        df.sort_values(["stay_id", "t"])
        .groupby("stay_id", group_keys=False)
        .apply(lambda g: g.sample(n=1, random_state=seed))
        .reset_index(drop=True)
    )
    return out
