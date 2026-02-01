from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class CharteventsColumns:
    stay_id: str
    charttime: str
    itemid: str
    valuenum: str
    valueuom: str | None


@dataclass(frozen=True)
class EmarColumns:
    hadm_id: str
    charttime: str
    med: str
    emar_id: str | None


@dataclass(frozen=True)
class IcustaysColumns:
    stay_id: str
    hadm_id: str
    intime: str
    outtime: str


def _resolve_column(cols: Sequence[str], candidates: Iterable[str]) -> str | None:
    """Resolve a column name from a list of candidates.

    Args:
        cols (Sequence[str]): The list of column names.
        candidates (Iterable[str]): The list of candidate column names.

    Returns:
        str | None: The resolved column name.
    """
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        if cand in cols_l:
            return cols[cols_l.index(cand)]
    return None


def load_d_items(data_dir: Path) -> pd.DataFrame:
    """Load the d_items file.

    Args:
        data_dir (Path): The data directory.

    Returns:
        pd.DataFrame: The d_items dataframe.
    """
    d_items = pd.read_csv(data_dir / "nw_icu/d_items.csv")
    d_items.columns = d_items.columns.str.lower()
    d_items["label_l"] = d_items["label"].astype(str).str.lower()
    d_items["linksto_l"] = d_items["linksto"].astype(str).str.lower()
    return d_items


def filter_chartevents_dictionary(d_items: pd.DataFrame) -> pd.DataFrame:
    """Filter the d_items dataframe to only include chartevents.

    Args:
        d_items (pd.DataFrame): The d_items dataframe.

    Returns:
        pd.DataFrame: The filtered d_items dataframe.
    """
    return d_items[d_items["linksto_l"].eq("chartevents")].copy()


def itemids_exact(d_ce: pd.DataFrame, label_exact: str) -> tuple[pd.DataFrame, list[int]]:
    """Find the itemids for a given label.

    Args:
        d_ce (pd.DataFrame): The d_items dataframe.
        label_exact (str): The exact label to search for.

    Returns:
        tuple[pd.DataFrame, list[int]]: The filtered d_items dataframe and the list of itemids.
    """
    mask = d_ce["label_l"].eq(label_exact.lower())
    out = d_ce.loc[mask, ["itemid", "label", "linksto"]].drop_duplicates()
    itemids = out["itemid"].dropna().astype(int).unique().tolist()
    return out, itemids


def resolve_chartevents_columns(ce_path: Path) -> CharteventsColumns:
    """Resolve the chartevents columns.

    Args:
        ce_path (Path): The path to the chartevents file.

    Returns:
        CharteventsColumns: The resolved chartevents columns.
    """
    cols = pd.read_csv(ce_path, nrows=0).columns
    stay = _resolve_column(cols, ["stay_id", "icustay_id", "stayid"])
    charttime = _resolve_column(cols, ["charttime", "chart_time", "time", "event_time"])
    itemid = _resolve_column(cols, ["itemid", "item_id"])
    valuenum = _resolve_column(cols, ["valuenum", "value", "value_num", "value_numeric"])
    valueuom = _resolve_column(cols, ["valueuom", "value_uom", "unit", "units"])

    if not stay or not charttime or not itemid or not valuenum:
        raise ValueError("chartevents must include stay_id, charttime, itemid, valuenum")

    return CharteventsColumns(stay_id=stay, charttime=charttime, itemid=itemid, valuenum=valuenum, valueuom=valueuom)


def load_chartevents(
    ce_path: Path,
    itemids: set[int],
    columns: CharteventsColumns,
    chunk_size: int = 2_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the chartevents file.

    Args:
        ce_path (Path): The path to the chartevents file.
        itemids (set[int]): The set of itemids to load.
        columns (CharteventsColumns): The resolved chartevents columns.
        chunk_size (int, optional): The chunk size. Defaults to 2000000.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        pd.DataFrame: The chartevents dataframe.
    """
    usecols = [columns.stay_id, columns.charttime, columns.itemid, columns.valuenum]
    if columns.valueuom is not None:
        usecols.append(columns.valueuom)

    events_list: list[pd.DataFrame] = []
    for chunk in pd.read_csv(ce_path, usecols=usecols, chunksize=chunk_size):
        chunk.columns = chunk.columns.str.lower()
        rename_map = {
            columns.stay_id.lower(): "stay_id",
            columns.charttime.lower(): "charttime",
            columns.itemid.lower(): "itemid",
            columns.valuenum.lower(): "valuenum",
        }
        if columns.valueuom is not None:
            rename_map[columns.valueuom.lower()] = "valueuom"
        chunk = chunk.rename(columns=rename_map)

        chunk = chunk[chunk["itemid"].isin(itemids)].copy()
        if len(chunk) == 0:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["stay_id", "charttime", "valuenum"])

        events_list.append(chunk)

    if len(events_list) == 0:
        if verbose:
            print("chartevents: no matching rows found")
        cols = ["stay_id", "charttime", "itemid", "valuenum", "valueuom"]
        return pd.DataFrame(columns=cols)

    ce = pd.concat(events_list, ignore_index=True)
    if verbose:
        print("chartevents shape:", ce.shape)
    return ce


def resolve_emar_columns(emar_path: Path) -> EmarColumns:
    """Resolve the emar columns.

    Args:
        emar_path (Path): The path to the emar file.

    Returns:
        EmarColumns: The resolved emar columns.
    """
    cols = pd.read_csv(emar_path, nrows=0).columns
    hadm = _resolve_column(cols, ["hadm_id", "hadmid"])
    charttime = _resolve_column(cols, ["charttime", "scheduletime", "storetime", "time", "event_time"])
    med = _resolve_column(cols, ["medication", "med_name", "drug", "drug_name", "name"])
    emar_id = _resolve_column(cols, ["emar_id"])

    if not hadm or not charttime or not med:
        raise ValueError("emar must include hadm_id, time, and medication columns")

    return EmarColumns(hadm_id=hadm, charttime=charttime, med=med, emar_id=emar_id)


def load_emar(emar_path: Path) -> pd.DataFrame:
    """Load the emar file.

    Args:
        emar_path (Path): The path to the emar file.

    Returns:
        pd.DataFrame: The emar dataframe.
    """
    columns = resolve_emar_columns(emar_path)
    usecols = [columns.hadm_id, columns.charttime, columns.med]
    if columns.emar_id is not None:
        usecols.append(columns.emar_id)

    emar = pd.read_csv(emar_path, usecols=usecols)
    emar.columns = emar.columns.str.lower()

    rename_map = {
        columns.hadm_id.lower(): "hadm_id",
        columns.charttime.lower(): "charttime",
        columns.med.lower(): "med",
    }
    if columns.emar_id is not None:
        rename_map[columns.emar_id.lower()] = "emar_id"
    emar = emar.rename(columns=rename_map)

    if "emar_id" not in emar.columns:
        emar = emar.reset_index().rename(columns={"index": "emar_id"})

    emar["charttime"] = pd.to_datetime(emar["charttime"], errors="coerce")
    emar = emar.dropna(subset=["hadm_id", "charttime", "med"]).copy()
    emar["med_l"] = emar["med"].astype(str).str.lower()

    return emar


def resolve_icustays_columns(icu_path: Path) -> IcustaysColumns:
    """Resolve the icustays columns.

    Args:
        icu_path (Path): The path to the icustays file.

    Returns:
        IcustaysColumns: The resolved icustays columns.
    """
    cols = pd.read_csv(icu_path, nrows=0).columns
    stay = _resolve_column(cols, ["stay_id", "icustay_id", "stayid"])
    hadm = _resolve_column(cols, ["hadm_id", "hadmid"])
    intime = _resolve_column(cols, ["intime", "icu_intime", "in_time"])
    outtime = _resolve_column(cols, ["outtime", "icu_outtime", "out_time"])

    if not stay or not hadm or not intime or not outtime:
        raise ValueError("icustays must include stay_id, hadm_id, intime, outtime")

    return IcustaysColumns(stay_id=stay, hadm_id=hadm, intime=intime, outtime=outtime)


def load_icustays(icu_path: Path) -> pd.DataFrame:
    """Load the icustays file.

    Args:
        icu_path (Path): The path to the icustays file.

    Returns:
        pd.DataFrame: The icustays dataframe.
    """
    columns = resolve_icustays_columns(icu_path)
    icu = pd.read_csv(icu_path, usecols=[columns.stay_id, columns.hadm_id, columns.intime, columns.outtime])
    icu.columns = icu.columns.str.lower()
    icu = icu.rename(
        columns={
            columns.stay_id.lower(): "stay_id",
            columns.hadm_id.lower(): "hadm_id",
            columns.intime.lower(): "intime",
            columns.outtime.lower(): "outtime",
        }
    )
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu = icu.dropna(subset=["stay_id", "hadm_id", "intime", "outtime"]).copy()
    return icu
