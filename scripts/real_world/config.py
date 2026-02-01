from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

DEFAULT_DATA_DIR = Path("/path/to/data/nwicu")


@dataclass(frozen=True)
class RealWorldConfig:
    """Configuration for real-world data processing."""

    data_dir: Path = DEFAULT_DATA_DIR
    chunk_size: int = 2000000
    min_history_hours: int = 2
    horizon_hours: int = 1
    tolerance_hours: int = 2
    lags: Tuple[int, ...] = (1, 2, 4)
    action_window_hours: int = 1
    map_threshold: float = 65.0
    cap_rows_per_stay: int = 3
    cap_mode: str = "block"
    sample_one_per_stay: bool = True
    test_size: float = 0.3
    seed: int = 0
    vasopressor_regex: str = r"(norepi|nor-?epinephrine|epinephrine|vasopressin|phenylephrine|dopamine)"
    verbose: bool = True
