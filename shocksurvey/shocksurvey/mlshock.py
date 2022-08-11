from __future__ import annotations

from datetime import datetime as dt
from enum import Enum
from pathlib import Path
from tkinter import W

import numpy as np
import pandas as pd

from shocksurvey import ENV, gen_timestamp, ts_as_dt
from shocksurvey.shock import Shock, ShockProperties

COLUMNS = [
    "time",
    "direction",
    "burst_start",
    "burst_end",
    "Bx_us",
    "By_us",
    "Bz_us",
    "B_us_abs",
    "sB_us_abs",
    "Delta_theta_B",
    "Ni_us",
    "Ti_us",
    "Vx_us",
    "Vy_us",
    "Vz_us",
    "beta_i_us",
    "Pdyn_us",
    "thBn",
    "sthBn",
    "normal_x",
    "normal_y",
    "normal_z",
    "MA",
    "sMA",
    "Mf",
    "sMf",
    "B_jump",
    "Ni_jump",
    "Te_jump",
    "pos_x",
    "pos_y",
    "pos_z",
    "sc_sep_min",
    "sc_sep_max",
    "sc_sep_mean",
    "TQF",
]


def load_data(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        # Load from ENV
        path = Path(ENV.ML_DATA_LOCATION)

    return pd.read_csv(path, comment="#", names=COLUMNS)


def locate_closest_shock(data: pd.DataFrame, timestamp: float) -> pd.Series:
    closest_index = np.argmin(np.abs(data.time - timestamp))
    return data.iloc[closest_index]


class MLProperties(ShockProperties):
    def __init__(self, timestamp: float, threshold: float = 3600) -> None:
        """Loads the properties for a shock."""
        self.timestamp = timestamp
        self.all_props: pd.DataFrame = load_data()
        self.props: pd.Series = locate_closest_shock(self.all_props, self.timestamp)

        if self.props.time - self.timestamp > threshold:
            raise ValueError(
                f"No shock with timestamp {gen_timestamp(dt.utcfromtimestamp(timestamp), 'pretty')} found in database. Closest: {gen_timestamp(dt.utcfromtimestamp(self.props.time), 'pretty')}"
            )

    def get_trange(self) -> list[str]:
        def f(ts):
            return gen_timestamp(ts_as_dt(ts), "pyspedas")

        return list(map(f, [self.props.burst_start, self.props.burst_end]))
