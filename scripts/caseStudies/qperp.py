from __future__ import annotations

from datetime import datetime as dt
from typing import Callable, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm
from numpy import typing as npt
from phdhelper.helpers import override_mpl
from shocksurvey import gen_timestamp, logger
from shocksurvey.constants import SC
from shocksurvey.mlshock import MLProperties
from shocksurvey.shock import Shock, Spacecraft
from shocksurvey.spedas import FGM, FPI, FSM


class Bins(NamedTuple):
    start: np.ndarray
    end: np.ndarray


Splitter = Callable[[int, float], Bins]


def split_by_fraction(length: int, fraction: float) -> Bins:
    if not 0 < fraction <= 1:
        raise ValueError(
            f"fraction should be in the range [0,1). Recieved {fraction:0.2f}"
        )
    bin_start = np.linspace(0, length, int(1 / fraction) + 1, dtype=int)[:-1]
    bin_end = np.linspace(0, length, int(1 / fraction) + 1, dtype=int)[1:]
    return Bins(bin_start, bin_end)


def gen_windowed_data(
    data: pd.DataFrame,
    split_fn: Splitter,
    *args,
) -> tuple[Bins, list[pd.DataFrame]]:
    bins = split_fn(len(data.index), *args)
    return bins, window_data(bins, data)


def window_data(bins: Bins, data: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    for i in range(bins.start.size):
        out.append(data.iloc[bins.start[i] : bins.end[i]])
    return out


# @logger.catch
def main():
    override_mpl.override()

    timestamp = 1520916120.0
    trange = ["2018-03-13/04:42:00", "2018-03-13/04:57:00"]

    shock = Shock(trange=trange)
    properties = MLProperties(timestamp=timestamp)
    shock.properties = properties

    mms1 = shock.add_spacecraft(SC.MMS1)
    mms1.add_fgm(trange=properties.get_trange())
    assert mms1.fgm is not None
    assert mms1.fgm.data is not None

    fpi = FPI(timestamp=timestamp, sc=SC.MMS1)
    fpi.load()

    mms2 = shock.add_spacecraft(SC.MMS2)
    mms2.add_fgm(properties.get_trange())
    assert mms2.fgm is not None
    assert mms2.fgm.data is not None


if __name__ == "__main__":
    main()
