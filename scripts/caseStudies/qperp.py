from __future__ import annotations

from curses import window
from datetime import datetime as dt
from itertools import combinations
from pathlib import Path
from typing import Callable, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy import typing as npt
from phdhelper.helpers import override_mpl
from shocksurvey import ENV, gen_timestamp, logger, ts_as_dt
from shocksurvey.constants import SC
from shocksurvey.mlshock import MLProperties
from shocksurvey.shock import Shock, Spacecraft
from shocksurvey.spedas import FGM, FPI, FSM
from tqdm import tqdm


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


def mag_3d(x: xr.DataArray, y: xr.DataArray, z: xr.DataArray) -> xr.DataArray:
    def func(a, b, c):
        return np.sqrt(a**2 + b**2 + c**2)

    return xr.apply_ufunc(func, x, y, z)


def win_dataset(
    data: xr.Dataset,
    start: int,
    end: int,
    names: list[str] = ["mag", "rad"],
) -> xr.Dataset:
    out = {}
    for n in names:
        out[n] = data[n][start:end]
    return xr.Dataset(out)


def plot_timeseries(ax: plt.Axes, fgm: xr.Dataset, windows: Bins) -> plt.Axes:  # type: ignore
    """Create a time series plot of |B| data."""

    # New line for each window
    for win in range(windows.start.size):
        data = fgm.mag[windows.start[win] : windows.end[win]]
        ax.plot(data.time, data.loc[:, "bt"])

    # Create new labels easier to read than mpl default
    labs = np.arange(int(fgm.time[0]), int(fgm.time[-1]), dtype=int)
    labs = labs[np.nonzero(labs % 60 == 0)]
    labs_str = [dt.strftime(ts_as_dt(la), "%H:%M") for la in labs]

    # Decoration
    ax.set_xticks(labs, labs_str)
    ax.set_xlabel("Time")
    ax.set_ylabel("|B|")

    return ax


# @logger.catch
def main():
    override_mpl.override()

    # Define the shock time
    timestamp = 1520916120.0
    trange = ["2018-03-13/04:42:00", "2018-03-13/04:57:00"]

    # Create shock object
    shock = Shock(trange=trange)
    # Create shock properties
    properties = MLProperties(timestamp=timestamp)
    # Add them to the shock
    shock.properties = properties

    mms: list[Spacecraft] = []  # mms 1-4
    for s in SC:
        mms.append(shock.add_spacecraft(s))  # Add spacecraft to shock & store in list
        mms[-1].add_fgm(properties.get_trange())  # Add fgm data to sc
    mms[0].add_fpi(properties.get_trange())  # Add fpi data for mms1

    # Create a list of 6 spacecraft pairs
    sc_pairs: list[tuple[Spacecraft, Spacecraft]] = list(combinations(mms, 2))

    # Define the windows
    windows = split_by_fraction(len(mms[0].fgm.data.mag.time), 1 / 5)  # type:ignore

    # Create output figure
    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios": [20, 80]})
    ax1, ax2 = cast(tuple[plt.Axes, plt.Axes], axs)  # type: ignore

    # Satisfy type checker
    assert mms[0].fgm is not None
    assert mms[0].fgm.data is not None

    # Create a timeseries plot in top panel
    ax1 = plot_timeseries(ax1, mms[0].fgm.data, windows)

    # Iterate over windows
    for win in range(windows.start.size):
        # Using MMS1 as the reference SC
        fgm1 = cast(
            xr.DataArray,
            mms[0].fgm.data.mag[windows.start[win] : windows.end[win]],  # type:ignore
        )

        # mean magnetic field for window
        b0 = float(fgm1.sel(b="bt").mean().values)
        f = lambda x: float(fgm1.time[x].values)
        time_slice = [f(1), f(-1)]
        # get ion velocity for same range as window
        fpi1 = mms[0].fpi.data.dis_bulkv.sel(time=slice(*time_slice))  # type: ignore

        # Mean bulk flow speed for window
        v0 = float(
            mag_3d(
                fpi1.sel(velocity="vx"),
                fpi1.sel(velocity="vy"),
                fpi1.sel(velocity="vz"),
            ).mean()
        )

        # Generate a set of lags (as indices) logarithmically spaced
        num_lags = 500
        fgm1_len = len(fgm1.time)
        lags_x = np.logspace(np.log10(1), np.log10(fgm1_len), num_lags, dtype=int)

        # Lag for each component bx,by,bz
        all_xyz = []
        for component in "xyz":
            # Initialise an array of nans
            lags = np.empty((num_lags, fgm1_len))
            lags[:] = np.nan

            # Iterate over each log lag
            for j, lag in enumerate(lags_x):
                # Fill only up to length - lag in array, since we don't want to have
                # a periodic array then increasing the lag makes the output array smaller
                # as there will be less and less overlap
                lags[j, : fgm1_len - lag] = abs(
                    fgm1.sel(b=f"b{component}").values[: fgm1_len - lag]  # type:ignore
                    - fgm1.sel(b=f"b{component}").values[lag:]  # type:ignore
                )  # difference between original and lagged data
            all_xyz.append(lags)

        lags = np.sqrt(
            np.abs(sum([xyz**2 for xyz in all_xyz]))
        )  # root square sum of compoenets
        fluc = np.nanmean(lags, axis=1) / b0  # nanmean so empty (nan) items not counted

        # Plot the lags assuming Taylor (in km)
        line = ax2.loglog(
            lags_x * v0 * 1 / 128,
            fluc,
            label=["sheath", "sheath", "sheath", "shock", "sw"][win],
        )
        # Store colour for use in scatter plots below
        c = line[0].get_color()  # type:ignore

        # delta b for each pair
        sep_flucs_list = []
        # Separation of each pair
        separations_list = []

        # Loop over the 6 pairs
        for pair in sc_pairs:
            fgm1 = win_dataset(
                pair[0].fgm.data, windows.start[win], windows.end[win]  # type:ignore
            )
            fgm2 = win_dataset(
                pair[1].fgm.data, windows.start[win], windows.end[win]  # type:ignore
            )

            # Calculate the difference, use linear interpolation so timesteps match exactly
            diff = abs(fgm1.rad - fgm2.interp_like(fgm1).rad)
            magnitude: xr.DataArray = mag_3d(
                diff[:, 0],
                diff[:, 1],
                diff[:, 2],
            ).mean()  # RMS magnitude

            # Store spearations
            separations_list.append(magnitude.values)

            # Index (seconds / spacecraft cadence)
            lag_index = int(magnitude.values * 128 / v0)
            point_fluc = (
                abs(
                    fgm1.mag.sel(b="bt").values[: len(fgm1.mag.time) - lag_index]
                    - fgm2.mag.sel(b="bt").values[lag_index:]
                ).mean()
                / b0
            )
            sep_flucs_list.append(point_fluc)

            # Scatter plot x=separation y=delta B
            ax2.scatter(
                magnitude.values, point_fluc, color=c, marker="x"  # type:ignore
            )

        # Stats
        index_of_mean_sep = np.argmin(
            np.abs(lags_x * v0 / 128 - np.mean(separations_list))
        )
        f1 = fluc[index_of_mean_sep]
        fs = np.mean(sep_flucs_list)
        logger.debug(
            f"MMS1: {f1:.2f} | SC Pairs: {fs:.2f} | Ratio: {fs/f1:.2f} |  B0: {b0:.2f} | V0: {v0:.2f}"
        )

    ax2.set_xlabel("lag (km)")
    ax2.set_ylabel(r"$\delta B/B_0$")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{Path(ENV.BASENAME)/'scripts/casestudies/qperp.jpg'}", dpi=300)
    plt.show()

    return
    for win in tqdm(range(len(fgm1.time))):
        lags[:, : lags.shape[1] - win] = abs(
            fgm1.mag.sel(b="bt").loc[win:]
            - fgm1.mag.sel(b="bt").loc[: lags.shape[1] - win]
        ).values
    lags = np.nanmean(lags, axis=1)
    bulkv = np.linalg.norm([properties.props[f"V{win}_us"] for win in "xyz"])
    lags_x = np.arange(0, len(lags) * 2 * np.pi * (1 / 128) / bulkv, len(lags))
    plt.plot(lags_x, lags)
    plt.show()


if __name__ == "__main__":
    main()
