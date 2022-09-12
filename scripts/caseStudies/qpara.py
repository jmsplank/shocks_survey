from __future__ import annotations

from datetime import datetime as dt
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pytz
import xarray as xr
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from phdhelper.helpers import override_mpl
from shocksurvey import ENV, logger, ts_as_dt
from shocksurvey.constants import FGM_CADENCE, SC
from shocksurvey.shock import Shock, Spacecraft
from shocksurvey.spedas import replace_gaps_with_nan
from tqdm import tqdm


class Bins(NamedTuple):
    start: np.ndarray
    end: np.ndarray


def to_d(s: str, t_key: str = "%Y-%m-%d %H:%M:%S.%f") -> dt:
    return dt.strptime(s, t_key)


def split_by_fraction(length: int, fraction: float) -> Bins:
    if not 0 < fraction <= 1:
        raise ValueError(
            f"fraction should be in the range [0,1). Recieved {fraction:0.2f}"
        )
    bin_start = np.linspace(0, length, int(1 / fraction) + 1, dtype=int)[:-1]
    bin_end = np.linspace(0, length, int(1 / fraction) + 1, dtype=int)[1:]
    return Bins(bin_start, bin_end)


def split_by_times(
    data: xr.DataArray, groups: dict[str, list[str]], dim_name: str
) -> Bins:
    f = lambda x: dt.strptime(x, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=pytz.utc)
    timestamps = {
        k: (dt.timestamp(f(v[0])), dt.timestamp(f(v[1]))) for k, v in groups.items()
    }
    bin_starts = []
    bin_ends = []
    index = data[dim_name].values
    logger.debug(f"{dt.utcfromtimestamp(index[0]):%Y-%m-%d %H:%M:%S.%f}")
    logger.debug(f"{dt.utcfromtimestamp(index[-1]):%Y-%m-%d %H:%M:%S.%f}")
    logger.debug(f"data: {index[0]:.0f}, {index[-1]:.0f} : dict: {timestamps}")
    for key in timestamps:
        bin_starts.append(np.argmin(np.abs(index - timestamps[key][0])))
        bin_ends.append(np.argmin(np.abs(index - timestamps[key][1])))
    return Bins(start=np.array(bin_starts), end=np.array(bin_ends))


def plot_timeseries(ax: plt.Axes, mag: xr.DataArray, windows: Bins) -> plt.Axes:  # type: ignore
    """Create a time series plot of |B| data."""

    # New line for each window
    for win in range(windows.start.size):
        data = mag[windows.start[win] : windows.end[win]]
        ax.plot(data.time, data.loc[:, "bt"])

    # Create new labels easier to read than mpl default
    labs = np.arange(int(mag.time[0]), int(mag.time[-1]), dtype=int)
    steps = np.array([1 / 6, 0.5, 1, 5, 10, 15, 20, 30, 60]) * 60
    duration = float(mag.time[-1].values - mag.time[0].values)
    exact_step = duration / 10
    rounded_step = steps[np.argmax(steps > exact_step)]
    labs = labs[np.nonzero(labs % rounded_step == 0)]
    fmt_str = "%H:%M"
    if rounded_step < 60:
        fmt_str += ":%S"
    labs_str = [dt.strftime(ts_as_dt(la), fmt_str) for la in labs]

    # Decoration
    ax.set_xticks(labs, labs_str)
    ax.set_xlabel("Time")
    ax.set_ylabel("|B|")

    return ax


def plot_fluctuations(data_dict: dict[str, Any]) -> tuple[Figure, list[Axes]]:
    fig, axs = plt.subplots(
        nrows=3, ncols=1, gridspec_kw={"height_ratios": [20, 60, 20]}
    )
    ax1, ax2, ax3 = cast(tuple[plt.Axes, plt.Axes, plt.Axes], axs)  # type: ignore

    ax1 = plot_timeseries(ax1, data_dict["mms1_mag"], data_dict["windows_times"])

    for win in range(len(data_dict["wins"])):
        wd = data_dict["wins"][win]
        line = ax2.loglog(
            wd["ssc_lines"][0][:, 0],
            wd["ssc_lines"][0][:, 1],
            label=wd["win_name"],
            ls=["-", ":", "--", "-.", (0, (3, 5, 1, 5, 1, 5))][win],
        )
        c = line[0].get_color()  # type:ignore

        scatterfmt = dict(
            facecolors="none",
            edgecolors=c,
        )
        ms = "XDo^s"[win]

        # Scatter plot x=separation y=delta B
        ax2.scatter(
            wd["pair_flucs"][:, 0],
            wd["pair_flucs"][:, 1],
            marker=ms,
            **scatterfmt,
        )
        ax3.scatter(
            wd["pair_angles"][:, 0],
            wd["pair_angles"][:, 1],
            marker=ms,
            **scatterfmt,
        )
    ax2.set_xlabel("lag (km)")
    ax2.set_ylabel(r"$\delta B/B_0$")
    ax2.legend()

    ax3.axhline(y=1, color="k")
    ax3.set_xlim((0, 180))
    ax3_y_deviation = np.max(np.abs(np.array(ax3.get_ylim()) - 1)) * 1.1
    ax3.set_ylim((1 - ax3_y_deviation, 1 + ax3_y_deviation))
    ax3.set_xlabel(r"$\theta_l$")
    ax3.set_ylabel(
        r"$\frac{\left.|\delta \mathbf{B}|/B_0\right|_\mathrm{Pair}}{\left.|\delta \mathbf{B}|/{B_0}\right|_\mathrm{Taylor}}$"
    )

    plt.tight_layout()
    return fig, axs


# @logger.catch
def main(trange: list[str], win_dict: dict[str, list[str]]) -> dict[str, Any]:
    """analysis relating to the Taylor hypothesis

    Args:
        trange (list[str]): (start_time, end_time)
        win_dict (dict[str, list[str]]): dict of window start and end times

    Returns:
        dict[str, Any]: Information for plotting analysis
            schema:
            data_dict = {
                windows_times: Bins,
                mms1_mag: xr.DataArray,
                wins: {
                    win_name: str,
                    win_num: int,
                    ssc_lines: [
                        array(lags_km, fluc),
                        ...
                    ],
                    pair_flucs: array(separations, fluctuations),
                    pair_angles: array(angles_to_b0, ratios_to_line),
                    mean_stats: {
                        fluc_at_mean_sep: float,
                        mean_fluc_of_pairs: float,
                        average_b: float,
                        average_v: float,
                    }
                }
            }
    """
    override_mpl.override()

    # Create shock object
    shock = Shock(trange=trange)

    mms: list[Spacecraft] = []  # mms 1-4
    for s in SC:
        mms.append(shock.add_spacecraft(s))  # Add spacecraft to shock & store in list
        mms[-1].add_fgm(trange)  # Add fgm data to sc
    mms[0].add_fpi(trange, download=True)  # Add fpi data for mms1

    # Satisfy type checker
    assert mms[0].fgm is not None
    assert mms[0].fgm.data is not None

    mms1_mag = replace_gaps_with_nan(mms[0].fgm.data.mag, FGM_CADENCE)

    # Create a list of 6 spacecraft pairs
    sc_pairs: list[tuple[Spacecraft, Spacecraft]] = list(combinations(mms, 2))

    # Define the windows
    windows: Bins = split_by_times(
        mms1_mag,
        win_dict,
        "time",
    )

    data_out = {}
    data_out["windows_times"] = windows
    data_out["mms1_mag"] = mms1_mag

    wins = []
    # Iterate over windows
    for win in range(windows.start.size):
        win_key = list(win_dict.keys())[win]
        logger.info(f"start {win_key} -> {win_dict[win_key]}")
        logger.info(f"indices: {windows.start[win]} : {windows.end[win]}")
        # Using MMS1 as the reference SC
        fgm1 = cast(
            xr.DataArray,
            mms1_mag[windows.start[win] : windows.end[win]],  # type:ignore
        )
        wd = {"win_name": win_key, "win_num": win}

        # mean magnetic field for window
        b0 = float(fgm1.sel(b="bt").mean(skipna=True).values)
        b0_vec = fgm1.mean(dim="time", skipna=True).sel(b=["bx", "by", "bz"]).values
        b0_vec /= np.linalg.norm(b0_vec)
        logger.debug(
            f"{list(win_dict.keys())[win]} b0 direction: {b0_vec} magnitude {b0:.3f}nT"
        )

        # get ion velocity for same range as window
        f = lambda x: float(fgm1.time[x].values)
        time_slice = [f(1), f(-1)]
        fpi1 = mms[0].fpi.data.dis_bulkv.sel(time=slice(*time_slice))  # type: ignore

        # Mean bulk flow speed for window
        v0 = np.nanmean(
            np.linalg.norm(fpi1.sel(velocity=["vx", "vy", "vz"]).values, axis=1)
        )

        # Generate a set of lags (as indices) logarithmically spaced
        num_lags = 500
        fgm1_len = len(fgm1.time)
        lags_x = np.logspace(np.log10(1), np.log10(fgm1_len - 1), num_lags, dtype=int)
        lags_x = np.unique(lags_x)
        logger.debug(f"Removing {num_lags-len(lags_x)} duplicate lags_x from list")
        num_lags = len(lags_x)

        # Initialise an array of nans
        lags = np.empty((num_lags, fgm1_len, 3))
        lags[:] = np.nan

        # Iterate over each log lag
        for j, lag in enumerate(lags_x):
            # Fill only up to length - lag in array, since we don't want to have
            # a periodic array then increasing the lag makes the output array smaller
            # as there will be less and less overlap
            lags[j, : fgm1_len - lag, :] = (
                fgm1.sel(b=["bx", "by", "bz"]).values[: fgm1_len - lag]  # type:ignore
                - fgm1.sel(b=["bx", "by", "bz"]).values[lag:]  # type:ignore
            )  # difference between original and lagged data

        lags = np.abs(np.linalg.norm(lags, axis=2))
        fluc = np.nanmean(lags, axis=1) / b0  # nanmean so empty (nan) items not counted

        wd["ssc_lines"] = []
        wd["ssc_lines"].append(np.column_stack((lags_x / 128 * v0, fluc)))

        # delta b for each pair
        pair_fluctuations = []
        # Separation of each pair
        pair_separations = []
        # Angles to average magnetic field in window
        pair_angle_to_b0 = []
        # Ratio of single spacecraft fluctuation to fluctuations between pair
        pair_ratio_to_line = []

        # Loop over the 6 pairs
        for pair in sc_pairs:
            fgm1 = replace_gaps_with_nan(pair[0].fgm.data.mag, FGM_CADENCE)[
                windows.start[win] : windows.end[win]
            ]
            fgm2 = replace_gaps_with_nan(pair[1].fgm.data.mag, FGM_CADENCE)[
                windows.start[win] : windows.end[win]
            ]
            fgm2 = fgm2.interp_like(fgm1)  # Match timesteps exactly
            logger.debug(f"{fgm1.time.values[0]-fgm2.time.values[0]=:.2f}")

            rad1 = replace_gaps_with_nan(pair[0].fgm.data.rad, FGM_CADENCE)[
                windows.start[win] : windows.end[win]
            ]
            rad2 = replace_gaps_with_nan(pair[1].fgm.data.rad, FGM_CADENCE)[
                windows.start[win] : windows.end[win]
            ]
            rad2 = rad2.interp_like(rad1)

            # Calculate the difference
            diff: np.ndarray = (rad1.values - rad2.values)[:, :3]  # type: ignore
            magnitude: float = np.nanmean(np.abs(np.linalg.norm(diff, axis=1)))

            # pos_vec = diff.mean(dim="time", skipna=True).sel(r=["x", "y", "z"]).values
            pos_vec = np.nanmean(diff, axis=0)
            pos_vec /= np.linalg.norm(pos_vec)
            angle_to_b = np.rad2deg(np.arccos(np.clip(np.dot(b0_vec, pos_vec), -1, 1)))
            logger.debug(
                f"({pair[0].name}, {pair[1].name}) -> angle to b0: {angle_to_b:03.0f} degrees"
            )

            # Store spearations
            pair_separations.append(magnitude)

            # Index (seconds / spacecraft cadence)
            lag_index = int(magnitude * 128 / v0)
            sc1_lagsized: np.ndarray = fgm1.sel(b=["bx", "by", "bz"]).values[
                : len(fgm1.time) - lag_index
            ]
            sc2_lagsized: np.ndarray = fgm2.sel(b=["bx", "by", "bz"]).values[lag_index:]
            lag_diff_vec = sc1_lagsized - sc2_lagsized  # type: ignore
            lag_diff = np.linalg.norm(lag_diff_vec, axis=1)
            point_fluc = np.nanmean(np.abs(lag_diff)) / b0
            pair_fluctuations.append(point_fluc)

            logger.info(
                f"({pair[0].name}, {pair[1].name}) -> sep: {magnitude:05.2f}km :: fluc: {point_fluc:06.3f}"
            )

            taylor_fluc_at_sep_i = np.argmin(np.abs(lags_x * v0 / 128 - magnitude))
            taylor_fluc_at_sep = fluc[taylor_fluc_at_sep_i]

            pair_angle_to_b0.append(angle_to_b)
            pair_ratio_to_line.append(point_fluc / taylor_fluc_at_sep)

        wd["pair_flucs"] = np.column_stack((pair_separations, pair_fluctuations))
        wd["pair_angles"] = np.column_stack((pair_angle_to_b0, pair_ratio_to_line))

        # Stats
        index_of_mean_sep = np.argmin(
            np.abs(lags_x * v0 / 128 - np.mean(pair_separations))
        )
        f1 = fluc[index_of_mean_sep]
        fs = np.mean(pair_fluctuations)

        wd["mean_stats"] = dict(
            fluc_at_mean_sep=f1,
            mean_fluc_of_pairs=fs,
            average_b=b0,
            average_v=v0,
        )

        wins.append(wd)

    data_out["wins"] = wins

    return data_out


def correct_win_dict(wd: dict[str, list[str]]) -> dict[str, list[str]]:

    out: dict[str, list[dt]] = {k: [to_d(i) for i in v] for k, v in wd.items()}
    out = {k: v for k, v in sorted(out.items(), key=lambda item: item[1][0])}

    keys = list(out.keys())
    end_t = out[keys[0]][1]
    for k in keys[1:]:
        out[k][0] = end_t
        end_t = out[k][1]

    def f(lst: list[dt], t_key: str = "%Y-%m-%d %H:%M:%S.%f") -> list[str]:
        return [dt.strftime(i, t_key) for i in lst]

    return {k: f(v) for k, v in out.items()}


if __name__ == "__main__":
    trange = ["2020-03-20/19:26:00", "2020-03-20/20:56:00"]
    win_dict = {
        "fs": ["2020-03-20 19:24:11.4915", "2020-03-20 20:26:01.5791"],
        "shock": ["2020-03-20 20:26:01.5791", "2020-03-20 20:34:53.6865"],
        "ms": ["2020-03-20 20:34:56.2682", "2020-03-20 20:55:51.1508"],
    }
    win_dict = correct_win_dict(win_dict)
    data_dict = main(trange, win_dict)

    fig, axs = plot_fluctuations(data_dict=data_dict)
    savepath = (
        lambda e: Path(ENV.BASENAME)
        / f"scripts/caseStudies/{to_d(win_dict[list(win_dict.keys())[0]][0]):%Y%m%d}_taylorTest.{e}"
    )
    plt.savefig(savepath("pdf"), dpi=300)
    # plt.savefig(savepath("svg"), dpi=300)
    plt.savefig(savepath("jpg"), dpi=300)
    plt.show()
