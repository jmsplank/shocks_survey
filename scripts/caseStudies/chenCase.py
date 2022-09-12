from __future__ import annotations

from curses import window
from datetime import datetime as dt
from itertools import combinations
from typing import Callable, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from numpy import typing as npt
from phdhelper.helpers import override_mpl
from shocksurvey import gen_filepath, gen_timestamp, logger, ts_as_dt
from shocksurvey.constants import FGM_CADENCE, SC
from shocksurvey.mlshock import MLProperties
from shocksurvey.shock import Shock, Spacecraft
from shocksurvey.spedas import FGM, FPI, FSM, replace_gaps_with_nan
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


# @logger.catch
def main():
    override_mpl.override()

    # timestamp = 1520916120.0
    # trange = ["2018-03-13/04:42:00", "2018-03-13/04:57:00"]
    trange = ["2015-10-16/09:24:11", "2015-10-16/09:25:24"]
    # timestamp = dt.strptime(trange[0], "%Y-%m-%d/%H:%M:%S").timestamp()

    shock = Shock(trange=trange)
    # properties = MLProperties(timestamp=timestamp)
    # shock.properties = properties
    # b0 = properties.norm_b()
    # v0 = properties.norm_v()

    mms: list[Spacecraft] = []
    for s in SC:
        mms.append(shock.add_spacecraft(s))
        mms[-1].add_fgm(trange)
    mms[0].add_fpi(trange, download=False)
    sc_pairs: list[tuple[Spacecraft, Spacecraft]] = list(combinations(mms, 2))
    separations = {}
    separations_list = []

    windows = split_by_fraction(len(mms[0].fgm.data.mag.time), 1)  # type:ignore
    # fig, ax = plt.subplots()
    # for win in range(windows.start.size):
    #     data = mms[0].fgm.data.mag[windows.start[win] : windows.end[win]]
    #     ax.plot(data.time, data.loc[:, "bt"])
    # ax.set_xlabel("Time")
    # ax.set_ylabel("|B|")
    # labs = np.arange(
    #     int(mms[0].fgm.data.time[0]), int(mms[0].fgm.data.time[-1]), dtype=int
    # )
    # labs = labs[np.nonzero(labs % 60 == 0)]
    # labs_str = [dt.strftime(ts_as_dt(la), "%H:%M") for la in labs]
    # ax.set_xticks(labs, labs_str)
    # plt.show()
    for win in range(windows.start.size):
        # Using MMS1 as the reference SC
        # fgm1 = cast(
        #     xr.DataArray,
        #     mms[0].fgm.data.mag[windows.start[win] : windows.end[win]],  # type:ignore
        # )
        fgm1 = replace_gaps_with_nan(
            mms[0].fgm.data.mag[windows.start[win] : windows.end[win]], FGM_CADENCE
        )
        b0 = float(fgm1.sel(b="bt").mean().values)
        f = lambda x: float(fgm1.time[x].values)
        time_slice = [f(1), f(-1)]
        fpi1 = mms[0].fpi.data.dis_bulkv.sel(time=slice(*time_slice))  # type: ignore

        v0 = float(
            mag_3d(
                fpi1.sel(velocity="vx"),
                fpi1.sel(velocity="vy"),
                fpi1.sel(velocity="vz"),
            ).mean()
        )

        num_lags = 100
        fgm1_len = len(fgm1.time)
        lags_x = np.logspace(np.log10(1), np.log10(fgm1_len), num_lags, dtype=int)
        logger.warning(f"removing {lags_x.size - np.unique(lags_x).size} duplicates")
        lags_x = np.unique(lags_x)
        num_lags = lags_x.shape[0]
        out = []
        for ic, component in enumerate(list("xyz")):
            lags = np.empty((num_lags, fgm1_len))
            lags[:] = np.nan
            for j, lag in enumerate(lags_x):
                lags[j, : fgm1_len - lag] = (
                    fgm1.sel(b=f"b{component}").values[: fgm1_len - lag]  # type:ignore
                    - fgm1.sel(b=f"b{component}").values[lag:]  # type:ignore
                )
            out.append(lags)
        lags = np.sqrt(np.abs(sum([o**2 for o in out])))  # |B(x+l)-B(x)|
        logger.debug(f"{lags.shape=}")
        m = np.nanmean(lags, axis=1) / b0  # <|B(x+l)-B(x)|>_x / b0

        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax = cast(list[plt.Axes], ax)  # type: ignore
        plags = lags[np.isfinite(lags)]
        logger.debug(f"min: {plags[plags>0].min()} | max: {plags.max()}")
        ax[0].grid(False)
        ax[0].pcolormesh(
            lags_x * v0 / 128,
            np.arange(fgm1_len) / 128,
            lags.T,
            norm=LogNorm(vmin=plags[plags > 0].min(), vmax=plags.max()),
            cmap="viridis",
        )
        ax[0].set_xscale("log")
        ax[1].loglog(lags_x * v0 / 128, m)
        ax[1].set_xlabel("Lag (km)")
        ax[0].set_ylabel("Time (s)")
        ax[1].set_ylabel(r"$\delta B/|B|$")

        # CHENTEST
        data = np.array(
            [
                [1.4543998581800373, 0.0026416483203860978],
                [2.7677282098848375, 0.0047542644301110625],
                [6.827792324102809, 0.01024275221381593],
                [12.059463454124948, 0.016155980984398754],
                [16.631156867048468, 0.02053525026457148],
                [24.097001932096894, 0.02673519359933993],
                [33.22994592714548, 0.03357698268059217],
                [53.13817563527232, 0.0447756260319464],
                [102.24375930366504, 0.06339913511724847],
                [165.4498751792918, 0.07773650302387762],
                [239.55245384091285, 0.08869857990181923],
                [330.13283811511076, 0.09880789942928697],
                [472.05408018177724, 0.10875727897549065],
                [666.5119375819471, 0.11274137659327856],
                [726.6023388007027, 0.11547819846894584],
                [907.2366541049897, 0.12409377607517198],
                [989.0299596282338, 0.12710617996147452],
            ]
        )
        ax[1].plot(data[:, 0], data[:, 1])
        logger.debug(
            f"Ratio me/chen: {(np.interp(data[:,0], lags_x, m)/data[:,1]).mean():.4f}"
        )

        fluc = np.nanmean(lags, axis=1) / b0
        line = plt.loglog(
            lags_x * v0 * 1 / 128,
            fluc,
            label=["sheath", "sheath", "sheath", "shock", "sw"][win],
        )
        c = line[0].get_color()  # type:ignore

        sep_flucs_list = []
        for pair in sc_pairs:
            fgm1 = win_dataset(
                pair[0].fgm.data, windows.start[win], windows.end[win]  # type:ignore
            )
            fgm2 = win_dataset(
                pair[1].fgm.data, windows.start[win], windows.end[win]  # type:ignore
            )

            diff = abs(fgm1.rad - fgm2.interp_like(fgm1).rad)
            magnitude: xr.DataArray = mag_3d(
                diff[:, 0],
                diff[:, 1],
                diff[:, 2],
            ).mean()
            separations_list.append(magnitude.values)
            separations[(pair[0].name, pair[1].name)] = magnitude.values
            lag_index = int(magnitude.values * 128 / v0)
            point_fluc = (
                abs(
                    fgm1.mag.sel(b="bt").values[: len(fgm1.mag.time) - lag_index]
                    - fgm2.mag.sel(b="bt").values[lag_index:]
                ).mean()
                / b0
            )
            sep_flucs_list.append(point_fluc)
            # lab = f"{pair[0].name.name[-1]}{pair[1].name.name[-1]}"
            plt.scatter(
                magnitude.values, point_fluc, color=c, marker="x"  # type:ignore
            )
        f1 = fluc[np.argmin(np.abs(lags_x * v0 / 128 - np.mean(separations_list)))]
        fs = np.mean(sep_flucs_list)  # type:ignore
        logger.debug(
            f"MMS1: {f1:2f} | SC Pairs: {fs:.2f} | Ratio: {fs/f1:.2f} |  B0: {b0:.2f} | V0: {v0:.2f}"
        )

        # print(
        #     "["
        #     + ",".join(
        #         [
        #             f"[{separations_list[i]}, {sep_flucs_list[i]}]"
        #             for i in range(len(separations_list))
        #         ]
        #     )
        #     + "]"
        # )

    plt.xlabel("lag (km)")
    plt.ylabel(r"$\delta B/B_0$")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
