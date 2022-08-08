from __future__ import annotations

from datetime import datetime as dt

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from phdhelper.helpers import override_mpl
from shocksurvey import gen_filepath, gen_timestamp, load_arr_from_data
from shocksurvey.mlshocks import SHK, load_data
from shocksurvey.spedas import Shock, download_data_from_timestamp
from tqdm import tqdm

override_mpl.override()


def f(x):
    return dt.strptime(x, "%d/%m/%Y %H:%M")


def g(y):
    return gen_timestamp(f(y), "pyspedas")


def old_rolling_window(
    start: float, stop: float, width: float = 60, step: float = 1
) -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(start, stop, step), np.arange(start + width, stop, step))


def rolling_window(
    arr,
    step: int,
    dt: float | None = 1 / 8192,
    width: float | None = None,
    iwidth: int | None = None,
):
    if not dt:
        pass
    if not iwidth and width:
        iwidth = np.argmin(np.abs((arr - arr[0]) - width))  # type:ignore
    assert iwidth is not None
    return (
        np.arange(0, len(arr) - iwidth, step, dtype=int),
        np.arange(iwidth, len(arr), step, dtype=int),
        iwidth,
    )


def fractional_window(
    arr: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    arr_length = len(arr)
    step_size = int(arr_length * fraction)
    bin_starts = np.arange(0, arr_length - step_size, step_size)
    bin_ends = np.arange(step_size, arr_length, step_size)
    return bin_starts, bin_ends, step_size


def rolling_data(
    arr: np.ndarray, bin_starts: np.ndarray, bin_ends: np.ndarray
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    out = np.empty(len(bin_starts))
    for i, _ in enumerate(out):
        out[i] = np.nanmean(arr[bin_starts[i] : bin_ends[i]], axis=0)
    return out


def second_structure(b: np.ndarray, iw: int | None = None) -> np.ndarray:
    if iw is not None:
        auto = np.correlate(b, b, mode="same")[len(b) // 2 : len(b) // 2 + iw]
    else:
        auto = np.correlate(b, b, mode="same")[len(b) // 2 :]
    return auto / auto[0]


def apply_fn_to_windows(data, fn, t_s, t_e, iwidth):
    """ """
    out = np.empty((len(t_s), iwidth))
    for win in tqdm(range(len(t_s))):
        sub_data = data[t_s[win] : t_e[win]]
        res = fn(sub_data)
        out[win, :] = res
    return out


def gen_structurefn(
    data: npt.NDArray, no_cache: bool = False
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    CACHE_STR = (
        gen_timestamp(dt.utcfromtimestamp(timestamp), "files")
        + f"_structurefn_{int(sum(data))}.npy"
    )
    cache_fpath = gen_filepath("cache", CACHE_STR)
    if cache_fpath.is_file() or no_cache:
        cached_data = np.load(cache_fpath)
        return cached_data[:, :, 0], cached_data[:, :, 1]
    else:
        out = np.empty((len(data), len(data)))
        out[:] = np.nan
        for i in range(len(data)):
            out[i, : out.shape[1] - i] = np.abs(np.roll(data, -i) - data)[: out.shape[1] - i]  # type: ignore
        res_mean = np.nanmean(out, axis=1)
        res_std = np.sqrt(np.nanstd(out, axis=1))
        np.save(cache_fpath, np.dstack([res_mean, res_std]))
        return res_mean, res_std


def load_structurefn(
    fname: str = "structurefn_(118912, 3840).npy", folder: str = "20180313044200_cached"
):
    return load_arr_from_data(folder, fname)


def create_animated_strucfn(x, yframes):
    fig, ax = plt.subplots()

    (line,) = ax.plot(x, yframes[0])

    def animate(frame: int) -> tuple[Line2D]:
        line.set_ydata(yframes[frame])
        return (line,)

    print(f"Frames: {yframes.shape=} :: {yframes.shape[0]}")

    ani = animation.FuncAnimation(
        fig, animate, yframes.shape[0], interval=20, blit=True
    )

    writer = animation.FFMpegWriter(fps=15, bitrate=1800)  # type:ignore
    ani.save(gen_filepath(folder, "animated_structurefn.mp4"), writer=writer)  # type: ignore


if __name__ == "__main__":
    time = ["13/03/2018 04:42", "13/03/2018 04:57"]

    timestamp = dt.timestamp(dt.strptime(time[0], "%d/%m/%Y %H:%M"))
    print(timestamp)

    data = load_data()
    print(len(data))
    closest_shock_index = np.argmin(np.abs(data[SHK.TIME] - timestamp))
    closest_shock = data.iloc[closest_shock_index]

    trange = list(map(g, time))
    print(closest_shock)
    print(trange)

    cdf_path = download_data_from_timestamp(
        timestamp=closest_shock[SHK.BURST_START], no_download=False, trange=trange
    )
    shock = Shock(cdf_path, closest_shock)
    print(shock)

    td = 1 / 128
    # t_s, t_e, iwidth = rolling_window(shock.data["time"], 1, width=60)
    t_s, t_e, iwidth = fractional_window(np.array(shock.data["b_gse"]["Bt"]), 1 / 6)

    b_0 = np.array(
        [
            np.nanmean(np.array(shock.data["b_gse"]["Bt"])[t_s[i] : t_e[i]])
            for i in range(len(t_s))
        ]
    )

    t = np.arange(0, td * (iwidth), td)
    folder = gen_timestamp(dt.utcfromtimestamp(timestamp), "files") + "_cached"

    # sf = load_structurefn(folder=folder)[::120, :]
    # sf = gen_structurefn(shock.data["b_gse"]["Bt"], t_s, t_e, iwidth)
    sf = apply_fn_to_windows(
        np.array(shock.data["b_gse"]["Bt"]),
        lambda data: gen_structurefn(data)[0],
        t_s,
        t_e,
        iwidth,
    )

    sf_norm = np.divide(sf, np.tile(b_0, (sf.shape[1], 1)).T)

    for win in range(sf_norm.shape[0]):
        line = plt.loglog(t, sf_norm[win], label=f"{win+1}/{sf_norm.shape[0]}")
    plt.legend()
    mag_fluc = np.empty(sf.shape[1])
    plt.show()
