import os
from datetime import datetime as dt
from os.path import dirname, join, exists
import inspect

import pandas as pd

from shocksurvey import ENV


class DC:
    """Data Columns"""

    TIME = "time"
    DIRECTION = "direction"
    BURST_START = "burst_start"
    BURST_END = "burst_end"
    BX_US = "Bx_us"
    BY_US = "By_us"
    BZ_US = "Bz_us"
    B_US_ABS = "B_us_abs"
    SB_US_ABS = "sB_us_abs"
    DELTA_THETA_B = "Delta_theta_B"
    NI_US = "Ni_us"
    TI_US = "Ti_us"
    VX_US = "Vx_us"
    VY_US = "Vy_us"
    VZ_US = "Vz_us"
    BETA_I_US = "beta_i_us"
    PDYN_US = "Pdyn_us"
    THBN = "thBn"
    STHBN = "sthBn"
    NORMAL_X = "normal_x"
    NORMAL_Y = "normal_y"
    NORMAL_Z = "normal_z"
    MA = "MA"
    SMA = "sMA"
    MF = "Mf"
    SMF = "sMf"
    B_JUMP = "B_jump"
    NI_JUMP = "Ni_jump"
    TE_JUMP = "Te_jump"
    POS_X = "pos_x"
    POS_Y = "pos_y"
    POS_Z = "pos_z"
    SC_SEP_MIN = "sc_sep_min"
    SC_SEP_MAX = "sc_sep_max"
    SC_SEP_MEAN = "sc_sep_mean"
    TQF = "TQF"

    def all_params(self):
        attrs = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attrs = [a for a in attrs if not (a[0].startswith("__"))]
        return [a[1] for a in attrs]


def load_data(file_path: str = None) -> pd.DataFrame:
    """Loads CSV data file as pandas dataframe.
    IN:
        file_path: if None then loaded from config.env
    """
    if file_path is None:
        file_path = ENV.ML_DATA_LOCATION
    data = pd.read_csv(file_path, comment="#")
    return data


def filter_data(
    data: pd.DataFrame,
    non_burst: bool = True,
    missing_data: list = [],
    missing_value: float = -1e30,
) -> pd.DataFrame:
    """Filter the data according to flags
    IN:
        data:           The data
        non_burst:      Remove data that doesn't have an associated burst interval
        missing_data:   List of columns to filter for missing data.
                        Removes entire row from data if value in any specified column
                        is equal to missing_value. E.g. [DC.THBN, DC.MA] removes rows
                        where theta_bn and mach number contain missing data.
        missing_value:  Value to use as missing.
    OUT:
        data: Filtered data
    """
    if non_burst:
        data = remove_non_burst(data)
    if len(missing_data) > 0:
        for key in missing_data:
            data = data[data[key] != missing_value]
    return data


def get_plot_file(
    timestamp: float,
    file_fmt: str = "MMS1_shocks__{TIME_FMT}.png",
    time_fmt: str = "%Y%m%d_%H%M%S",
    full_path: bool = True,
) -> str:
    """Get filename of shock from timestamp.
    IN:
        timestamp:  Timestamp of the shock in UTC seconds from 01/01/1970
        file_fmt:   Naming convention of plots. Substitute {TIME_FMT} where
                    formatted date should go
        time_fmt:   Format str used in strftime to generate plot time
        full_path:  Return full path to file or just the filename & check file exists
    OUT:
        Full path to file or filename
    """
    time_dt = dt.utcfromtimestamp(timestamp)
    time_fmt = dt.strftime(time_dt, time_fmt)
    plot_name = file_fmt.replace("{TIME_FMT}", time_fmt)

    if full_path:
        plot_path = join(dirname(ENV.ML_DATA_LOCATION), "plots", plot_name)
        # Test file exists
        if not exists(plot_path):
            raise FileNotFoundError(f"file cannot be found at {plot_path}.\nCheck file_fmt and time_fmt parameters as well as ML_DATA_LOCATION in config.env")  # fmt: skip
        return plot_path

    return plot_name


def open_shock_plot(timestamp: str, **kwargs) -> None:
    """Wrapper for get_shock_plot. Opens image file in default viewer"""
    file = get_plot_file(timestamp, **kwargs)
    os.system(f"open {file}")


def remove_non_burst(data):
    return data[data.burst_start != 0]
