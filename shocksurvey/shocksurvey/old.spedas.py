from __future__ import annotations

import dataclasses
from contextlib import redirect_stdout
from dataclasses import InitVar, asdict, dataclass
from datetime import datetime as dt
from datetime import timedelta
from enum import Enum
from io import StringIO
from optparse import Option
from os.path import join
from pprint import pformat
from typing import Any, List, Literal, Optional, Union

import cdflib
import numpy as np
import pandas as pd
from pyspedas.mms import mms_load_fgm

from shocksurvey import ENV, bprint, gen_timestamp


def download_data_from_timestamp(
    timestamp: float,
    experiment: Literal["fgm"] = "fgm",
    trange: Optional[List[str]] = None,
    no_download: bool = False,
    verbose: bool = False,
) -> List[str]:
    """Downloads cdf data.
    IN:
        timestamp:      The timestamp
        experiment:     E.g. 'fgm' for magnetometer data
        trange:         Set to ['YYYY-MM-DD/HH:MM:SS', 'YYYY-MM-DD/HH:MM:SS']
                        or leave as None to generate automatically.
        no_download:    True to get filepath (may not exist if data hasn't been downloaded)
                        False to download data
        verbose:        Set to True to print output from pyspedas
    OUT:
        fpath:          Full filepath to downloaded data
    """
    if not trange:
        trange = timestamp_to_trange(timestamp)

    if experiment == "fgm":
        with redirect_stdout(StringIO()) as stdout:
            fpath = mms_load_fgm(trange=trange, data_rate="brst", available=True)
            if not no_download:
                mms_load_fgm(trange=trange, data_rate="brst", notplot=True)
        if verbose:
            print("\n".join(stdout))
        return [get_mms_folder_path(timestamp, f) for f in fpath]
    else:
        raise NotImplementedError(f"Expreiment '{experiment}' not implemented.")


def timestamp_to_trange(
    timestamp: float, hours: int = 0, minutes: int = 1, seconds: int = 0
) -> List[str]:
    t_as_dt = dt.utcfromtimestamp(timestamp)
    t_delta = t_as_dt + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    trange = [
        gen_timestamp(t_as_dt, "pyspedas"),
        gen_timestamp(t_delta, "pyspedas"),
    ]
    return trange


def get_mms_folder_path(timestamp: float, fname: str) -> str:
    t_dt = dt.utcfromtimestamp(timestamp)
    sub_folders = "mms1/fgm/brst/l2"
    date_folders = f"{t_dt:%Y/%m/%d}"
    return join(ENV.MMS_DATA_DIR, sub_folders, date_folders, fname)


class CDF_VARS(str, Enum):
    time = "Epoch"
    r_time = "Epoch_state"
    b_gse = "mms1_fgm_b_gse_brst_l2"
    r_gse = "mms1_fgm_r_gse_brst_l2"
    label_b_gse = "label_b_gse"
    label_r_gse = "label_r_gse"


@dataclass
class Shock:
    cdf_path: List[str]
    properties: pd.Series

    def __post_init__(self):
        self.summary_plot_path = ""
        self.data: dict[str, Union[np.ndarray, pd.DataFrame]]
        if len(self.cdf_path) == 1:
            self.data = self.load_data_from_cdf(self.cdf_path[0])
        elif len(self.cdf_path) > 1:
            self.data = self.merge_cdf()
        else:
            raise ValueError("At least one CDF needs to be specified")

    def __getitem__(self, item) -> Any:
        if item in self.properties.index:
            return self.properties[item]
        else:
            raise KeyError(f"Cannot find {item} in Shock.properties")

    def __repr__(self) -> str:
        str = f"{' Shock ':-^80}\n"
        str += f"{self.cdf_path = }\n"
        str += f"{self.summary_plot_path = }\n"
        str += f"self.properties:\n{self.properties}\n{'-'*20}\n"
        str += f"self.data:\n{pformat(self.data)}\n{'-'*20}\n"
        return str

    def shock_to_df(self) -> dict[str, Any]:
        shock_dict = asdict(self)
        return shock_dict

    def merge_cdf(self):
        datasets = [self.load_data_from_cdf(cdf) for cdf in self.cdf_path]
        merged = {}
        # All datasets should have the same variables
        keys = datasets[0].keys()
        for k in keys:
            data = [cdf[k] for cdf in datasets]
            if all(isinstance(d, np.ndarray) for d in data):
                merged[k] = np.concatenate(data, axis=0)
            elif all(isinstance(d, pd.DataFrame) for d in data):
                merged[k] = pd.concat(data, axis=0)  # type: ignore
        return merged

    def load_data_from_cdf(
        self, path: str
    ) -> dict[str, Union[np.ndarray, pd.DataFrame]]:

        cdf: cdflib.cdfread.CDF = cdflib.CDF(path)  # type: ignore

        info: dict[str, Any] = cdf.cdf_info()
        variables = info["zVariables"]
        data = {}

        def create_and_label_frame(var, label: str, time: np.ndarray) -> pd.DataFrame:
            if label is None or label not in variables:
                raise ValueError(f"Supplied label {label} not in the zVariables")

            col = [s.strip() for s in cdf.varget(label)[0]]
            var_data = cdf.varget(var)
            return pd.DataFrame(var_data, columns=col, index=time)

        time: Optional[np.ndarray] = None
        r_time: Optional[np.ndarray] = None
        if CDF_VARS.time in variables:
            time = cdflib.cdfepoch.unixtime(
                cdf.varget(CDF_VARS.time), to_np=True
            )  # type:ignore
            data[CDF_VARS.time.name] = time
        if CDF_VARS.r_time in variables:
            r_time = cdflib.cdfepoch.unixtime(
                cdf.varget(CDF_VARS.r_time), to_np=True
            )  #  type:ignore
            data[CDF_VARS.r_time.name] = r_time
        if CDF_VARS.b_gse in variables:
            assert time is not None
            data[CDF_VARS.b_gse.name] = create_and_label_frame(
                CDF_VARS.b_gse, CDF_VARS.label_b_gse, time
            )
        if CDF_VARS.r_gse in variables:
            assert r_time is not None
            data[CDF_VARS.r_gse.name] = create_and_label_frame(
                CDF_VARS.r_gse, CDF_VARS.label_r_gse, r_time
            )

        return data


@dataclass
class Data:
    def add_var(self, var_name, var_data):
        pass
