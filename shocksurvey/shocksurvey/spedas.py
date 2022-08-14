from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from tkinter import W
from typing import Any, Callable, ClassVar, Literal, NamedTuple, Type

import cdflib
import numpy as np
import pandas as pd
import xarray as xr
from numpy import typing as npt
from pyspedas.mms import mms_load_fgm, mms_load_fpi

from shocksurvey import ENV, gen_timestamp, logger
from shocksurvey.constants import FPI_DTYPES, SC, Instruments


def get_date_folders(timestamp: float) -> Path:
    t_dt = dt.utcfromtimestamp(timestamp)
    date_folders = f"{t_dt:%Y/%m/%d}"
    return Path(date_folders)


def get_mms_folder_path(
    sc: SC,
    instrument: Instruments,
    rate: str,
    level: str,
) -> Path:
    sub_folders = Path(sc.value, instrument.value, rate, level)
    path = Path(ENV.MMS_DATA_DIR, sub_folders)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(str(path))
    return path


def find_existing_cdf(folder_path: Path) -> list[Path]:
    return list(folder_path.glob("*.cdf"))


def timestamp_to_trange(
    timestamp: float, hours: int = 0, minutes: int = 1, seconds: int = 0
) -> list[str]:
    t_as_dt = dt.utcfromtimestamp(timestamp)
    t_delta = t_as_dt + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    trange = [
        gen_timestamp(t_as_dt, "pyspedas"),
        gen_timestamp(t_delta, "pyspedas"),
    ]
    return trange


def create_and_label_df(
    data: npt.ArrayLike, time: npt.ArrayLike, labels: list[str]
) -> pd.DataFrame:
    col_names = [s.strip().lower().replace(" ", "_") for s in labels]
    logger.debug(f"{labels=} :: {np.array(data).shape=}")
    return pd.DataFrame(data, columns=col_names, index=pd.Series(time, name="time"))  # type: ignore


def arr_to_dataarr(
    data: npt.ArrayLike,
    dim_names: list[str],
    coordinates: dict[str, npt.ArrayLike],
    long_name: str,
    units: str,
    description: str,
) -> xr.DataArray:
    out = xr.DataArray(data=data, dims=dim_names, coords=coordinates)
    out.attrs["long_name"] = long_name
    out.attrs["units"] = units
    out.attrs["description"] = description
    return out


@dataclass
class CDF(ABC):
    instrument: ClassVar[Instruments]
    rate: ClassVar[str]
    level: ClassVar[str]
    timestamp: float
    sc: SC
    data: dict[Enum, pd.DataFrame] | None = None

    def __post_init__(self):
        self.folder: Path = get_mms_folder_path(
            self.sc,
            self.instrument,
            self.rate,
            self.level,
        )
        self.date_folder = get_date_folders(self.timestamp)
        self.trange: list[str] = timestamp_to_trange(self.timestamp)
        self.probe = str(self.sc)[-1]

    def update_trange(self, trange: list[str]) -> None:
        self.trange = trange

    def get_files_list_from_server(
        self, load_fn: Callable[..., list[str]]
    ) -> list[Path]:
        fpaths: list[str] = load_fn(
            trange=self.trange,
            probe=self.probe,
            data_rate=self.rate,
            available=True,
        )
        logger.debug(" :: ".join(fpaths))
        return [Path(fp) for fp in fpaths]

    @abstractmethod
    def download(self) -> list[Path]:
        pass

    @abstractmethod
    def load(self) -> None:
        pass


class FGM_LABEL(NamedTuple):
    B: pd.DataFrame
    R: pd.DataFrame


class FPI(CDF):
    instrument: ClassVar[Instruments] = Instruments.FPI
    rate: ClassVar[str] = "brst"
    level: ClassVar[str] = "l2"
    dtypes: list[FPI_DTYPES] = [FPI_DTYPES.DES, FPI_DTYPES.DIS]
    data: xr.Dataset | None

    def get_files_list_from_server(self) -> list[Path]:
        files: list[Path] = super().get_files_list_from_server(mms_load_fpi)
        paths: list[Path] = []
        for datatype in self.dtypes:
            for fp in files:
                if datatype in str(fp):
                    paths.append(self.folder / datatype.value / self.date_folder / fp)
        logger.debug(paths)
        return paths

    def download(self) -> list[Path]:
        fpaths: list[Path] = self.get_files_list_from_server()

        mms_load_fpi(
            trange=self.trange,
            data_rate=self.rate,
            notplot=True,
            probe=self.probe,
            datatype=["dis-moms", "des-moms"],  # type: ignore
        )
        return fpaths

    def load_fpi(self, path: Path, datatype: FPI_DTYPES) -> xr.Dataset:
        def _gen_mom(mom: str, dtype: str) -> str:
            """Super fn. generates a zVariable name from an FPI_MOMS name."""
            return f"{self.sc}_{dtype}_{mom}_{self.rate}"

        # Partial application of _gen_mom codes datatype
        short_datatype: str = datatype.split("-")[0]
        gen_mom: Callable[[str], str] = partial(_gen_mom, dtype=datatype.split("-")[0])
        if "i" in datatype.value:
            species = "ion"
        else:
            species = "electron"

        class FPI_MOMS(str, Enum):
            """Enum of desired moments"""

            time = "Epoch"
            tempperp = gen_mom("tempperp")
            temppara = gen_mom("temppara")
            bulkv = gen_mom("bulkv_gse")
            numberdensity = gen_mom("numberdensity")
            energyspectrogram = gen_mom("energyspectr_omni")
            energybins = gen_mom("energy")

        ################################################################################
        # Load data
        cdf: cdflib.cdfread.CDF = cdflib.CDF(path)  # type: ignore
        info: dict[str, Any] = cdf.cdf_info()
        variables: list[str] = info["zVariables"]

        # Lit of desired zVariables
        logger.debug(list(FPI_MOMS))
        for var in FPI_MOMS:
            if var not in variables:
                raise ValueError(f"Expected variable {var} not in zVariables")

        time: npt.ArrayLike = cdflib.cdfepoch.unixtime(
            cdf.varget(FPI_MOMS.time, to_np=True)
        )
        # Assuming all energy bins are the same
        ebins = cdf.varget(FPI_MOMS.energybins)
        ebins = np.array(ebins[0])  # Collapse to single list

        def f(x: FPI_MOMS, spec: Literal["des", "dis"]) -> str:
            return f"{spec}_{x.name}"

        f = partial(f, spec=short_datatype)  # type: ignore

        vars = {
            f(FPI_MOMS.tempperp): arr_to_dataarr(
                cdf.varget(FPI_MOMS.tempperp),
                dim_names=["time"],
                coordinates={"time": time},
                long_name=f"{species} perpendicular temperature",
                units="eV",
                description=f"{species} perpendicular temperature",
            ),
            f(FPI_MOMS.temppara): arr_to_dataarr(
                cdf.varget(FPI_MOMS.temppara),
                dim_names=["time"],
                coordinates={"time": time},
                long_name=f"{species} parallel temperature",
                units="eV",
                description=f"{species} parallel temperature",
            ),
            f(FPI_MOMS.bulkv): arr_to_dataarr(
                cdf.varget(FPI_MOMS.bulkv),
                dim_names=["time", "velocity"],
                coordinates={"time": time, "velocity": ["vx", "vy", "vz"]},
                long_name=f"{species} bulk velocity",
                units="km/s",
                description=f"{species} bulk velocity",
            ),
            f(FPI_MOMS.numberdensity): arr_to_dataarr(
                cdf.varget(FPI_MOMS.numberdensity),
                dim_names=["time"],
                coordinates={"time": time},
                long_name=f"{species} numberdensity",
                units="cm^-3",
                description=f"{species} numberdensity",
            ),
            f(FPI_MOMS.energyspectrogram): arr_to_dataarr(
                cdf.varget(FPI_MOMS.energyspectrogram),
                dim_names=["time", "energy"],
                coordinates={"time": time, "energy": ebins},
                long_name=f"{species} energy spectrogram",
                units="keV/(cm^2 s sr keV)",
                description=f"{species} energy spectrogram",
            ),
        }

        data = xr.Dataset(vars)
        return data

    def load(self) -> None:
        cdf: list[Path] = self.get_files_list_from_server()
        logger.debug(cdf)
        out = []
        for datatype in self.dtypes:
            # dis-moms or des-moms
            sub_cdf: list[Path] = [dt_cdf for dt_cdf in cdf if datatype in str(dt_cdf)]
            if not all([dt_cdf.exists() for dt_cdf in sub_cdf]):
                raise Exception(
                    f"Some files do not exist in the expected location. Please download using .download().\nMissing files {' - '.join([str(s) for s in sub_cdf])}"
                )
            moments_data: list[xr.Dataset] = []
            for data_file in sub_cdf:
                moments_data.append(self.load_fpi(data_file, datatype))
            out.append(xr.concat(moments_data, dim="time"))
        self.data = xr.merge(out)


class FGM(CDF):
    instrument: ClassVar[Instruments] = Instruments.FGM
    rate: ClassVar[str] = "brst"
    level: ClassVar[str] = "l2"
    labels: ClassVar[Type[FGM_LABEL]] = FGM_LABEL
    data: xr.Dataset | None

    def get_files_list_from_server(self) -> list[Path]:
        return super().get_files_list_from_server(mms_load_fgm)  # type: ignore

    def download(self) -> list[Path]:
        fpaths: list[Path] = self.get_files_list_from_server()

        mms_load_fgm(
            trange=self.trange,
            data_rate=self.rate,
            notplot=True,
            probe=self.probe,
        )
        return [self.folder / self.date_folder / f for f in fpaths]

    def load_cdf(self, path: Path) -> xr.Dataset:
        cdf: cdflib.cdfread.CDF = cdflib.CDF(path)  # type: ignore
        info: dict[str, Any] = cdf.cdf_info()
        variables = info["zVariables"]
        logger.debug(variables)

        class CDF_VARS(str, Enum):
            time = "Epoch"
            r_time = "Epoch_state"
            b_gse = f"mms{self.probe}_fgm_b_gse_brst_l2"
            r_gse = f"mms{self.probe}_fgm_r_gse_brst_l2"
            label_b_gse = "label_b_gse"
            label_r_gse = "label_r_gse"

        for var in CDF_VARS:
            assert var in variables

        time: npt.ArrayLike = cdflib.cdfepoch.unixtime(
            cdf.varget(CDF_VARS.time), to_np=True
        )
        r_time: npt.ArrayLike = cdflib.cdfepoch.unixtime(
            cdf.varget(CDF_VARS.r_time), to_np=True
        )
        out = {}
        out["mag"] = arr_to_dataarr(
            data=cdf.varget(CDF_VARS.b_gse),
            dim_names=["time", "b"],
            coordinates={"time": time, "b": ["bx", "by", "bz", "bt"]},
            long_name="FGM GSE",
            units="nT",
            description="Fluxgate Magnetometer",
        )
        out["rad"] = arr_to_dataarr(
            data=cdf.varget(CDF_VARS.r_gse),
            dim_names=["time", "r"],
            coordinates={"time": r_time, "r": ["x", "y", "z", "rt"]},
            long_name="Radius GSE",
            units="km",
            description="Radius",
        )
        return xr.Dataset(out)

    def load(self) -> None:
        cdf: list[Path] = [
            self.folder / self.date_folder / c
            for c in self.get_files_list_from_server()
        ]
        if not all([cdf_file.exists() for cdf_file in cdf]):
            # No files already downloaded
            cdf = self.download()

        cdf_files_data: list[xr.Dataset] = [self.load_cdf(c) for c in cdf]

        self.data = xr.concat(cdf_files_data, dim="time")


class FSM(CDF):
    instrument: ClassVar[Instruments] = Instruments.FPI
    rate: ClassVar[str] = "brst"
    level: ClassVar[str] = "l3"
