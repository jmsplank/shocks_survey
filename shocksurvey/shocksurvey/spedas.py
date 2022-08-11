from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timedelta
from enum import Enum, auto
from heapq import merge
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Type

import cdflib
import pandas as pd
from numpy import typing as npt
from pyspedas.mms import mms_load_fgm

from shocksurvey import ENV, gen_timestamp, logger
from shocksurvey.constants import SC, Instruments


def get_mms_folder_path(
    timestamp: float,
    sc: SC,
    instrument: Instruments,
    rate: str,
    level: str,
) -> Path:
    t_dt = dt.utcfromtimestamp(timestamp)
    sub_folders = Path(sc.value, instrument.value, rate, level)
    date_folders = f"{t_dt:%Y/%m/%d}"
    path = Path(ENV.MMS_DATA_DIR, sub_folders, date_folders)
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
    return pd.DataFrame(data, columns=col_names, index=pd.Series(time, name="time"))  # type: ignore


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
            self.timestamp, self.sc, self.instrument, self.rate, self.level
        )
        self.trange: list[str] = timestamp_to_trange(self.timestamp)
        self.probe = str(self.sc)[-1]

    def update_trange(self, trange: list[str]) -> None:
        self.trange = trange

    @abstractmethod
    def download(self) -> list[Path]:
        pass

    @abstractmethod
    def load(self) -> None:
        pass


class FPI(CDF):
    instrument: ClassVar[Instruments] = Instruments.FPI
    rate: ClassVar[str] = "brst"
    level: ClassVar[str] = "l2"

    def load(self) -> None:
        # Check for existing data
        cdf = find_existing_cdf(self.folder)
        if len(cdf) == 0:
            pass


class FGM_LABEL(NamedTuple):
    B: pd.DataFrame
    R: pd.DataFrame


class FGM(CDF):
    instrument: ClassVar[Instruments] = Instruments.FGM
    rate: ClassVar[str] = "brst"
    level: ClassVar[str] = "l2"
    labels: ClassVar[Type[FGM_LABEL]] = FGM_LABEL
    data: FGM_LABEL | None

    def get_files_list_from_server(self) -> list[Path]:
        fpaths: list[str] = mms_load_fgm(
            trange=self.trange,
            probe=self.probe,
            data_rate=self.rate,
            available=True,
        )
        logger.debug("\n".join(fpaths))
        return [Path(fp) for fp in fpaths]

    def download(self) -> list[Path]:
        fpaths: list[Path] = self.get_files_list_from_server()

        mms_load_fgm(
            trange=self.trange,
            data_rate=self.rate,
            notplot=True,
            probe=self.probe,
        )
        return [self.folder / f for f in fpaths]

    def load_cdf(self, path: Path) -> FGM_LABEL:
        cdf: cdflib.cdfread.CDF = cdflib.CDF(path)  # type: ignore
        info: dict[str, Any] = cdf.cdf_info()
        variables = info["zVariables"]

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
        b = create_and_label_df(
            cdf.varget(CDF_VARS.b_gse),
            time,
            cdf.varget(CDF_VARS.label_b_gse)[0],
        )
        r = create_and_label_df(
            cdf.varget(CDF_VARS.r_gse),
            r_time,
            cdf.varget(CDF_VARS.label_r_gse)[0],
        )

        return FGM_LABEL(b, r)

    def load(self) -> None:
        cdf: list[Path] = [self.folder / c for c in self.get_files_list_from_server()]
        if not all([cdf_file.exists() for cdf_file in cdf]):
            # No files already downloaded
            cdf = self.download()

        cdf_files_data: list[FGM_LABEL] = [self.load_cdf(c) for c in cdf]
        merged = {}
        for i, _ in enumerate(cdf_files_data[0]):
            data = [cdf_data[i] for cdf_data in cdf_files_data]
            merged[i] = pd.concat(data, axis=0).sort_index()

        self.data = FGM_LABEL(*[v for _, v in merged.items()])


class FSM(CDF):
    instrument: ClassVar[Instruments] = Instruments.FPI
    rate: ClassVar[str] = "brst"
    level: ClassVar[str] = "l3"
