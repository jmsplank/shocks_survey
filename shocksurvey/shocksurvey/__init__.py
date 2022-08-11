from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from sys import stderr
from typing import Any, Literal, Optional

import dotenv
import numpy as np
from loguru import logger
from pyspedas.mms.mms_config import CONFIG

config_path = "config.env"
dotenv.load_dotenv(config_path)


@dataclass
class Env:
    ML_DATA_LOCATION: str
    HTML_SAVE_DIR: str
    MMS_DATA_DIR: str
    BASENAME: str


try:
    env_dict = dotenv.dotenv_values(config_path)
    env_dict = {k: (v if v is not None else "") for k, v in env_dict.items()}
    ENV = Env(**env_dict)
except TypeError as e:
    err = re.findall(r"'(.*?)'", str(e))[0]  # Find name of parameter
    raise NameError(f"Unable to match environment variable {err} to a parameter in class Env.\nMake sure to add the corresponding property to the Env class in shocksurvey/__init__.")  # fmt: skip

logger.remove(0)
logger.add(
    stderr,
    format="<red>[{level}]</red> | {module}/{function} | Msg: <green>{message}</green> @ {time}",
    colorize=True,
)
logger.add(
    Path(ENV.BASENAME) / "logs/log.log",
    rotation="50 MB",
    format="<red>[{level}]</red> | {module}/{function} | Msg: <green>{message}</green> @ {time}",
)

CONFIG["local_data_dir"] = ENV.MMS_DATA_DIR


def ts_as_dt(timestamp: float) -> dt:
    return dt.utcfromtimestamp(timestamp)


def gen_timestamp(
    time: Optional[dt] = None,
    output_type: Literal["functional", "files", "pretty", "pyspedas"] = "functional",
) -> str:
    """Convenience fn to generate timestamps in common formats.
    IN:
        time:           datetime object to format, default is now
        output_type:    common types of formatting for different uses
    OUT:
        formatted time
    """
    if not time:
        time = dt.now()

    if output_type == "pretty":
        return f"{time:%d/%m/%Y %H:%M:%S}"
    elif output_type == "files":
        return f"{time:%Y%m%d%H%M%S}"
    elif output_type == "pyspedas":
        return f"{time:%Y-%m-%d/%H:%M:%S}"

    return f"{time:%Y%m%d%H%M%S}"


def bprint(var: Any) -> None:
    caller_local_vars = inspect.currentframe()
    assert caller_local_vars is not None
    assert caller_local_vars.f_back is not None
    caller_local_vars = caller_local_vars.f_back.f_locals.items()
    caller_var_name = [
        (var_name, var_val) for var_name, var_val in caller_local_vars if var_val is var
    ][0]
    print(f"{' '+caller_var_name[0]+' ':-^80}", caller_var_name[1], "", sep="\n")


def gen_filepath(sub_folder: str | Path, name: str) -> Path:
    base = Path(ENV.BASENAME)
    folder = base / "data" / sub_folder
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def save_arr_to_data(sub_folder: str | Path, name: str, arr: np.ndarray) -> None:
    fname = gen_filepath(sub_folder, name)
    np.save(fname, arr)


def load_arr_from_data(sub_folder: str | Path, name: str) -> np.ndarray:
    fname = gen_filepath(sub_folder, name)
    data = np.load(fname)
    return data
