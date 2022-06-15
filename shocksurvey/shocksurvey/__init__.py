import re
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Literal, Optional

import dotenv

config_path = "config.env"
dotenv.load_dotenv(config_path)


@dataclass
class Env:
    ML_DATA_LOCATION: str
    HTML_SAVE_DIR: str


try:
    ENV = Env(**dotenv.dotenv_values(config_path))
except TypeError as e:
    err = re.findall(r"'(.*?)'", str(e))[0]  # Find name of parameter
    raise NameError(f"Unable to match environment variable {err} to a parameter in class Env.\nMake sure to add the corresponding property to the Env class in shocksurvey/__init__.")  # fmt: skip


def gen_timestamp(
    time: Optional[dt] = None,
    output_type: Literal["functional", "files", "pretty"] = "functional",
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

    if output_type is "pretty":
        return f"{time:%d/%m/%Y %H:%M:%S}"
    elif output_type is "files":
        return f"{time:%Y-%m-%dT%H-%M-%S}"

    return f"{time:%Y%m%d%H%M%S}"
