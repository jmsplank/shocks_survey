import re
from dataclasses import dataclass

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
