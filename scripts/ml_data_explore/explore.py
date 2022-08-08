from __future__ import annotations

from dataclasses import InitVar
from dataclasses import asdict
from dataclasses import asdict as dataclassdict
from pprint import pprint
from typing import Any, List

import numpy as np
import pandas as pd
from phdhelper.helpers import override_mpl
from shocksurvey.mlshocks import SHK, filter_data, load_data
from shocksurvey.spedas import Shock, download_data_from_timestamp

override_mpl.override()


data = load_data()

data = filter_data(data, missing_data=[SHK.THBN, SHK.MA])
data["ssthBnMA"] = np.sqrt(data.sthBn**2 + data.sMA**2)  # Combined error
data = data[data.ssthBnMA <= 5]
print(f"Combined error > 5 removed:     len {len(data)}")

rows: List[Shock] = []
for i in range(5):
    i *= 90 / 5
    for j in range(2):
        if j == 0:
            subData = data[(data.thBn >= i) & (data.thBn < i + 90 / 5) & (data.MA < 10)]
        else:
            subData = data[
                (data.thBn >= i) & (data.thBn < i + 90 / 5) & (data.MA >= 10)
            ]
        row = subData.iloc[np.argmin(subData.ssthBnMA)]
        fpath = download_data_from_timestamp(row[SHK.TIME], no_download=True)
        shock = Shock(
            cdf_path=fpath,
            properties=row,
        )
        # shock.load_data_from_cdf()
        rows.append(shock)
        print(shock)
        break
    break
