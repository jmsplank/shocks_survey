import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from phdhelper.helpers import override_mpl
from shocksurvey.mlshocks import DC, filter_data, load_data, multi_plot_html

override_mpl.override()

data = load_data()
data = filter_data(data, missing_data=[DC.THBN, DC.MA])


data["ssthBnMA"] = np.sqrt(data.sthBn**2 + data.sMA**2)  # Combined error
# sortedData = data.sort_values("ssthBnMA")

# plt.hist(data.ssthBnMA, bins=100)

data = data[data.ssthBnMA <= 5]
print(f"Combined error > 5 removed:     len {len(data)}")


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(data.thBn, data.MA, c=data.ssthBnMA)

rows = []
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
        rows.append(row)
        # ax.scatter(row.thBn, row.MA, c="k", s=100, marker="x")
        print(row.name)
# plt.show()

multi_plot_html(rows)
