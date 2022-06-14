import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from phdhelper.helpers import override_mpl
from shocksurvey.mlshocks import DC, load_data, open_shock_plot, remove_non_burst

override_mpl.override()

print("Loading data")
data = load_data()
print(f"Data loaded:                    len {len(data)}")
data = remove_non_burst(data)
print(f"Non burst intervals removed:    len {len(data)}")

data = data[data.thBn != -1e30]
print(f"Missing theta bn removed:       len {len(data)}")
data = data[data.MA != -1e30]
print(f"Missing mach number removed:    len {len(data)}")


data["ssthBnMA"] = np.sqrt(data.sthBn**2 + data.sMA**2)  # Combined error
# sortedData = data.sort_values("ssthBnMA")

# plt.hist(data.ssthBnMA, bins=100)

data = data[data.ssthBnMA <= 5]
print(f"Combined error > 5 removed:     len {len(data)}")


fig = plt.figure()
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, polar=True)

# ax.set_thetamin(0)
# ax.set_thetamax(90)

ax.scatter(data.thBn, data.MA, c=data.ssthBnMA)

for i in range(5):
    i *= 90 / 5
    subData = data[(data.thBn >= i) & (data.thBn < i + 90 / 5)]
    row = subData.iloc[np.argmin(subData.ssthBnMA)]
    open_shock_plot(row.time)
    ax.scatter(row.thBn, row.MA, c="k", s=100, marker="x")
    print(row.name)
plt.show()
