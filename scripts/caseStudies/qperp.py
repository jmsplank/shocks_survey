from __future__ import annotations

from datetime import datetime as dt

import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from shocksurvey import gen_timestamp, logger
from shocksurvey.constants import SC
from shocksurvey.mlshock import MLProperties
from shocksurvey.shock import Shock, Spacecraft
from shocksurvey.spedas import FGM, FPI, FSM


# @logger.catch
def main():
    override_mpl.override()

    timestamp = 1520916120.0
    trange = ["2018-03-13/04:42:00", "2018-03-13/04:57:00"]

    shock = Shock(trange=trange)
    properties = MLProperties(timestamp=timestamp)
    shock.properties = properties

    mms1 = shock.add_spacecraft(SC.MMS1)
    mms1.add_fgm(trange=properties.get_trange())
    assert mms1.fgm is not None
    assert mms1.fgm.data is not None

    mms2 = shock.add_spacecraft(SC.MMS2)
    mms2.add_fgm(properties.get_trange())
    assert mms2.fgm is not None
    assert mms2.fgm.data is not None

    fig, ax = plt.subplots()
    ax.plot(mms1.fgm.data.B.index, mms1.fgm.data.B.bt)
    ax.plot(mms2.fgm.data.B.index, mms2.fgm.data.B.bt)
    plt.show()


if __name__ == "__main__":
    main()
