from __future__ import annotations

from dataclasses import dataclass

from shocksurvey.constants import SC
from shocksurvey.spedas import FGM, FPI, FSM


@dataclass
class Spacecraft:
    name: SC
    timestamp: float
    fgm: FGM | None = None
    fpi: FPI | None = None
    fsm: FSM | None = None

    def add_fgm(self, trange: list[str] | None = None) -> None:
        if not trange:
            self.fgm = FGM(self.timestamp, self.name)
        else:
            self.fgm = FGM(self.timestamp, self.name)
            self.fgm.update_trange(trange)
        self.fgm.load()

    def add_fpi(self, trange: list[str] | None = None, download: bool = False) -> None:
        if not trange:
            self.fpi = FPI(self.timestamp, self.name)
        else:
            self.fpi = FPI(self.timestamp, self.name)
            self.fpi.update_trange(trange)
        if download:
            self.fpi.download()
        self.fpi.load()
