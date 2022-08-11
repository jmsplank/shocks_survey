from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime as dt
from enum import Enum
from tkinter import W
from typing import NamedTuple

from shocksurvey.constants import SC
from shocksurvey.sc import Spacecraft


class ShockProperties(ABC):
    pass


class Shock:
    def __init__(
        self,
        trange: list[str],
        properties: ShockProperties | None = None,
        sc: dict[SC, Spacecraft] = {},
    ) -> None:
        self.trange = trange
        self.properties = properties
        self.sc = sc

    def trange_as_dt(self) -> list[dt]:
        def f(s: str) -> dt:
            return dt.strptime(s, "%Y-%m-%d/%H:%M:%S")

        return list(map(f, self.trange))

    @property
    def timestamp(self):
        return dt.timestamp(self.trange_as_dt()[0])

    def add_spacecraft(self, name: SC) -> Spacecraft:
        if name in self.sc:
            raise NameError(f"Name {name} not a valid spacecraft name.")

        spacecraft = Spacecraft(name=name, timestamp=self.timestamp)
        self.sc[name] = spacecraft
        return spacecraft
