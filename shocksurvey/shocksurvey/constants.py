from enum import Enum, auto


class Instruments(str, Enum):
    FGM = "fgm"  # Fluxgate magnetometer
    FPI = "fpi"  # Fast Plasma Instrument
    FSM = "fsm"  # Fluxgate-Searchcoil Merged dataset


class SC(str, Enum):
    MMS1 = "mms1"
    MMS2 = "mms2"
    MMS3 = "mms3"
    MMS4 = "mms4"
