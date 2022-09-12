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


class FPI_DTYPES(str, Enum):
    DIS = "dis-moms"
    DES = "des-moms"


FGM_CADENCE = 1 / 128
FPI_I_CADENCE = 0.15
FPI_E_CADENCE = 0.03
