from os import times
import pytest
from shocksurvey.mlshocks import *
from datetime import datetime as dt
from datetime import timezone


def test_load_data_no_path():
    assert len(load_data()) == 2797


def test_load_data_bad_path():
    with pytest.raises(FileNotFoundError):
        load_data("/bad/path")


def test_load_data_all_columns_present():
    assert set(load_data().columns) == set(DC.all_params(DC))


def test_get_plot_file_correct_timestamp():
    timestamp = dt(2015, 10, 7, 13, 46, 33)
    timestamp = timestamp.replace(tzinfo=timezone.utc).timestamp()
    assert get_plot_file(timestamp) == join(
        dirname(ENV.ML_DATA_LOCATION), "plots", "MMS1_shocks__20151007_134633.png"
    )


def test_get_plot_file_incorrect_timestamp():
    timestamp = dt(2022, 6, 14, 16, 00, 00)
    timestamp = timestamp.replace(tzinfo=timezone.utc).timestamp()
    with pytest.raises(FileNotFoundError):
        get_plot_file(timestamp, full_path=True)
