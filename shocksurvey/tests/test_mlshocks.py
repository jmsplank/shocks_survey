from dataclasses import asdict
from datetime import datetime as dt
from datetime import timezone
from os import remove

import pytest
from freezegun import freeze_time
from shocksurvey.mlshocks import *


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


def test_rows_to_html_params_correct_rows():
    data = load_data()
    rows = [data.iloc[i] for i in range(5)]
    params = [asdict(r) for r in rows_to_html_params(rows)]
    correct_params = [
        {
            "THBN": "71.6",
            "MA": " 6.8",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_111910.png",
            "basename": "MMS1_shocks__20151007_111910.png",
        },
        {
            "THBN": "87.6",
            "MA": " 9.0",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_113508.png",
            "basename": "MMS1_shocks__20151007_113508.png",
        },
        {
            "THBN": "89.9",
            "MA": " 9.0",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_113724.png",
            "basename": "MMS1_shocks__20151007_113724.png",
        },
        {
            "THBN": "86.1",
            "MA": " 7.9",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_114439.png",
            "basename": "MMS1_shocks__20151007_114439.png",
        },
        {
            "THBN": "70.6",
            "MA": " 5.1",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_120710.png",
            "basename": "MMS1_shocks__20151007_120710.png",
        },
    ]
    assert params == correct_params


def test_rows_to_html_params_dict():
    data = load_data()
    rows = data.iloc[:5]
    params = [asdict(r) for r in rows_to_html_params(rows)]
    correct_params = [
        {
            "THBN": "71.6",
            "MA": " 6.8",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_111910.png",
            "basename": "MMS1_shocks__20151007_111910.png",
        },
        {
            "THBN": "87.6",
            "MA": " 9.0",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_113508.png",
            "basename": "MMS1_shocks__20151007_113508.png",
        },
        {
            "THBN": "89.9",
            "MA": " 9.0",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_113724.png",
            "basename": "MMS1_shocks__20151007_113724.png",
        },
        {
            "THBN": "86.1",
            "MA": " 7.9",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_114439.png",
            "basename": "MMS1_shocks__20151007_114439.png",
        },
        {
            "THBN": "70.6",
            "MA": " 5.1",
            "filepath": "/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_120710.png",
            "basename": "MMS1_shocks__20151007_120710.png",
        },
    ]
    assert params == correct_params


@freeze_time("2020-06-15 17:26:30.0")
@pytest.mark.parametrize("timestamp_name", [False, True])
def test_generate_html(timestamp_name):
    data = load_data()
    plots = rows_to_html_params(data.iloc[:5])
    fname = generate_html(plots, timestamp_name=timestamp_name)
    with open(fname, "r") as file:
        data = file.read()
    os.remove(fname)
    correct_data = """<h1>Generated 15/06/2020 at 17:26:30.000000</h1>
<ul>
    
    <li>
        <a href="#MMS1_shocks__20151007_111910.png">MMS1_shocks__20151007_111910.png</a>: THBN: 71.6 | MA:  6.8
    </li>
    
    <li>
        <a href="#MMS1_shocks__20151007_113508.png">MMS1_shocks__20151007_113508.png</a>: THBN: 87.6 | MA:  9.0
    </li>
    
    <li>
        <a href="#MMS1_shocks__20151007_113724.png">MMS1_shocks__20151007_113724.png</a>: THBN: 89.9 | MA:  9.0
    </li>
    
    <li>
        <a href="#MMS1_shocks__20151007_114439.png">MMS1_shocks__20151007_114439.png</a>: THBN: 86.1 | MA:  7.9
    </li>
    
    <li>
        <a href="#MMS1_shocks__20151007_120710.png">MMS1_shocks__20151007_120710.png</a>: THBN: 70.6 | MA:  5.1
    </li>
    
</ul>

<h2>
    <a name="MMS1_shocks__20151007_111910.png">MMS1_shocks__20151007_111910.png</a>
</h2>
theta_bn: 71.6
<br />
mach number:  6.8
<br />
<img src="/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_111910.png" alt="MMS1_shocks__20151007_111910.png" width="1008" height="1075" />
<br />

<h2>
    <a name="MMS1_shocks__20151007_113508.png">MMS1_shocks__20151007_113508.png</a>
</h2>
theta_bn: 87.6
<br />
mach number:  9.0
<br />
<img src="/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_113508.png" alt="MMS1_shocks__20151007_113508.png" width="1008" height="1075" />
<br />

<h2>
    <a name="MMS1_shocks__20151007_113724.png">MMS1_shocks__20151007_113724.png</a>
</h2>
theta_bn: 89.9
<br />
mach number:  9.0
<br />
<img src="/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_113724.png" alt="MMS1_shocks__20151007_113724.png" width="1008" height="1075" />
<br />

<h2>
    <a name="MMS1_shocks__20151007_114439.png">MMS1_shocks__20151007_114439.png</a>
</h2>
theta_bn: 86.1
<br />
mach number:  7.9
<br />
<img src="/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_114439.png" alt="MMS1_shocks__20151007_114439.png" width="1008" height="1075" />
<br />

<h2>
    <a name="MMS1_shocks__20151007_120710.png">MMS1_shocks__20151007_120710.png</a>
</h2>
theta_bn: 70.6
<br />
mach number:  5.1
<br />
<img src="/Users/jamesplank/Documents/PHD/shocks_survey/data/ml_shocks/plots/MMS1_shocks__20151007_120710.png" alt="MMS1_shocks__20151007_120710.png" width="1008" height="1075" />
<br />
"""
    assert data == correct_data
