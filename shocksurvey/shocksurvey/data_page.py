from jinja2 import Environment, FileSystemLoader
import os
from dataclasses import dataclass
from shocksurvey.mlshocks import DC, get_plot_file
from datetime import datetime as dt


def rows_to_html_params(rows):
    plots = []

    @dataclass
    class Plot:
        THBN: float
        MA: float
        filepath: str
        basename: str

    for row in rows:
        filepath = get_plot_file(row[DC.TIME])
        plots.append(
            Plot(
                THBN=f"{row[DC.THBN]:>4.1f}",
                MA=f"{row[DC.MA]:>4.1f}",
                filepath=filepath,
                basename=os.path.basename(filepath),
            )
        )
    return plots


def generate_html(plots):
    root = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(root, "data_page")
    env = Environment(loader=FileSystemLoader(dir))
    template = env.get_template("template_data_page.jinja2")

    filename = os.path.join(dir, "data_page.html")
    with open(filename, "w") as file:
        file.write(
            template.render(
                h1=f"Generated {dt.now():%d/%m/%Y} at {dt.now():%H:%M:%S.%f}",
                plots=plots,
            )
        )
    return filename


def multi_plot_html(rows):
    plots = rows_to_html_params(rows)
    html = generate_html(plots)
    os.system(f"open {html}")
