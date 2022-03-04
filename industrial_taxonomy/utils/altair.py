# Scripts to save altair charts

import json
import os
import altair as alt
import pandas as pd
from altair import Chart
from altair_saver import save
from geopandas import GeoDataFrame
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.webdriver import WebDriver

from industrial_taxonomy import PROJECT_DIR


FIG_PATH = f"{PROJECT_DIR}/outputs/figures"

# Checks if the right paths exist and if not creates them when imported
os.makedirs(f"{FIG_PATH}/png", exist_ok=True)
os.makedirs(f"{FIG_PATH}/html", exist_ok=True)


def google_chrome_driver_setup() -> WebDriver:
    # Set up the driver to save figures as png
    driver = webdriver.Chrome(ChromeDriverManager().install())
    return driver


def save_altair(fig: Chart, name: str, driver: WebDriver, path=FIG_PATH) -> None:
    """Saves an altair figure as png and html

    Args:
        fig: altair chart
        name: name to save the figure
        driver: webdriver
        path: path to save the figure
    """
    save(
        fig,
        f"{path}/png/{name}.png",
        method="selenium",
        webdriver=driver,
        scale_factor=5,
    )
    fig.save(f"{path}/html/{name}.html")


def altair_text_resize(chart: Chart, sizes: tuple = (12, 14)) -> Chart:
    """Resizes the text of axis labels and legends in an altair chart

    Args:
        chart: chart to resize
        sizes: label size and title size

    Returns:
        An altair chart
    """

    ch = chart.configure_axis(
        labelFontSize=sizes[0], titleFontSize=sizes[1], labelLimit=300
    ).configure_legend(labelFontSize=sizes[0], titleFontSize=sizes[1])
    return ch


def make_save_path(path: str):
    """Make save paths in case we are not using
    the standard one
    """

    os.makedirs(f"{path}/png", exist_ok=True)
    os.makedirs(f"{path}/html", exist_ok=True)


def choro_plot(
    geo_df: GeoDataFrame,
    count_var: str,
    count_var_name: str,
    region_name: str = "region",
    scheme: str = "spectral",
    scale_type: str = "linear",
) -> Chart:
    """Plot an altair choropleth

    Args:
        geo_df: geodf
        count_var: name of the variable we are plotting
        count_var_name: clean name for the count variable
        region_name is the name of the region variable
        scheme: is the colour scheme. Defaults to spectral
        scale_type: is the type of scale we are using. Defaults to log

    Returns:
        An altair chart

    """

    base_map = (  # Base chart with outlines
        alt.Chart(alt.Data(values=json.loads(geo_df.to_json())["features"]))
        .project(type="mercator")
        .mark_geoshape(filled=False, stroke="gray")
    )

    choropleth = (  # Filled polygons and tooltip
        base_map.transform_calculate(region=f"datum.properties.{region_name}")
        .mark_geoshape(filled=True, stroke="darkgrey", strokeWidth=0.2)
        .encode(
            size=f"properties.{count_var}:N",
            color=alt.Color(
                f"properties.{count_var}:N",
                title=count_var_name,
                scale=alt.Scale(scheme=scheme, type=scale_type),
                sort="descending",
            ),
            tooltip=[
                "region:N",
                alt.Tooltip(f"properties.{count_var}:Q", format="1.2f"),
            ],
        )
    )

    return base_map + choropleth


def lad_nuts1_lookup(year: int = 2019) -> dict:
    """Read a lookup between local authorities and NUTS"""

    if year == 2019:
        lu_df = pd.read_csv(
            "https://opendata.arcgis.com/datasets/3ba3daf9278f47daba0f561889c3521a_0.csv"
        )
        return lu_df.set_index("LAD19CD")["RGN19NM"].to_dict()
    else:
        lu_df = pd.read_csv(
            "https://opendata.arcgis.com/datasets/054349b09c094df2a97f8ddbd169c7a7_0.csv"
        )
        return lu_df.set_index("LAD20CD")["RGN20NM"].to_dict()
