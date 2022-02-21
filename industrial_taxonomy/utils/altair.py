# Scripts to save altair charts

import os
from altair import Chart
from altair_saver import save
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

    """

    ch = chart.configure_axis(
        labelFontSize=sizes[0], titleFontSize=sizes[1]
    ).configure_legend(labelFontSize=sizes[0], titleFontSize=sizes[1])
    return ch
