"""Helper functions for NSPL flow."""
import re
from io import BytesIO
from typing import Optional
from zipfile import ZipFile

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
import industrial_taxonomy

LOOSE_UK_BOUNDS = {
    "long": (-7.6, 1.7),
    "lat": (50.0, 58.7),
}  # Rough bounds of UK, ok for DQ check
#


def find_download_url(driver: WebDriver, geo_portal_url: str) -> str:
    """Find download button and extract download URL."""
    driver.implicitly_wait(10)  # Poll DOM for up to 10 seconds when finding elements
    driver.get(geo_portal_url)
    return driver.find_element_by_link_text("Download").get_attribute("href")


#
def chrome_driver() -> WebDriver:
    """Headless Selenium Chrome Driver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(
        executable_path="/Users/imanmuse/miniconda3/chromedriver",
        options=chrome_options,
    )
    driver.set_page_load_timeout(10)
    driver.set_script_timeout(10)
    return driver


#


def filter_nspl_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter out NSPL rows without location data."""
    keep_rows = (
        data.laua_code.notna()
        & data.lat.between(*LOOSE_UK_BOUNDS["lat"])
        & data.long.between(*LOOSE_UK_BOUNDS["long"])
    )  # Null lat-longs are actually coded as (99,0)

    return data.loc[keep_rows]


def download_zip(url: str) -> ZipFile:
    """Download a URL and load into `ZipFile`."""
    response = requests.get(url)
    response.raise_for_status()
    return ZipFile(BytesIO(response.content), "r")


#
def read_nspl_data(
    zipfile: ZipFile,
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    nspl_zip_path = get_nspl_csv_zip_path(zipfile)

    usecols = {
        "pcds",  # Postcode
        "laua",  # Local Authority
        "lat",
        "long",
    }

    nspl_data = pd.read_csv(
        zipfile.open(nspl_zip_path),
        nrows=nrows,
        usecols=usecols,
        index_col="pcds",
    )

    nspl_data = nspl_data.reset_index()
    nspl_data.index = nspl_data["laua"]
    nspl_data.rename(columns={"laua": "laua_code"}, inplace=True)
    return nspl_data


#


def nspl_joined(data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:

    return pd.merge(data1, data2, how="left", on="laua")


def resetting_index(data1: pd.DataFrame) -> pd.DataFrame:

    return data1.set_index("pcds")


def read_laua_names(
    zipfile: ZipFile,
) -> pd.DataFrame:
    nspl_zip_path = get_laua_csv_zip_path(zipfile)
    data = pd.read_csv(
        zipfile.open(nspl_zip_path),
        # Ignore Welsh name column
        usecols=lambda name: not name.endswith("W"),
    )
    # Get code (CD) column name, e.g. LAD20CD.
    # Done dynamically because LAD year may change
    code_column_name = data.columns[data.columns.str.contains("CD")][0]
    data.rename(columns={"LAD20CD": "laua", "LAD20NM": "laua_name"}, inplace=True)
    data = data.set_index("laua")

    return data


#
def get_nspl_csv_zip_path(zipfile: ZipFile) -> str:
    """Get the path within the zip folder to the NSPL CSV lookup."""
    return next(
        filter(
            lambda name: re.match(r"Data/NSPL_[A-Z]{3}_[0-9]{4}_UK.csv", name),
            zipfile.namelist(),
        )
    )


#
def get_laua_csv_zip_path(zipfile: ZipFile) -> str:
    """Get the path within the zip folder to the LAUA CSV lookup."""
    return next(
        filter(
            lambda name: re.match(r"Documents/LA_UA.*.csv", name),
            zipfile.namelist(),
        )
    )
