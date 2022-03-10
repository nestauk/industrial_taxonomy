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
from webdriver_manager.chrome import ChromeDriverManager
import industrial_taxonomy

LOOSE_UK_BOUNDS = {
    "long": (-7.6, 1.7),
    "lat": (50.0, 58.7),
}  # Rough bounds of UK, ok for DQ check


def find_download_url(driver: WebDriver, geo_portal_url: str) -> str:
    """Find download button and extract download URL."""
    driver.implicitly_wait(10)  # Poll DOM for up to 10 seconds when finding elements
    driver.get(geo_portal_url)
    return driver.find_element_by_link_text("Download").get_attribute("href")


def chrome_driver() -> WebDriver:
    """Headless Selenium Chrome Driver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(
        executable_path=ChromeDriverManager().install(),
        options=chrome_options,
    )
    driver.set_page_load_timeout(10)
    driver.set_script_timeout(10)
    return driver


def filter_nspl_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter out NSPL rows without location data."""
    keep_rows = (
        data.index.notna()
        & data.lat.between(*LOOSE_UK_BOUNDS["lat"])
        & data.long.between(*LOOSE_UK_BOUNDS["long"])
    )  # Null lat-longs are actually coded as (99,0)

    return data.loc[keep_rows]


def download_zip(url: str) -> ZipFile:
    """Download a URL and load into `ZipFile`."""
    response = requests.get(url)
    response.raise_for_status()
    return ZipFile(BytesIO(response.content), "r")


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

    # Settiing index to laua to avoid pcds being lost on merge.
    return nspl_data.reset_index().set_index("laua", drop=True)


def read_laua_names(
    zipfile: ZipFile,
) -> pd.DataFrame:
    nspl_zip_path = get_laua_csv_zip_path(zipfile)
    data = pd.read_csv(
        zipfile.open(nspl_zip_path),
        # Ignore Welsh name column
        usecols=lambda name: not name.endswith("W"),
    )

    code_column_name = data.columns[data.columns.str.contains("CD")][0]

    return data


def nspl_joined(
    nspl_data_df: pd.DataFrame, laua_names_df: pd.DataFrame
) -> pd.DataFrame:
    """joining nspl_names with nspl_data on corresponding laua_code"""

    joined_df = pd.merge(
        nspl_data_df, laua_names_df, how="left", left_on="laua", right_on="LAD20CD"
    )

    joined_df.rename(
        columns={"LAD20CD": "laua_code", "LAD20NM": "laua_name"}, inplace=True
    )

    # setting pcds to index so pcds can later be set to keys when the joined_dataset is converted to dict.
    return joined_df.set_index("pcds", drop=True)


def get_nspl_csv_zip_path(zipfile: ZipFile) -> str:
    """Get the path within the zip folder to the NSPL CSV lookup."""
    return next(
        filter(
            lambda name: re.match(r"Data/NSPL_[A-Z]{3}_[0-9]{4}_UK.csv", name),
            zipfile.namelist(),
        )
    )


def get_laua_csv_zip_path(zipfile: ZipFile) -> str:
    """Get the path within the zip folder to the LAUA CSV lookup."""
    return next(
        filter(
            lambda name: re.match(r"Documents/LA_UA.*.csv", name),
            zipfile.namelist(),
        )
    )
