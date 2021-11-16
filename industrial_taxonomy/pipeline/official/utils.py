"""Get and extract official data
"""

import requests
import pandas as pd


def get(url: str) -> bytes:
    """get ONS table in URL"""
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def excel_to_df(content: bytes, sheets: int, skiprows: int):
    """Parse excel to dataframe
    Arguments:
        sheets: sheet number to keep
        skipwrows: rows to skip
    """
    return pd.read_excel(content, skiprows=skiprows, sheet_name=sheets)
