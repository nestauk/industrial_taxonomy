"""Get and extract official data
"""

import requests
import pandas as pd


def get(url: str) -> bytes:
    """get ONS table in URL"""
    response = requests.get(url)
    response.raise_for_status()
    return response.content
