"""Clean and process GDP data"""

import pandas as pd
import re


def process_gdp_table(table: pd.DataFrame):
    """Cleans up the gdp table"""
    _table = (
        table.dropna(
            axis=0, subset=["LA code"]  # We are dropping bottom rows without a LA code
        )
        .rename(columns={"2019\n[note 3]": "2019"})
        .rename(columns={x: str(x) for x in range(1998, 2020)})
    )

    _table.columns = [re.sub(" ", "_", col.lower()) for col in _table.columns]

    return _table
