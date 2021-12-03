"""Clean and process GDP data"""

import pandas as pd

YEAR_RANGE = range(1998, 2020)


def process_gdp_table(table: pd.DataFrame):
    """Removes table footnotes and renames columns"""
    _table = (
        table.dropna(
            axis=0, subset=["LA code"]  # We are dropping bottom rows without a LA code
        )
        .rename(columns={"2019\n[note 3]": "2019"})
        .rename(columns={x: str(x) for x in YEAR_RANGE})
        .rename(columns=lambda s: s.lower().replace(" ", "_"))
    )

    return _table
