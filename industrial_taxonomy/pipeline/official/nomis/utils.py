"""Utilities to fetch Nomis data"""

import pandas as pd


def process_nomis(
    df: pd.DataFrame,
    indicator_name: str,
    value_column: str,
    source: str,
    indicator_column: str = "MEASURES_NAME",
):
    """Process nomis data

    Arguments:
        df: nomis table
        indicator_name: name of indicator
        value_column: value column
        source: data source
        indicator_column: column that contains the indicator

    Returns:
        A clean table with secondary data
    """
    return (
        df.query(f"{indicator_column}=='{indicator_name}'")[
            ["DATE", "GEOGRAPHY_NAME", "GEOGRAPHY_CODE", value_column, "OBS_VALUE"]
        ]
        .reset_index(drop=True)
        .rename(columns={"OBS_VALUE": "VALUE", value_column: "VARIABLE"})
        .assign(source=source)
        .rename(columns=str.lower)
        .assign(
            date=lambda df: [  # We parse APS dates returned in format "y-m"
                d if type(d) == int else int(d.split("-")[0]) for d in df["date"]
            ]
        )
    )
