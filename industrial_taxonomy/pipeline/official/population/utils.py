"""Utilities to clean population estimate code"""


def clean_popest(table):
    """Cleans the population estimate data"""

    return table[["Code", "All ages"]].rename(
        columns={"Code": "geo_code", "All ages": "pop_2020"}
    )
