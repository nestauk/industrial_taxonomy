"""Extracting SIC code structure lookups from Excel."""
import pandas as pd
import requests
from toolz import interleave

LEVELS = ["section", "division", "group", "class", "subclass"]


def get(url: str) -> bytes:
    """Get SIC 2007 structure from ONS hosted excel file."""
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def excel_to_df(content: bytes) -> pd.DataFrame:
    """Parse Excel to Dataframe."""
    return (
        pd.read_excel(
            content,
            skiprows=1,
            names=interleave([LEVELS, map(lambda x: f"{x}_name", LEVELS)]),
            dtype=str,
        )
        .dropna(how="all")
        .apply(lambda column: column.str.strip(), axis=1)
    )


def fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing information in spreadsheet."""
    return (
        df.pipe(_generic_fill, "section")
        .pipe(_generic_fill, "division")
        .pipe(_generic_fill, "group")
        .pipe(_subclass_fill)
    )


def companies_house_extras() -> pd.DataFrame:
    """Add Companies House specific SIC codes."""
    extras = {
        "subclass": ["74990", "98000", "99999"],
        "subclass_name": [
            "Non-trading company",
            "Residents property management",
            "Dormant company",
        ],
    }

    return pd.DataFrame(extras)


def normalise_codes(df):
    """Remove dots and slashes from SIC digits."""
    df.loc[:, LEVELS] = df.loc[:, LEVELS].apply(
        lambda col: col.str.replace("[./]", "", regex=True)
    )
    return df


def _subclass_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve."""
    # Forward fill classes
    df.loc[:, ("class", "class_name")] = df.loc[:, ("class", "class_name")].ffill()

    # Backfill subclass by one
    # (each bfill eventually yields a duplicate row which we can drop with
    # `drop_duplicates` rather than tedious index accounting)
    df.loc[:, ("subclass", "subclass_name")] = df.loc[
        :, ("subclass", "subclass_name")
    ].bfill(limit=1)

    # If no subclass, derive from class by adding a zero, and using class name
    idx = df["subclass"].isna()
    df.loc[idx] = df.loc[idx].assign(
        subclass=lambda x: x["class"] + "/0", subclass_name=lambda x: x.class_name
    )

    return df.drop_duplicates()  # Drops dups we induced with bfill


def _generic_fill(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ffill columns relating to `col`, dropping rows that were originally not NaN."""
    cols = [col, f"{col}_name"]
    subdf = (
        df[cols].copy()
        # Drop rows that only have either code or name - they correspond to notes
        .loc[lambda x: x[cols].isna().sum(1) != 1]
    )
    label_idx = subdf[cols].notna().sum(1).astype(bool)
    return subdf.ffill().loc[~label_idx].join(df.drop(cols, 1))
