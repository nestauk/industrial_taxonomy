"""Data getters for Companies House data."""
from functools import lru_cache
from typing import Optional

from pandas import DataFrame, Series, to_datetime
from metaflow import Run
from metaflow.client.core import MetaflowData

from industrial_taxonomy import config
from industrial_taxonomy.utils.metaflow import namespace_context

if config is None:
    raise FileNotFoundError("Could not find config file.")
RUN_ID = config["flows"]["companies_house"]["run_id"]


@lru_cache()
def _get_run() -> Run:
    run = Run(f"CompaniesHouseMergeDumpFlow/{RUN_ID}")
    if not run.successful:
        raise ValueError(f"{run} was not successful")
    return run


def _get_run_data() -> MetaflowData:
    """Get data artifacts of canonical Companies House run."""
    # Run is outside of 'project:industrial_taxonomy' namespace therefore the
    # namespace must be switched temporarily to get the data artifacts of this
    # run. `run.data` must be access in the namespace context otherwise
    # metaflow raises a namespace error!
    with namespace_context(None):
        return _get_run().data


def get_organisation(run: Optional[Run] = None) -> DataFrame:
    """Get primary organisation data for each Company.

    Args:
        run: Run of `CompaniesHouseMergeDumpFlow`

    Returns:
        Index:
            Company Number
        Columns:
            Name: category, dtype: str, Company category, e.g. PLC
            Name: status, dtype: str, Company status, e.g. Active, Liquidation etc.
            Name: country_of_origin, dtype: str, Country of Origin
            Name: dissolution_date, dtype: datetime64, Date (if any) company dissolved
            Name: incorporation_date, dtype: datetime64, Date company incorporated
    """
    run_data = run.data if run else _get_run_data()
    return (
        run_data.organisation.astype(
            {"company_number": str, "status": str, "category": str}
        )
        .assign(
            dissolution_date=lambda df: df.dissolution_date.pipe(_parse_date),
            incorporation_date=lambda df: df.incorporation_date.pipe(_parse_date),
        )
        .drop("uri", axis=1)
        .set_index("company_number")
    )


def get_sector(run: Optional[Run] = None) -> DataFrame:
    """Get sector rankings for each Company.

    Args:
        run: Run of `CompaniesHouseMergeDumpFlow`

    Returns:
        Index:
            Company Number
        Columns:
            Name: SIC5_code, dtype: str, 5-digit SIC code
            Name: rank, dtype: int, Rank of sector (lower number => more likely)
    """
    run_data = run.data if run else _get_run_data()
    return (
        run_data.organisationsector.sort_values("date")
        .drop_duplicates(["company_number", "rank"], keep="last")
        .rename(columns={"sector_id": "SIC5_code"})
        .drop("date", 1)
        .astype({"company_number": str, "rank": int, "SIC5_code": int})
        .set_index("company_number")
    )


def get_name(run: Optional[Run] = None) -> DataFrame:
    """Get name history of Companies.

    Args:
        run: Run of `CompaniesHouseMergeDumpFlow`

    Returns:
        Index:
            Company Number
        Columns:
            Name: name_age_index, dtype: int, How many renames ago name was current
            Name: name, dtype: str, Company Name
            Name: invalid_date, dtype: datetime64, When name changed (NaT => current)
    """
    run_data = run.data if run else _get_run_data()
    return (
        run_data.organisationname.assign(
            invalid_date=lambda df: _parse_date(df.invalid_date)
        )
        .astype({"company_number": str, "name_age_index": int, "name": "str"})
        .set_index("company_number")
    )


def _parse_date(series: Series) -> Series:
    return to_datetime(series, errors="coerce", dayfirst=True)
