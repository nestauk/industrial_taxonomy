"""Data getters for Glass AI data."""
from functools import lru_cache
from typing import Optional

from pandas import DataFrame
from metaflow import Run
from metaflow.client.core import MetaflowData

from industrial_taxonomy import config
from industrial_taxonomy.utils.metaflow import namespace_context


if config is None:
    raise FileNotFoundError("Could not find config file.")
RUN_ID = config["flows"]["glass"]["run_id"]  # type: ignore


@lru_cache()
def _get_run() -> Run:
    run = Run(f"GlassMergeMainDumpFlow/{RUN_ID}")
    if not run.successful:
        raise ValueError(f"{run} was not successful")
    return run


def _get_run_data() -> MetaflowData:
    """Get data artifacts of canonical Glass AI run."""
    # Run is outside of 'project:industrial_taxonomy' namespace therefore the
    # namespace must be switched temporarily to get the data artifacts of this
    # run. `run.data` must be access in the namespace context otherwise
    # metaflow raises a namespace error!
    with namespace_context(None):
        return _get_run().data


def get_organisation(run: Optional[Run] = None) -> DataFrame:
    """Glass organisations

    Args:
        run: Run of `GlassMergeMainDumpFlow`

    Returns:
        Index:
            Glass organisation ID
        Columns:
            Name: name, dtype: str, Organisation name auto-extracted from `website`
            Name: website, dtype: str, Website URL (raw and unparsed)
    """
    run_data = run.data if run else _get_run_data()
    return (
        run_data.organisation.astype({"name": str, "website": str})
        .drop("active", axis=1)
        .set_index("org_id")
    )


def get_address(run: Optional[Run] = None) -> DataFrame:
    """Address information extracted from Glass websites (longitudinal).

    Args:
        run: Run of `GlassMergeMainDumpFlow`

    Returns:
        Index:
            Glass organisation ID
        Columns:
            Name: address_text, dtype: str, Full address text (inc. possible postcode)
            Name: postcode, dtype: str, Postcode
            Name: rank, dtype: int, Order address read from website during scraping
    """
    run_data = run.data if run else _get_run_data()
    address = run_data.address
    organisation_address = run_data.organisationaddress

    return (
        organisation_address.merge(address, on="address_id")
        .drop("address_id", 1)
        .astype({"address_text": str, "postcode": str})
        .pipe(_deduplicate_by_date_and_other, "rank")
        .drop("date", axis=1)
        .set_index("org_id")
    )


def get_sector(run: Optional[Run] = None) -> DataFrame:
    """Sector (LinkedIn taxonomy) information for Glass Businesses (longitudinal).

    Args:
        run: Run of `GlassMergeMainDumpFlow`

    Returns:
        Index:
            Glass organisation ID
        Columns:
            Name: sector_name, dtype: str, LinkedIn taxonomy sector name (camel-cased)
            Name: rank, dtype: int, Black-box rank from Glass (lower => more likely)

    """
    run_data = run.data if run else _get_run_data()
    sector = run_data.sector
    organisation_sector = run_data.organisationsector
    return (
        organisation_sector.merge(sector, on="sector_id")
        .drop("sector_id", 1)
        .astype({"sector_name": str})
        .pipe(_deduplicate_by_date_and_other, "rank")
        .drop("date", axis=1)
        .set_index("org_id")
    )


def get_organisation_description(run: Optional[Run] = None) -> DataFrame:
    """Latest description of business activities for Glass businesses.

    Args:
        run: Run of `GlassMergeMainDumpFlow`

    Returns:
        Index:
            Glass organisation ID
        Columns:
            Name: description, dtype: str, Company description extracted by Glass
    """
    run_data = run.data if run else _get_run_data()

    return (
        run_data.organisationdescription.pipe(_deduplicate_by_date_and_other, "org_id")
        .drop("date", axis=1)
        .set_index("org_id")
    )


def _deduplicate_by_date_and_other(df: DataFrame, other: str) -> DataFrame:
    return df.sort_values("date", ascending=False).drop_duplicates(
        subset=("org_id", other), keep="first"
    )
