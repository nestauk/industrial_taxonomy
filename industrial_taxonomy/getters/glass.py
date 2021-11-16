"""Data getters for Glass AI data."""
from functools import lru_cache
from typing import Optional

import pandas as pd
from metaflow import Run
from metaflow.client.core import MetaflowData

import industrial_taxonomy
from industrial_taxonomy.utils.metaflow import namespace_context


RUN_ID = industrial_taxonomy.config["flows"]["glass"]["run_id"]


@lru_cache()
def get_run_data() -> MetaflowData:
    """Get data artifacts of canonical Glass AI run."""
    # Run is outside of 'project:industrial_taxonomy' namespace therefore the
    # namespace must be switched temporarily to get the data artifacts of this
    # run. `run.data` must be access in the namespace context otherwise
    # metaflow raises a namespace error!
    with namespace_context(None):
        run = Run(f"GlassMergeMainDumpFlow/{RUN_ID}")
        if not run.successful:
            raise ValueError(f"{run} was not successful")
        return run.data


def get_organisation(run: Optional[Run] = None) -> pd.DataFrame:
    """Glass organisations."""
    run_data = run.data if run else get_run_data()
    return run_data.organisation


def get_address(run: Optional[Run] = None) -> pd.DataFrame:
    """Address information extracted from Glass websites (longitudinal)."""
    run_data = run.data if run else get_run_data()
    address = run_data.address
    organisation_address = run_data.organisationaddress
    return organisation_address.merge(address, on="address_id").drop("address_id", 1)


def get_sector(run: Optional[Run] = None) -> pd.DataFrame:
    """Sector (LinkedIn taxonomy) information for Glass Businesses (longitudinal)."""
    run_data = run.data if run else get_run_data()
    sector = run_data.sector
    organisation_sector = run_data.organisationsector
    return organisation_sector.merge(sector, on="sector_id").drop("sector_id", 1)


def get_organisation_description(run: Optional[Run] = None) -> pd.DataFrame:
    """Description of business activities for Glass businesses (longitudinal)."""
    run_data = run.data if run else get_run_data()
    return run_data.organisationdescription


def get_organisation_metadata(run: Optional[Run] = None) -> pd.DataFrame:
    """Metadata for Glass businesses (longitudinal)."""
    run_data = run.data if run else get_run_data()
    return run_data.organisationmetadata
