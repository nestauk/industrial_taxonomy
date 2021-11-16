"""Data getters for Companies House data."""
from functools import lru_cache
from typing import Optional

import pandas as pd
from metaflow import Run
from metaflow.client.core import MetaflowData

import industrial_taxonomy
from industrial_taxonomy.utils.metaflow import namespace_context


RUN_ID = industrial_taxonomy.config["flows"]["companies_house"]["run_id"]


@lru_cache()
def get_run_data() -> MetaflowData:
    """Get data artifacts of canonical Companies House run."""
    # Run is outside of 'project:industrial_taxonomy' namespace therefore the
    # namespace must be switched temporarily to get the data artifacts of this
    # run. `run.data` must be access in the namespace context otherwise
    # metaflow raises a namespace error!
    with namespace_context(None):
        run = Run(f"CompaniesHouseMergeDumpFlow/{RUN_ID}")
        if not run.successful:
            raise ValueError(f"{run} was not successful")
        return run.data


def get_organisation(run: Optional[Run] = None) -> pd.DataFrame:
    run_data = run.data if run else get_run_data()
    return run_data.organisation


def get_address(run: Optional[Run] = None) -> pd.DataFrame:
    run_data = run.data if run else get_run_data()
    return run_data.address


def get_sector(run: Optional[Run] = None) -> pd.DataFrame:
    """Returns most up-to-date sector rankings."""
    run_data = run.data if run else get_run_data()
    return (
        run_data.organisationsector.sort_values("date")
        .drop_duplicates(["company_number", "rank"], keep="last")
        .rename(columns={"date": "data_dump_date", "sector_id": "SIC5_code"})
    )


def get_name(run: Optional[Run] = None) -> pd.DataFrame:
    run_data = run.data if run else get_run_data()
    return run_data.organisationname
