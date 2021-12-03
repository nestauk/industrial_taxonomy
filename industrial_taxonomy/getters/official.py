# Data getters for official data

from functools import lru_cache

from metaflow import Flow, Run
from metaflow.exception import MetaflowNotFound
from typing import Optional

import pandas as pd


@lru_cache()
def get_run(flow_name: str) -> Run:
    """Gets last successful run executed with `--production`"""
    runs = Flow(flow_name).runs("project_branch:prod")
    try:
        return next(filter(lambda run: run.successful, runs))
    except StopIteration as exc:
        raise MetaflowNotFound("Matching run not found") from exc


def gva_lad(run: Optional[Run] = None):
    """get the GVA in a local authority

    Returns:
        Columns:
            Name: itl1_region, dtype: str, NUTS1 region (e.g. Scotland, South East etc)
            Name: la_code, dtype: str, local authority code
            Name: la_name, dtype: str, local authority name
            Name: year dtype: int, year (ranges between 1998 and 2019
            Name: gva dtype: float, £M Gross value added

    """

    if run is None:
        run = get_run("LocalGdpData")

    return pd.DataFrame(run.data.gva).melt(
        id_vars=["itl1_region", "la_code", "la_name"], var_name="year", value_name="gva"
    )


def population_lad(run: Optional[Run] = None):
    """get the population in a local authority

    Returns:
        Columns:
            Name: itl1_region, dtype: str, NUTS1 region (e.g. Scotland, South East etc)
            Name: la_code, dtype: str, local authority code
            Name: la_name, dtype: str, local authority name
            Name: year dtype: int, year (ranges between 1998 and 2019
            Name: pop dtype: float, population

    """

    if run is None:
        run = get_run("LocalGdpData")

    return pd.DataFrame(run.data.pop).melt(
        id_vars=["itl1_region", "la_code", "la_name"], var_name="year", value_name="pop"
    )


def gva_pc_lad():
    """Get the GVA per capita in a local authority

    Returns:
        Columns:
            Name: nuts1_name, dtype: str, NUTS1 region (e.g. Scotland, South East etc)
            Name: la_code, dtype: str, local authority code
            Name: la_name, dtype: str, local authority name
            Name: year dtype: int, year (ranges between 1998 and 2019
            Name: gva_pc dtype: float, GDP per capita

    """

    gva = gva_lad()
    pop = population_lad()

    return (
        gva.merge(pop, on=["itl1_region", "la_code", "la_name", "year"])
        .assign(gva_pc=lambda df: (1e6 * df["gva"] / df["pop"]).round(2))
        .rename(columns={"itl1_region": "nuts1_name"})
        .drop(axis=1, labels=["gva", "pop"])
    )


def nomis(run: Optional[Run] = None):
    """Get nomis tables including variables from
        Annual Population Survey (APS)
        Annual Survey of Hours and Earnings (ASHE)

    Returns:
        Columns:
            Name: year, dtype: int, year when the data was collected
                (in the case of APS it will refer to the last
                month of the year when education information is available)
            Name: la_code, dtype: str, local authority code
            Name: la_name, dtype: str, local authority name
            Name: variable, dtype: str, variable including:
                    Economic activity rate (APS)
                    Employment rate (APS)
                    % with tertiary education (APS)
                    % with no qualification (APS)
                    Annual pay (gross) £ (ASHE)
            Name: value, dtype: float, value for the variable
            Name: source, dtype: str, aps or ashe
    """

    # Standardise variables with the other tables
    column_name_lookup = {
        "date": "year",
        "geography_name": "la_name",
        "geography_code": "la_code",
    }

    if run is None:
        run = get_run("NomisTables")

    return pd.DataFrame(run.data.nomis_dict).rename(columns=column_name_lookup)
