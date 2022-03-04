# Data getters for official data

from functools import lru_cache

import geopandas as gp
from metaflow import Flow, Run
from metaflow.exception import MetaflowNotFound
from typing import Optional


try:  # Hack for type-hints on attributes
    import pandas as pd
except ImportError:
    pass


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

    Arguments:
        run: what run to get (if None it gets the lastest production run)

    Returns:
        Columns:
            Name: nuts1_name, dtype: str, NUTS1 region (e.g. Scotland, South East etc)
            Name: la_code, dtype: str, local authority code
            Name: la_name, dtype: str, local authority name
            Name: year dtype: int, year (ranges between 1998 and 2019
            Name: gva dtype: float, £M Gross value added

    """

    if run is None:
        run = get_run("LocalGdpData")

    return (
        pd.DataFrame(run.data.gva)
        .melt(
            id_vars=["itl1_region", "la_code", "la_name"],
            var_name="year",
            value_name="gva",
        )
        .rename(columns={"itl1_region": "nuts1_name"})
    )


def population_lad(run: Optional[Run] = None):
    """get the population in a local authority

    Arguments:
        run: what run to get (if None it gets the lastest production run)

    Returns:
        Columns:
            Name: nuts1_name, dtype: str, NUTS1 region (e.g. Scotland, South East etc)
            Name: la_code, dtype: str, local authority code
            Name: la_name, dtype: str, local authority name
            Name: year dtype: int, year (ranges between 1998 and 2019
            Name: pop dtype: float, population

    """

    if run is None:
        run = get_run("LocalGdpData")

    return (
        pd.DataFrame(run.data.pop)
        .melt(
            id_vars=["itl1_region", "la_code", "la_name"],
            var_name="year",
            value_name="pop",
        )
        .rename(columns={"itl1_region": "nuts1_name"})
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
        gva.merge(pop, on=["nuts1_name", "la_code", "la_name", "year"])
        .assign(gva_pc=lambda df: (1e6 * df["gva"] / df["pop"]).round(2))
        .drop(axis=1, labels=["gva", "pop"])
    )


def nomis(run: Optional[Run] = None):
    """Get nomis tables including variables from
        Annual Population Survey (APS)
        Annual Survey of Hours and Earnings (ASHE)

    Arguments:
        run: what run to get (if None it gets the lastest production run)

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


def local_benchmarking(run: Optional[Run] = None):
    """Get local benchmarking data from the ONS levelling up dataset
    and Nesta/BEIS local innovation dashboard

    Arguments:
        run: what run to get

    Returns:
        Columns:
            la_code: local authority code. Some
                missing values for BEIS areas coded
                with older NUTS codes where we don't
                have a lookup
            period: year (sometimes it is between years
                eg 2019-2020). The BEIS data generally
                includes multiple period per indicator,
                the ONS only includes one period
            indicator: name of the indicator
            source: whether it comes from ONS or Nesta / BEIS
            value: value of the variable (look in the
                raw data or schema for the unit)
            zscore: value normalised by indicator/year
    """

    if run is None:
        run = get_run("SecondaryIndicators")

    return run.data.secondary_table


def lad_boundaries(run: Optional[Run] = None) -> dict:
    """Read a geojson boundary and parse as geopandas"""

    if run is None:
        run = get_run("FetchBound")

    geojson = run.data.boundary

    return gp.GeoDataFrame.from_features(
        geojson["features"], crs=geojson["crs"]["properties"]["name"]
    )
