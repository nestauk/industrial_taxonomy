"""Scripts to fetch and process secondary data"""

import requests
import pandas as pd
import yaml
from io import StringIO
from scipy.stats import zscore
from toolz import pipe, partial


ONS_LU = "https://www.ons.gov.uk/visualisations/dvc1786/datadownload.csv"
NUTS_LA_LU = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LAD16_LAU118_NUTS318_NUTS218_NUTS118_UK_LUv2/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json"

# URL for data in github
BEIS_URL = (
    "https://raw.githubusercontent.com/nestauk/beis-indicators/dev/ds/data/processed"
)

BEIS_INDICATORS = [
    "hebci/ce_cpd_income.nuts3",
    "travel/travel_time_to_road_junctions.nuts3",
    "hebci/ip_revenue.nuts3",
    "hebci/consultancy_facilities_public_third.nuts3",
    "aps/aps_econ_active_stem_density_data.nuts3",
    "hebci/contract_research_non_sme.nuts3",
    "industry/economic_complexity_index.nuts3",
    "hesa/gbp_research_income.nuts3",
    "migration/net_migration_rate_international.nuts3",
    "hebci/consultancy_facilities_non_sme.nuts3",
    "aps/aps_econ_active_stem_associate_profs_data.nuts3",
    "cordis/cordis_funding.nuts3",
    "ashe_mean_salary/ashe_mean_salary.nuts3",
    "industry/employment_culture_entertainment_recreation.nuts3",
    "hebci/spinoff_investment.nuts3",
    "hebci/graduate_startups.nuts3",
    "travel/travel_time_to_airport.nuts3",
    "aps/aps_econ_active_stem_profs_data.nuts3",
    "hesa/area_university_site.nuts3",
    "hebci/contract_research_public_third.nuts3",
    "hesa/total_stem_postgraduates.nuts3",
    "hebci/regeneration_development.nuts3",
    "hesa/total_stem_students.nuts3",
    "travel/travel_time_to_work.nuts3",
    "hebci/consultancy_facilities_sme.nuts3",
    "hebci/collaborative_research_cash.nuts3",
    "migration/net_migration_rate_internal.nuts3",
    "eurostat/epo_patent_applications.nuts3",
    "travel/travel_time_to_rail.nuts3",
    "hebci/ce_cpd_learner_days.nuts3",
    "hesa/total_university_buildings.nuts3",
    "aps/aps_pro_occupations_data.nuts3",
    "aps/aps_nvq4_education_data.nuts3",
    "hebci/contract_research_sme.nuts3",
    "crunchbase/companies_founded.nuts3",
    "hesa/total_postgraduates.nuts3",
    "gtr/total_ukri_funding.nuts3",
    "eurostat/epo_hightech_patent_applications.nuts3",
    "broadband/broadband_download_speed_data.nuts3",
    "innovate_uk/gbp_innovate_uk_funding.nuts3",
    "housing/house_price_normalised.nuts3",
    "eurostat/eu_trademark_applications.nuts3",
    "hebci/spinoff_revenue.nuts3",
    "defra/air_pollution_mean_pm10.nuts3",
    "hesa/fte_research_students.nuts3",
]


def fetch_ons() -> pd.DataFrame:
    """Fetch ONS levelling up data"""
    response = requests.get(ONS_LU)
    file_object = StringIO(response.content.decode("latin"))
    return pd.read_csv(file_object, skiprows=2)


def clean_ons(ons_table: pd.DataFrame) -> pd.DataFrame:
    """Clean columns names in ONS data"""
    ons_ = ons_table.copy()
    ons_.columns = ons_.columns.str.lower().str.strip()
    return ons_.assign(indicator=lambda df: df["indicator"].str.replace("'", ""))


def make_zscore(table, group_var: list = ["indicator"]) -> pd.DataFrame:
    """Create a zscore for each indicator in the data"""

    return (
        table.groupby(group_var)
        .apply(lambda df: df.dropna().assign(zscore=lambda df: zscore(df["value"])))
        .reset_index(drop=True)
    )


def fetch_ons_levelling_up() -> pd.DataFrame:
    """Fetch and clean ONS data"""

    return pipe(fetch_ons(), clean_ons, make_zscore)


def standardise_ons(ons_data: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise variable nanes in ONS data"""

    return (
        pipe(ons_data, clean_ons, make_zscore)
        .rename(columns={"areacd": "la_code"})
        .assign(source="ons_levelling_up")[
            ["la_code", "period", "indicator", "source", "value", "zscore"]
        ]
    )


def fetch_nuts_lad() -> dict:
    """Fetch lookup between NUTS and LA codes"""

    return requests.get(NUTS_LA_LU).json()


def nuts_lad_lookup() -> dict:
    """Create lookup between NUTS and LA codes"""
    return pipe(
        fetch_nuts_lad(),
        lambda response: pd.DataFrame(
            [area["attributes"] for area in response["features"]]
        ),
        lambda df: df.set_index("NUTS318CD")["LAD16CD"].to_dict(),
    )


def indicator_name(source_indicator: str) -> str:
    """Extract indicator name from the source/indicator input"""

    return source_indicator.split(".")[0].split("/")[1]


def fetch_beis_table(source_indicator: str) -> pd.DataFrame:
    """
    Fetch a table from the Nesta BEIS github repo

    Args:
        source_indicator: source/indicator_name string

    Returns:
        A dataframe with the indicators
    """

    base_url = f"{BEIS_URL}/{source_indicator}"

    return pd.read_csv(base_url.format(indicator=source_indicator) + ".csv")


def standardise_beis(beis_table: pd.DataFrame, var_name, nuts_lad_lookup: dict):
    """Standardise the columnns etc for a BEIS table"""

    return (
        pipe(
            beis_table.rename(columns={var_name: "value", "year": "period"}).assign(
                indicator=var_name
            ),
            partial(make_zscore, group_var=["indicator", "period"]),
        )
        .assign(zscore=lambda df: zscore(df["value"]))
        .assign(la_code=lambda df: df["nuts_id"].map(nuts_lad_lookup))
        .assign(source="nesta_beis")[
            ["la_code", "period", "indicator", "source", "value", "zscore"]
        ]
    )


def fetch_schema(source_indicator: str):
    """Fetch the schema for an indicator"""

    base_url = f"{BEIS_URL}/{source_indicator}"

    return yaml.safe_load(requests.get(base_url + ".yaml").content)
