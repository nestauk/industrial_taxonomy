"""Utils to create a sector corpus"""
from toolz import pipe
from functools import partial
from typing import Dict, List
import pandas as pd

from industrial_taxonomy.getters.glass_house import (
    get_description_tokens,
    glass_sic4_lookup,
)

NE_CODES = {
    "CARDINAL",
    "DATE",
    "GPE",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERSON",
    "QUANTITY",
    "TIME",
}


def strip_nes(tokenised_dict: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Removes named entities from company tokenised descriptions"""

    return {
        k: [t for t in v if all(ne not in t for ne in NE_CODES)]
        for k, v in tokenised_dict.items()
    }


def filter_non_matched_comps(
    tokenised: Dict[int, List[str]], matched_ids: set
) -> Dict[int, List[str]]:
    """Removes tokenised descriptions of glass companies that were not matched with CH

    Args:
        tokenised: lookup between company ids and tokenised descriptions
        matched_ids: ids from glass companies matched with CH

    Returns:
        filtered tokenised descriptions dict
    """

    return {id_: tok for id_, tok in tokenised.items() if id_ in matched_ids}


def sector_tokens_lookup(
    tokenised: Dict[int, List[str]], gl_sic4: Dict[int, str], big_sectors: set
) -> Dict[int, List[str]]:
    """Creates a dict where keys are (big) sectors and
    values the tokenised descriptions of their companies.

    Args:
        tokenised: lookup between company ids and tokenised descriptions
        gl_sic4: lookup between company ids and SIC4s
        big_sectors: sectors above a certain size threshold

    Returns:
        dict with sectors and tokenised descriptions

    """

    return {
        sector: {id_: tok for id_, tok in tokenised.items() if gl_sic4[id_] == sector}
        for sector in big_sectors
    }


def make_sector_corpora(min_sector_size: int = 1000) -> Dict[str, List[str]]:
    """Creates a dict of sectors and the tokenised descriptions of their companies

    Args:
        min_sector_size: minimum sector size

    Returns:
        dict with sectors and tokenised descriptions for their companies
    """
    glass_sic4 = glass_sic4_lookup()
    selected_sectors = set(
        sector
        for sector, sector_n in pd.Series(glass_sic4).value_counts().items()
        if sector_n > min_sector_size
    )

    return pipe(
        get_description_tokens(),
        strip_nes,
        partial(filter_non_matched_comps, matched_ids=set(glass_sic4.keys())),
        partial(sector_tokens_lookup, gl_sic4=glass_sic4, big_sectors=selected_sectors),
    )
