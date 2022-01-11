"""Data getter for Glass to Companies House matching."""
from typing import Dict, List, Optional, TypedDict

from metaflow import Run
import numpy.typing as npt

from industrial_taxonomy import config
from industrial_taxonomy.utils.metaflow import get_run


if config is None:
    raise FileNotFoundError("Could not find config file.")
MIN_MATCH_SCORE: int = config["glass_house"]["min_match_score"]

# Type aliases
glass_id = int
company_number = str


class FullMatchResult(TypedDict):
    sim_mean: int  # Mean similarity between names
    # Glass
    org_id: glass_id
    org_name: str
    # Companies House
    company_number: company_number
    company_name: str


def glass_companies_house_lookup(
    run: Optional[Run] = None,
    min_match_threshold: int = MIN_MATCH_SCORE,
) -> Dict[glass_id, company_number]:
    """Lookup from glass organisation ID to Companies House number."""
    run = run or get_run("JacchammerFlow")

    if min_match_threshold < run.data.drop_matches_below:
        raise ValueError(
            f"`min_match_threshold` ({min_match_threshold}) is less than the "
            f"`{run}.data.drop_matches_below` ({run.data.drop_matches_below})"
        )

    return {
        row["org_id"]: row["company_number"]
        for row in glass_companies_house_matches(run)
        if row["sim_mean"] >= min_match_threshold
    }


def glass_companies_house_matches(
    run: Optional[Run] = None,
) -> List[FullMatchResult]:
    """Gets matches between Glass and Companies house."""
    run = run or get_run("JacchammerFlow")

    return run.data.full_top_matches


def get_description_tokens(run: Optional[Run] = None) -> Dict[str, List[str]]:
    """Processed Glass description tokens for which a Companies House match exists."""
    run = run or get_run("GlassNlpFlow")
    return run.data.documents


def description_embeddings(
    run: Optional[Run] = None,
) -> npt.ArrayLike:
    """Gets embeddings of Glass organisation descriptions.

    Args:
        run: Run ID for GlassEmbed flow

    Returns:
        A 2d array of size (n, m) where n is the number of companies and m is
        the length of each embedding.
    """
    run = run or get_run("GlassEmbed")
    return run.data.embeddings


def embedded_org_ids(run: Optional[Run] = None) -> List[int]:
    """Gets IDs of embedded Glass organisations."""
    run = run or get_run("GlassEmbed")
    return run.data.org_ids


def embedded_org_descriptions(run: Optional[Run] = None) -> List[str]:
    """Gets descriptions of embedded Glass organisations."""
    run = run or get_run("GlassEmbed")
    return run.org_descriptions


def encoder_name(run: Optional[Run] = None) -> str:
    """Gets name of model used to create Glass org embeddings."""
    run = run or get_run(run)
    return run.model_name
