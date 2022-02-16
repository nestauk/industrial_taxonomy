"""Data getter for Glass to Companies House matching."""
from typing import Dict, List, Optional, TypedDict

from metaflow import Run
import numpy.typing as npt
from matplotlib.figure import Figure

from industrial_taxonomy import config
from industrial_taxonomy.utils.metaflow import get_run
from industrial_taxonomy.utils.collections import reverse_dict
from industrial_taxonomy.getters.companies_house import get_sector


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


def glass_sic4_lookup() -> dict:
    """Creates a lookup between glass_ids and the first ranked 4 digit SIC codes"""

    return (
        get_sector()
        .assign(
            org_id=lambda df: df.index.map(
                reverse_dict(glass_companies_house_lookup())
            ),
            SIC4_CODE=lambda df: df["SIC5_code"].str.slice(stop=-1),
        )
        .query("rank == 1")
        .dropna(axis=0, subset=["org_id"])
        .set_index("org_id")["SIC4_CODE"]
        .to_dict()
    )


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
    return run.data.org_descriptions


def encoder_name(run: Optional[Run] = None) -> str:
    """Gets name of model used to create Glass org embeddings."""
    run = run or get_run("GlassEmbed")
    return run.data.model_name


def qa_tokenized_length_hist_fig(run: Optional[Run] = None) -> Figure:
    """Cumulative histogram of tokenized Glass description lengths."""
    run = run or get_run("GlassEmbedQA")
    return run.data.tokenized_length_hist_fig


def qa_percent_not_truncated(run: Optional[Run] = None) -> float:
    """Percent of descriptions not truncated by the model max input length."""
    run = run or get_run("GlassEmbedQA")
    return run.data.percent_not_truncated


def qa_embedding_matches(run: Optional[Run] = None) -> Dict:
    """Pairs of Glass organisations matched by the cosine similarity of
    their embeddings."""
    run = run or get_run("GlassEmbedQA")
    return run.data.embedding_matches
