"""Data getter for Glass to Companies House matching."""
from functools import lru_cache
from typing import Dict, List, Optional, TypedDict

from metaflow import Flow, Run
from metaflow.exception import MetaflowNotFound

from industrial_taxonomy import config


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


@lru_cache()
def get_run():
    """Last successful run executed with `--production`."""
    runs = Flow("JacchammerFlow").runs("project_branch:prod")
    try:
        return next(filter(lambda run: run.successful, runs))
    except StopIteration as exc:
        raise MetaflowNotFound("Matching run not found") from exc


def glass_companies_house_lookup(
    run: Optional[Run] = None,
    min_match_threshold: int = MIN_MATCH_SCORE,
) -> Dict[glass_id, company_number]:
    """Lookup from glass organisation ID to Companies House number."""
    run = run or get_run()

    return {
        row["org_id"]: row["company_number"]
        for row in glass_companies_house_matches(run)
        if row["sim_mean"] >= min_match_threshold
    }


def glass_companies_house_matches(
    run: Optional[Run] = None,
) -> List[FullMatchResult]:
    """Gets matches between Glass and Companies house."""
    run = run or get_run()

    return run.data.full_top_matches
