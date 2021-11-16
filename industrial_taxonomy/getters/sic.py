"""Data getters for SIC taxonomy."""
from functools import lru_cache
from typing import Dict, Optional

from metaflow import Flow, Run
from metaflow.exception import MetaflowNotFound

from industrial_taxonomy.pipeline.sic.utils import LEVELS

# Type aliases
code_name = str
code = str


@lru_cache()
def get_run():
    """Last successful run executed with `--production`."""
    runs = Flow("Sic2007Structure").runs("project_branch:prod")
    try:
        return next(filter(lambda run: run.successful, runs))
    except StopIteration as exc:
        raise MetaflowNotFound("Matching run not found") from exc


def level_lookup(level: int, run: Optional[Run] = None) -> Dict[code, code_name]:
    """Get SIC names for `level` index.

    Args:
        level: Number of SIC digits/letters to fetch lookup for
        run: Metaflow Run to get data artifacts from

    Returns:
        Lookup from SIC code to name

    Raises:
        ValueError: if 1 <= level <= 5
    """
    if run is None:
        run = get_run()

    if level not in range(1, len(LEVELS) + 1):
        raise ValueError(f"Level: {level} not valid.")

    return getattr(run.data, f"{LEVELS[level - 1]}_lookup")
