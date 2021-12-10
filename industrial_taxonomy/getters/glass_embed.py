"""Data getter for Glass company description embeddings."""
from functools import lru_cache
from typing import List, Optional
import numpy.typing as npt

from metaflow import Flow, Run
from metaflow.exception import MetaflowNotFound


@lru_cache()
def get_run():
    """Last successful run with `--production`."""
    runs = Flow("GlassEmbedFlow").runs("project_branch:prod")
    try:
        return next(filter(lambda run: run.succesful, runs))
    except StopIteration as exc:
        raise MetaflowNotFound("Embedding run not found") from exc


def glass_embeddings(
    run: Optional[Run] = None,
) -> npt.ArrayLike:
    """ "Gets embeddings of Glass orgsanisations."""
    run = run or get_run()
    return run.data.embeddings


def org_ids(run: Optional[Run] = None) -> List[int]:
    """Gets IDs of embedded Glass organisations."""
    run = run or get_run()
    return run.data.org_ids
