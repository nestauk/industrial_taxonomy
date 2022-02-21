"""Data getters for topsbm outputs"""

from metaflow import Run
from typing import Dict, List, Optional, Tuple

from industrial_taxonomy.utils.metaflow import get_run
from industrial_taxonomy.pipeline.glass_clusters.sbmtm import sbmtm


clustering_param = str
sector_id = str
glass_id = str


def text_sectors(
    run: Optional[Run] = None,
) -> Dict[clustering_param, List[Tuple[sector_id, glass_id]]]:
    """Gets text sector assignments from the topic modelling

    Args:
        run: what ClusterGlass flow run to use.
            Defaults to the latest production run

    Returns:
        A lookup between cluster assignment parameters (how many
        companies do we tell topsbm to assign to each cluster based on
        their representativeness) and the actual cluster assignments
        (a tuple with sector id and company id)

    """
    run = run or get_run("ClusterGlass")
    return run.data.clusters


def topsbm_models(run: Optional[Run] = None) -> Dict[sector_id, sbmtm]:
    """Gets topic models trained on each sector"""
    run = run or get_run("ClusterGlass")
    return run.data.models


def reassigned_text_sectors(run: Optional[Run] = None) -> Dict[Dict[List]]:
    """Gets organisations assigned to text sectors based on their K nearest
    neighbours in a semantic similarity search.

    Returns:
        A dictionary for each clustering parameter that contains the Glass org
        id, the best matching cluster, the average distance to the nearest
        neighbours from that cluster and the original cluster (if applicable).
    """
    run = run or get_run("ClusterReassignFlow")
    return run.data.clusters_reassigned
