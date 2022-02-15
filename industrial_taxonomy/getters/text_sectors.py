"""Data getters for topsbm outputs"""

from metaflow import Run
from typing import Dict, List, Optional, Tuple

from industrial_taxonomy.utils.metaflow import get_run
from industrial_taxonomy.pipeline.glass_clusters.sbmtm import sbmtm


clustering_param = str
sector_id = str
glass_id = str


def text_sectors(
    filtered: bool = True,
    run: Optional[Run] = None,
) -> Dict[clustering_param, List[Tuple[sector_id, glass_id]]]:
    """Gets text sector assignments from the topic modelling

    Args:
        filtered: filters low silhouette score clusters
        run: what ClusterGlass flow run to use.
            Defaults to the latest production run

    Returns:
        A lookup between cluster assignment parameters (how many
        companies do we tell topsbm to assign to each cluster based on
        their representativeness) and the actual cluster assignments
        (a tuple with sector id and company id)

    """
    run = run or get_run("ClusterGlass")

    if filtered is True:
        clusters = run.data.clusters
        filtered_clusters = get_run("FilterClusters").data.text_sectors_filtered

        return {
            assign_share: [
                text_s
                for text_s in sectors
                if text_s[0] in filtered_clusters[assign_share]
            ]
            for assign_share, sectors in clusters.items()
        }

    else:
        return run.data.clusters


def topsbm_models(run: Optional[Run] = None) -> Dict[sector_id, sbmtm]:
    """Gets topic models trained on each sector"""
    run = run or get_run("ClusterGlass")
    return run.data.models


def clustered_sectors(run: Optional[Run] = None) -> List[str]:
    """Get the sectors where we performed the clustering"""

    run = run or get_run("ClusterGlass")

    return run.data.sectors
