"""Data getters for topsbm outputs"""

from collections import defaultdict
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


def org_ids_reassigned(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """The Glass organisation IDs of the companies in this output"""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.org_ids
    else:
        return run.data.org_ids_rest


def original_text_sector(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """the text sector for each company from the
    original clustering. If the companies were not included in that
    the text sector clustering, this will be `None` for each company."""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.original_text_sector
    else:
        return run.data.original_text_sector_rest


def assigned_text_sector(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """the text sector of each company after reassignment"""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.assigned_text_sector
    else:
        return run.data.assigned_text_sector_rest


def knn_org_ids(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """the Glass organisation IDs of the K nearest neighbours"""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.knn_org_ids
    else:
        return run.data.knn_org_ids_rest


def knn_original_text_sectors(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """the text sectors of the K nearest neighbours before reassignment"""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.knn_original_text_sectors
    else:
        return run.data.knn_original_text_sectors_rest


def knn_assigned_text_sectors(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """the text sectors of the K nearest neighbours after reassignment"""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.knn_assigned_text_sectors
    else:
        return run.data.knn_assigned_text_sectors_rest 


def knn_sims(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """the similarity scores of the K nearest neighbours"""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.knn_sims
    else:
        return run.data.knn_sims_rest 


def knn_text_sector_agg(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """unique set of text sectors of the K nearest neighbours after reassignent"""
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.knn_text_sector_agg
    else:
        return run.data.knn_text_sector_agg_rest


def knn_text_sector_agg_sims(
    run: Optional[Run] = None,
    clustered: bool = True,
):
    """average K nearest neighbours similarity of the unique set of text
    sectors (if more than one nearest neighbour belong to the same text sector,
    this is the average similarity score for those neighbours)
    """
    run = run or get_run("ClusterReassignFlow")
    if clustered:
        return run.data.knn_text_sector_agg_sims
    else:
        return run.data.knn_text_sector_agg_sims_rest
