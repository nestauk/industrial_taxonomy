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


def reassigned_text_sectors(
    run: Optional[Run] = None,
    from_text_sectors: bool = True,
) -> Dict[str, Dict]:
    """Gets organisations assigned to text sectors based on their K nearest
    neighbours in a semantic similarity search.

    Returns:
        A dictionary for each clustering parameter that contains the Glass org
        id, the best matching cluster, the average distance to the nearest
        neighbours from that cluster and the original cluster (if applicable).
    """
    run = run or get_run("ClusterReassignFlow")

    if from_text_sectors:
        attributes = {
            "org_ids": run.data.org_ids,
            "original_text_sector": run.data.original_text_sector,
            "assigned_text_sector": run.data.assigned_text_sector,
            "knn_org_ids": run.data.knn_org_ids,
            "knn_original_text_sectors": run.data.knn_original_text_sectors,
            "knn_assigned_text_sectors": run.data.knn_assigned_text_sectors,
            "knn_sims": run.data.knn_sims,
            "knn_text_sector_agg": run.data.knn_text_sector_agg,
            "knn_text_sector_agg_sims": run.data.knn_text_sector_agg_sims,
        }
    else:
        attributes = {
            "org_ids": run.data.org_ids_rest,
            "original_text_sector": run.data.original_text_sector_rest,
            "assigned_text_sector": run.data.assigned_text_sector_rest,
            "knn_org_ids": run.data.knn_org_ids_rest,
            "knn_original_text_sectors": run.data.knn_original_text_sectors_rest,
            "knn_sims": run.data.knn_sims_rest,
            "knn_assigned_text_sectors": run.data.knn_assigned_text_sectors_rest,
            "knn_text_sector_agg": run.data.knn_text_sector_agg_rest,
            "knn_text_sector_agg_sims": run.data.knn_text_sector_agg_sims_rest,
        }
    output = defaultdict(dict)
    for attr_key, param_vals in attributes.items():
        # print(param_vals)
        for param, vals in param_vals.items():
            output[param][attr_key] = vals

    return output
