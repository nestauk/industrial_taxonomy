from collections import defaultdict
from operator import itemgetter
from toolz.itertoolz import partition_all
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional


def get_clusters_embeddings(
    clusters: List[Tuple[int, int]],
    org_ids: List[int],
    embeddings: npt.NDArray,
) -> npt.NDArray:
    """Returns the embeddings for a set of clusters.

    Args:
        clusters: List of cluster ID + org ID pairs.
        org_ids: List of Glass org IDs corresponding to embedding rows.
        embeddings: 2-D array of Glass org embeddings.

    Returns:
        Embeddings for the organisations in clusters, in the order that
            they were passed.
    """
    glass_org_ids_clusters = [c[1] for c in clusters]
    glass_org_id_to_loc = dict(zip(org_ids, range(len(org_ids))))
    embedding_locs = np.array(itemgetter(*glass_org_ids_clusters)(glass_org_id_to_loc))
    return embeddings[embedding_locs], embedding_locs


def get_non_clusters_embeddings(
    cluster_embedding_locs: npt.NDArray,
    embeddings: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Returns the embeddings for orgs not included in a set of clusters.

    Args:
        cluster_embedding_locs: 1-D array of glass org ID positional indexes
            for orgs in a cluster.
        embeddings: 2-D array of Glass org embeddings.

    Returns:
        Embeddings for the organisations and their positional indexes.
    """
    mask = np.zeros(embeddings.shape[0], dtype=bool)
    mask[cluster_embedding_locs] = True
    return embeddings[~mask], np.argwhere(~mask).ravel()


def generate_index(
    index: "faiss.Index",
    embeddings: npt.NDArray,
) -> "faiss.Index":
    """Generate a FAISS index from an array.

    Args:
        index: A FAISS Index class.
         embeddings: An array of embeddings.

    Returns:
        index: Constructed FAISS index with dimensions = embeddings.shape[1].
    """
    dimensions = embeddings.shape[1]
    index = index(dimensions)
    index.add(embeddings)
    return index


def find_knn(
    embeddings: npt.NDArray,
    index: "faiss.Index",
    k: int,
    chunk_size: Optional[int] = None,
) -> List[Tuple[npt.NDArray, npt.NDArray]]:
    """Gets the k nearest neighbours to a set of embeddings from an index.

    Args:
        embeddings: 2-d array of embeddings.
        index: A 'trained' `faiss` index.
        k: The number of nearest neighbours to search for.
        chunk_size: If an integer is passed, the embeddings will be split into
            chunks of this size.

    Returns:
        knn_ids: 2-d array of the index IDs of the k nearest
        sims: Cosine similarities of from
    """
    knn = []
    for embedding_chunk in partition_all(chunk_size, embeddings):
        sims, knn_ids = index.search(np.array(embedding_chunk), k + 1)
        knn.extend(list(zip(knn_ids, sims)))

    return knn


def intify_clusters(clusters: List[Tuple[str, str]]) -> List[Tuple[str, int]]:
    """Converts org IDs in cluster tuples from sting to integer.

    Args:
        clusters: List of cluster ID + string org ID pairs.

    Returns:
        List of cluster ID and integer org ID pairs.
    """
    return [(cluster, int(org_id)) for cluster, org_id in clusters]


def decompose_knn(
    knn_ids: npt.NDArray,
    sims: npt.NDArray,
    source: bool = True,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, Optional[int]]:
    """Extracts the K nearest neighbours and their similarities from a FAISS
    search, depending on whether the query vector is included in the index.

    Args:
        knn_ids: 1-d array of KNN IDs from a FAISS index search.
        sims: 1-d array of similarities from a FAISS index search.
        source: Whether the query vector and its ID are included in the
            index. If True, also returns the popped ID of that vector from the
            index.

    Returns:
        A tuple (knn_ids, sims, source_id (optional)), where knn_ids is a 1-d
        array of KNN IDs from a FAISS index search not includingthe query
        vector, sims is a 1-d array of similarities from a FAISS index search
        also not including the query vector and source_id is the FAISS index ID
        of the query vector if included.

    """
    if source:
        source_id = knn_ids[0]
        knn_ids = knn_ids[1:]
        sims = sims[1:]
        return knn_ids, sims, source_id
    else:
        knn_ids = knn_ids[:-1]
        sims = sims[:-1]
        return knn_ids, sims


def get_best_cluster(
    clusters: npt.NDArray,
    agg_sims: npt.NDArray,
) -> Tuple[int, int]:
    """Finds the cluster ID with the highest similarity value.

    Args:
        clusters: 1-d array of cluster IDs.
        agg_sims: 1-d array of aggregated similarities.

    Returns:
        Tuple of the cluster ID with the highest average similarity and the
        similarity score.
    """
    best_pos = np.argsort(agg_sims)[-1]
    return clusters[best_pos], agg_sims[best_pos]


def mean_cluster_similarities(
    clusters: npt.ArrayLike,
    sims: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculates the mean similarities for each neighbour within a cluster.

    Args:
        clusters: Cluster label for each nearest neighbour.
        sims: Similarity score for each nearest neighbour.

    Returns:
        A tuple (unique_clusters, mean_sims) where unique_clusters a sorted
        array of the cluster labels and mean_sims is the mean similarity score
        of the nearest neighbours associated with each label.
    """
    unique_clusters, inv, counts = np.unique(
        clusters, return_inverse=True, return_counts=True
    )
    oht = np.eye(unique_clusters.shape[0], inv.shape[0])[:, inv]
    oht = oht * sims
    mean_sims = np.sum(oht, axis=1) / counts
    return unique_clusters, mean_sims


def reassign_clustered(
    knn: List[Tuple[npt.NDArray, npt.NDArray]],
    clusters: List[Tuple[str, int]],
    min_sim_threshold: float = 0.6,
    n_iter: int = 20,
    epsilon: float = 0.05,
) -> List[Tuple[str, int]]:
    """Reassigns companies to new clusters based on the average similarity to
    nearest neighbours belonging to clusters.

    Args:
        knn: A list of pairs of nearest neighbour index IDs and their
            similarities.
        clusters: A list of cluster ID and org ID pairs.
        min_sim_threshold: Minimum cosine similarity for a cluster
            reassignment to be accepted.
        n_iter: Number of timer to iteratively reaassign companies to clusters.
        epsilon: Minimum fraction of companies required for an iteration of
            reassignment to happen. If the fraction of companies being
            reassigned falls below this value, then there will be no more
            reassignment iterations, even if n_iter has not been reached.

    Returns:
        clusters: A list of reassigned cluster ID and org ID pairs.

    """
    org_ids = [c[1] for c in clusters]

    shift = epsilon
    complete = 0
    while (shift >= epsilon) and (n_iter > complete):
        index_id_cluster_lookup = np.array([c[0] for c in clusters])

        changed = 0
        _clusters = []
        for org_id, (knn_ids, sims) in zip(org_ids, knn):
            knn_ids, sims, source_id = decompose_knn(
                knn_ids,
                sims,
                source=True,
            )

            knn_cluster_ids = index_id_cluster_lookup[knn_ids]
            unique_clusters, agg_sims = mean_cluster_similarities(knn_cluster_ids, sims)

            best_cluster, best_sim = get_best_cluster(unique_clusters, agg_sims)

            original_cluster = index_id_cluster_lookup[source_id]
            same_cluster = best_cluster == original_cluster

            if same_cluster:
                _clusters.append((original_cluster, org_id))
            else:
                if best_sim >= min_sim_threshold:
                    _clusters.append((best_cluster, org_id))
                    changed += 1
                else:
                    _clusters.append((original_cluster, org_id))

        clusters = _clusters
        complete += 1
        shift = changed / len(knn)

    return clusters


def assign_non_clustered(
    knn: List[Tuple[npt.NDArray, npt.NDArray]],
    org_ids: List[int],
    locs: npt.NDArray,
    clusters: List[Tuple[int, int]],
    min_sim_threshold: float = 0.6,
) -> List[Tuple[int, int]]:
    """Assigns companies that were not clustered to clusters based on the
    average similarity to nearest neighbours belonging to clusters.

    Args:
        knn: A list of pairs of nearest neighbour index IDs and their
            similarities.
        org_ids: A list of all organisation IDs with embeddings.
        locs: A list of positional indexes for organisations that are being
            assigned to a cluster.
        clusters: A list of cluster ID and org ID pairs.
        min_sim_threshold: Minimum cosine similarity for a cluster
            reassignment to be accepted.

    Returns:
        clusters: A list of assigned cluster ID and org ID pairs.

    """
    org_ids = np.array(org_ids)[locs]
    index_id_cluster_lookup = np.array([c[0] for c in clusters])

    clusters_assigned = []
    for org_id, (knn_ids, sims) in zip(org_ids, knn):
        knn_ids, sims = decompose_knn(
            knn_ids,
            sims,
            source=False,
        )

        knn_cluster_ids = index_id_cluster_lookup[knn_ids]
        unique_clusters, agg_sims = mean_cluster_similarities(
            knn_cluster_ids,
            sims,
        )

        best_cluster, best_sim = get_best_cluster(unique_clusters, agg_sims)

        if best_sim >= min_sim_threshold:
            clusters_assigned.append((best_cluster, org_id))
        else:
            clusters_assigned.append((None, org_id))

    return clusters_assigned


def knn_output(
    org_ids: List[int],
    original_clusters: List[Tuple[str, int]],
    assigned: List[Tuple[str, int]],
    knn: List[Tuple[npt.NDArray, npt.NDArray]],
    rest: bool = False,
) -> Dict[str, List]:
    """Creates a dictionary containing results of nearest neighbour text
    sector assignment process for each company.

    Args:
        org_ids:
        original_clusters: A list of cluster and organisation ID tuples.
        assigned: A list of assigned cluster and organisation ID tuples.
        knn: A list of nearest neighbour index IDs and their respective average
            similarities.
        reassigned: Use True if the orgs being passed were originally in text
            sectors. False if they are being assigned to text sectors.

    Returns:
        output:
    """

    index_id_ts_lookup = np.array([c[0] for c in original_clusters])
    index_id_org_id_lookup = np.array([c[1] for c in original_clusters])
    assigned_ts_lookup = np.array(c[1] for c in assigned)

    output = defaultdict(list)

    for knn_ids, sims in knn:
        if rest:
            knn_ids, sims = decompose_knn(knn_ids, sims, source=False)
        else:
            knn_ids, sims, _ = decompose_knn(knn_ids, sims)

        output["knn_org_ids"].append(index_id_org_id_lookup[knn_ids])
        output["knn_original_text_sectors"].append(list(index_id_ts_lookup[knn_ids]))
        output["knn_sims"].append(sims)
        output["knn_assigned_text_sectors"].append(assigned_ts_lookup[knn_ids])

    output["org_id"] = org_ids
    output["assigned_text_sector"] = [a[0] for a in assigned]

    if rest:
        output["original_text_sector"] = [None for _ in org_ids]
    else:
        output["original_text_sector"] = list(index_id_ts_lookup)

    return output
