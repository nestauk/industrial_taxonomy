from collections import defaultdict
from operator import itemgetter
from toolz.itertoolz import partition_all
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional, Dict, Callable


class Groupby:
    """Performs groupby and apply

    Args:
        keys (np.array): 1-D array of key values.
        values (np.array): 1-D array of values. Must be same length as `keys`.
    """

    def __init__(self, keys: npt.NDArray, values: npt.NDArray):
        self.keys = keys
        self.values = values

    def groupby_apply(self, function: Callable) -> npt.NDArray:
        """Groups values by each key according to their position and applies
        `function` to each group.

        Args:
            function: A function to apply to the values corresponding to each
                key. For example, `np.mean`, would return the mean of values
                for each key.

        Returns:
            agg: 2-D array containing unique keys in the first column and the
                corresponding aggregated values in the second column.
        """
        unique_keys = np.unique(self.keys)
        agg = []
        for k in unique_keys:
            v = self.values[self.keys == k]
            agg.append(function(v))
        agg = np.array(agg)
        return unique_keys, agg


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
    embedding_locs = list(itemgetter(*glass_org_ids_clusters)(glass_org_id_to_loc))
    return embeddings[embedding_locs]


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
        dists: Cosine distances of from
    """
    knn = []
    n = 0
    for embedding_chunk in partition_all(chunk_size, embeddings):
        dists, knn_ids = index.search(np.array(embedding_chunk), k + 1)
        knn.extend(list(zip(knn_ids[:, 1:], dists[:, 1:])))

        n += np.array(embedding_chunk).shape[0]
        print(f"Found neighbours for {n} samples.")

    return knn


def intify_clusters(clusters: List[Tuple[str, str]]) -> List[Tuple[str, int]]:
    """Converts org IDs in cluster tuples from sting to integer.

    Args:
        clusters: List of cluster ID + string org ID pairs.

    Returns:
        List of cluster ID and integer org ID pairs.
    """
    return [(cluster, int(org_id)) for cluster, org_id in clusters]


def assign_knn_cluster(
    knn: List[Tuple[npt.NDArray, npt.NDArray]],
    id_cluster_lookup: Dict[int, int],
    agg_func: Callable,
    org_ids: List[int],
) -> List[Tuple[int, str]]:
    """Finds the cluster with the smallest average distance from a set of K
    nearest neighbours.

    Args:
        knn: List of array tuples containing the nearest neighbour IDs and
            distances.
        id_cluster_lookup: Dictionary lookup from index IDs to to cluster
            labels.
        agg_func: A `numpy` function passed to `Groupby` to calculate the
            aggregate distance to nearest neighbour clusters.
        org_ids: List of the organisation IDs.

    Returns:
        nearest_clusters: Records of FAISS index IDs and their assigned cluster.
    """
    nearest_clusters = defaultdict(list)
    for org_id, (knn_ids, dists) in zip(org_ids, knn):
        knn_cluster_ids = itemgetter(*knn_ids)(id_cluster_lookup)
        gb = Groupby(np.array(knn_cluster_ids), dists)
        unique_clusters, aggregated = gb.groupby_apply(agg_func)
        unique_clusters = unique_clusters[np.argsort(aggregated)[::-1]]
        aggregated = np.sort(aggregated)[::-1]

        nearest_clusters["org_id"].append(org_id)
        nearest_clusters["best_cluster"].append(unique_clusters[0])
        nearest_clusters[f"best_cluster_{agg_func.__name__}_dist"].append(aggregated[0])

    return nearest_clusters


def add_original_clusters(
    knn_clusters: List[Dict],
    original_clusters: List[Tuple[int, int]],
) -> List[Dict]:
    """If an org has already been assigned to a cluster, this adds that cluster
    to the record. Otherwise the org will be assigned `None`.

    Args:
        knn_clusters: Records of clusters found via KNN search.
        original_clusters: List of org IDs that are already assigned to a
            cluster.

    Returns:
        knn_clusters: KNN clusters with original cluster.
    """
    original_clusters_lookup = dict(
        (org_id, clust_id) for clust_id, org_id in original_clusters
    )

    knn_clusters["original_cluster"] = [
        original_clusters_lookup.get(org_id, None) for org_id in knn_clusters["org_id"]
    ]
    return knn_clusters
