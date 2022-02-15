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
        agg = np.array([unique_keys, np.array(agg)])
        return agg


def find_knn(
    embeddings: npt.NDArray,
    index, # FAISS index
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
    for embedding_chunk in partition_all(chunk_size, embeddings):
        dists, knn_ids = index.search(embedding_chunk, k + 1)
        knn.extend(list(zip(knn_ids[:, 1:], dists[:, 1:])))

    return knn


def assign_knn_cluster(
    knn: List[Tuple[npt.NDArray, npt.NDArray]],
    id_cluster_lookup: Dict[int, int],
    agg_func: Callable,
    min_agg_distance: float,
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
        min_agg_distance: If the aggregate distance to nearest cluster is less
            than this, then samples are assigned to their existing cluster.

    Returns:
        nearest_clusters: List of IDs and their assigned cluster.
    """
    nearest_clusters = []
    for sample_id, (knn_ids, dists) in enumerate(knn):
        knn_cluster_ids = itemgetter(*knn_ids)(id_cluster_lookup)
        gb = Groupby(knn_cluster_ids, dists)
        aggregated = gb.groupby_apply(agg_func)
        smallest_dist = np.max(aggregated[:, 1])

        if smallest_dist >= min_agg_distance:
            nearest_cluster = aggregated[:, 0][np.argmax(aggregated[:, 1])]
            nearest_clusters.append((sample_id, nearest_cluster))
        else:
            nearest_clusters.append((sample_id, id_cluster_lookup[sample_id]))

    return nearest_clusters
