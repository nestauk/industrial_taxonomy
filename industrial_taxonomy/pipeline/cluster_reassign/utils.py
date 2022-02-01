import numpy as np


class Groupby:
    """Performs groupby and apply

    Args:
        keys (np.array): 1-D array of key values.
        values (np.array): 1-D array of values. Must be same length as `keys`.
    """

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def groupby_apply(self, function):
        """Groups values by each key according to their position and applies
        `function` to each group.

        Args:
            function: A function to apply to the values corresponding to each
                key. For example, `np.mean`, would return the mean of values
                for each key.

        Returns:
            2-D array containing unique keys in the first column and the
            corresponding aggregated values in the second column.
        """
        unique_keys = np.unique(self.keys)
        agg = []
        for k in unique_keys:
            v = self.values[self.keys == k]
            agg.append(function(v))
        agg = np.array([unique_keys, np.array(agg)])
        return agg

org_id_to_embedding_lookup = dict(
    zip(self.glass_org_ids, self.glass_org_embeddings)
)
org_id_to_cluster_id_lookup = dict(self.clusters)
index_to_org_id_lookup = dict(enumerate(self.org_ids))

org_ids_chunk = [c[0] for c in cluster_chunk]

def find_knn()


    cluster_labels_reassigned = []
    for label_chunk in partition_all(10_000, cluster_labels):
        query_embeddings = itemgetter(*org_ids_chunk)(org_id_to_embedding_lookup)




    dists, nearest_ids = self.index.search(query_embeddings, K + 1)
    dists = dists[:, 1:]
    nearest_ids = nearest_ids[:, 1:]

    for org_id, nearest, dist in zip(org_ids_chunk, nearest_ids, dists):
        nearest_cluster_ids = itemgetter(
            *itemgetter(*nearest)(index_to_org_id_lookup)
        )(org_id_to_cluster_id_lookup)
        gb = Groupby(nearest_cluster_ids, dist)
        aggregated = gb.groupby_apply(np.median)
        best_cluster = aggregated[:, 0][np.argmax(aggregated[:, 1])]

        self.clusters_reassigned.append((org_id, best_cluster))
