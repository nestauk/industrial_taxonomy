import numpy as np
from operator import itemgetter
from toolz.itertoolz import partition_all

from metaflow import (
    conda_base,
    current,
    FlowSpec,
    Parameter,
    project,
    step,
)

from industrial_taxonomy.getters.glass_embed import (
    glass_description_embeddings,
    org_ids,
)
from industrial_taxonomy.getters.cluster import (
    cluster_ids,
)  # TODO change this to match Juan's code

from utils import Groupby


K = 20


@conda_base(
    python="3.8", 
    libraries={
        "pytorch::faiss": "1.7.2",
        "scikit-learn": "1.0.2",
        }
)
@project(name="industrial_taxonomy")
class ClusterReassignFlow(FlowSpec):
    """Reassign companies to clusters based on their K nearest neighbours.

    Attributes:
        clusters: Glass organisation IDs and their cluster labels.
        clusters_reassigned: Glass organisation IDs and their reassigned
            cluster labels.
        glass_org_embeddings: Glass organisation embeddings from `GlassEmbed`.
        glass_org_ids: Glass organisation IDs of organisations with embeddings.
    """

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    @step
    def start(self):
        """Load data and truncate if in test mode."""
        n_samples = 20_000 if self.test_mode and not current.is_production else None

        self.glass_org_embeddings = glass_description_embeddings()[:n_samples]
        self.glass_org_ids = org_ids()[:n_samples]
        self.clusters = glass_cluster_ids()[:n_samples]

        self.next(self.silhouette_pre_reassign)

    @step
    def silhouette_pre_reassign(self):
        """Calculate sample silhouette score for Glass orgs based on cluster 
        IDs before reassignment.
        """
        from sklearn.metrics import silhouette_samples

        self.silhouette_before = silhouette_samples(
            self.glass_org_embeddings,
            self.clusters,
            metric="cosine"
        )
        self.next(generate_index)

    @step
    def generate_index(self):
        """Generates a FAISS index of Glass description embeddings."""
        import faiss

        dimensions = self.glass_org_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimensions)
        self.index.add(self.glass_org_embeddings)

        self.next(self.reassign_companies)

    @step
    def reassign_companies(self):
        """Queries FAISS index for the K nearest neighbours to each company
        given an embedding of its Glass description. Assigns a cluster label
        based on the labels of the nearest neighbours."""
        import faiss

        self.clusters_reassigned = []
        org_id_to_embedding_lookup = dict(
            zip(self.glass_org_ids, self.glass_org_embeddings)
        )
        org_id_to_cluster_id_lookup = dict(self.clusters)
        index_to_org_id_lookup = dict(enumerate(self.org_ids))

        for cluster_chunk in partition_all(10_000, self.clusters):
            org_ids_chunk = [c[0] for c in cluster_chunk]
            query_embeddings = itemgetter(*org_ids_chunk)(org_id_to_embedding_lookup)
            dists, nearest_ids = self.index.search(query_embeddings, K + 1)
            dists = dists[:, 1:]
            nearest_ids = nearest_ids[:, 1:]

            for org_id, nearest, dist in zip(org_ids_chunk, nearest_ids, dists):
                nearest_cluster_ids = itemgetter(
                    *itemgetter(*nearest)(index_to_org_id_lookup)
                )(org_id_to_cluster_id_lookup)
                gb = Groupby(nearest_cluster_ids, dist)
                aggregated = gb.groupby_apply(np.mean)
                best_cluster = aggregated[:, 0][np.argmax(aggregated[:, 1])]

                self.clusters_reassigned.append((org_id, best_cluster))

        self.next(self.end)

    def silhouette_post_reassign(self):
        """"Calculate sample silhouette score for Glass orgs based on cluster 
        IDs after reassignment.
        """
        from sklearn.metrics import silhouette_samples

        self.silhouette_before = silhouette_samples(
            self.glass_org_embeddings,
            self.clusters_reassigned,
            metric="cosine"
        )
        self.next(self.end)

    @step
    def end(self):
        """No-op."""
        pass


if __name__ == "__main__":
    ClusterReassignFlow()
