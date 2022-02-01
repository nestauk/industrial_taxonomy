import numpy as np

from metaflow import (
    conda_base,
    current,
    FlowSpec,
    Parameter,
    project,
    step,
)

from industrial_taxonomy.getters.text_sectors import text_sectors
from industrial_taxonomy.getters.glass_embed import (
    glass_description_embeddings,
    org_ids,
)

from utils import find_knn, assign_knn_cluster

K = 20
INDEX_SEARCH_CHUNK_SIZE = 10_000
REASSIGN_AGG_FUNC = np.median
MIN_REASSIGN_DIST = 0.8


@conda_base(
    python="3.8",
    libraries={
        "pytorch::faiss": "1.7.2",
        "scikit-learn": "1.0.2",
    },
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

        self.clusters = {}
        for param, clusters in text_sectors.items():
            clusters = dict(clusters)
            clusters = {
                org_id: clusters[org_id]
                for org_id in clusters.keys() & set(self.org_ids)
            }
            self.clusters[param] = clusters
        # convert code below to work with dict inputs
        # remove silhouette pre reassign
        self.next(self.generate_index)

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
        based on the labels of the nearest neighbours.
        """
        knn = find_knn(
            self.glass_org_embeddings,
            self.index,
            k=K,
            chunk_size=INDEX_SEARCH_CHUNK_SIZE,
        )

        index_id_to_cluster_lookup = dict(
            (i, c) for i, (_, c) in enumerate(self.clusters)
        )

        knn_clusters = assign_knn_cluster(
            knn,
            index_id_to_cluster_lookup,
            REASSIGN_AGG_FUNC,
            MIN_REASSIGN_DIST,
        )

        index_id_to_glass_id_lookup = dict(enumerate(self.glass_org_ids))
        self.clusters_reassigned = [
            (index_id_to_glass_id_lookup[i], c) for i, c in knn_clusters
        ]
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
