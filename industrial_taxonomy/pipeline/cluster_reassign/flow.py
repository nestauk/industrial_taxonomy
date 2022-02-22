import logging
import numpy as np

from metaflow import (
    conda,
    current,
    FlowSpec,
    Parameter,
    project,
    step,
    Run,
    batch,
)

from industrial_taxonomy.getters.glass_house import (
    description_embeddings,
    embedded_org_ids,
)

from utils import (
    find_knn,
    assign_knn_cluster,
    generate_index,
    get_clusters_embeddings,
    add_original_clusters,
    intify_clusters,
)


logger = logging.getLogger(__name__)

K = 20
INDEX_SEARCH_CHUNK_SIZE = 1_000
REASSIGN_AGG_FUNC = np.mean


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

    @conda(
        libraries={
            "graph-tool": "2.44",
        },
    )
    @step
    def start(self):
        """Load data and truncate if in test mode."""
        from industrial_taxonomy.getters.text_sectors import text_sectors

        self.clusters = text_sectors()
        if self.test_mode and not current.is_production:
            min_clusters_len = min([len(c) for c in self.clusters.values()])
            self.clusters = {
                k: v for k, v in self.clusters.items() if len(v) == min_clusters_len
            }

        for param, clusters in self.clusters.items():
            self.clusters[param] = intify_clusters(clusters)

        self.k = K

        self.next(self.reassign_companies)

    @batch(
        # Queue gives p3.2xlarge
        queue="job-queue-GPU-nesta-metaflow",
        image="metaflow-pytorch",
        gpu=1,
        memory=60000,
        cpu=8,
    )
    @conda(
        python="3.8",
        libraries={
            "faiss-gpu": "1.7.2",
        },
    )
    @step
    def reassign_companies(self):
        """Queries FAISS index for the K nearest neighbours to each company
        given an embedding of its Glass description. Assigns a cluster label
        based on the labels of the nearest neighbours.
        """
        import faiss

        nrows = 20_000 if self.test_mode and not current.is_production else None

        glass_embeddings = description_embeddings()
        glass_org_ids = embedded_org_ids()

        faiss.normalize_L2(glass_embeddings)

        self.clusters_reassigned = {}
        for param, clusters in self.clusters.items():
            embeddings = get_clusters_embeddings(
                clusters,
                glass_org_ids,
                glass_embeddings,
            )
            logger.info(f"Generating FAISS index for {param}")
            faiss_index = generate_index(
                faiss.IndexFlatIP,
                embeddings,
            )
            logger.info(f"Finding nearest neighbours for {param}")
            knn = find_knn(
                glass_embeddings[:nrows],
                faiss_index,
                k=self.k,
                chunk_size=INDEX_SEARCH_CHUNK_SIZE,
            )

            logger.info(f"Extracting nearest neighbour clusters for {param}")
            index_id_to_cluster_lookup = dict(enumerate(c[0] for c in clusters))
            knn_clusters = assign_knn_cluster(
                knn,
                index_id_to_cluster_lookup,
                REASSIGN_AGG_FUNC,
                glass_org_ids,
            )

            knn_clusters = add_original_clusters(
                knn_clusters,
                clusters,
            )

            self.clusters_reassigned[param] = knn_clusters
        self.next(self.end)

    @step
    def end(self):
        """No-op."""
        pass


if __name__ == "__main__":
    ClusterReassignFlow()
