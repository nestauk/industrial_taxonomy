import logging
import numpy as np

from metaflow import (
    conda,
    current,
    FlowSpec,
    Parameter,
    project,
    step,
    batch,
    pip,
)

from utils import (
    find_knn,
    generate_index,
    get_clusters_embeddings,
    intify_clusters,
    reassign_clustered,
    assign_non_clustered,
    get_non_clusters_embeddings,
    knn_output,
)


logger = logging.getLogger(__name__)

K = 20
INDEX_SEARCH_CHUNK_SIZE = 1_000
REASSIGN_AGG_FUNC = np.mean


@project(name="industrial_taxonomy")
class ClusterReassignFlow(FlowSpec):
    """Reassign companies to clusters based on their K nearest neighbours.

    Attributes:
        assigned_rest: Glass organisation IDs that were not clustered into text
            sectors and the text sectors that they have been assigned using a K
            nearest neighbours search.
        clusters: Glass organisation IDs and their cluster labels.
        entropy_after:
        entropy_before:
        reassigned: Glass organisation IDs and their reassigned
            cluster labels found using a K nearest neighbours search.
        k: Number of nearest neighbours used to perform search.
        knn:
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
        """Load data, convert org IDs to integers and truncate if in test mode."""
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
            # "faiss": "1.7.2"
        },
    )
    @step
    def reassign_companies(self):
        """Queries FAISS index for the K nearest neighbours to each company
        given an embedding of its Glass description. Assigns a cluster label
        based on the labels of the nearest neighbours.
        """
        import faiss

        from industrial_taxonomy.getters.glass_house import (
            description_embeddings,
            embedded_org_ids,
        )

        glass_embeddings = description_embeddings()
        glass_org_ids = embedded_org_ids()

        faiss.normalize_L2(glass_embeddings)

        self.org_ids = dict()
        self.knn = dict()
        self.knn_org_ids = dict()
        self.knn_original_text_sectors = dict()
        self.knn_sims = dict()
        self.assigned_text_sector = dict()
        self.original_text_sector = dict()
        self.knn_assigned_text_sectors = dict()
        self.knn_text_sector_agg = dict()
        self.knn_text_sector_agg_sims = dict()

        self.org_ids_rest = dict()
        self.knn_org_ids_rest = dict()
        self.knn_original_text_sectors_rest = dict()
        self.knn_sims_rest = dict()
        self.assigned_text_sector_rest = dict()
        self.original_text_sector_rest = dict()
        self.knn_assigned_text_sectors_rest = dict()
        self.knn_text_sector_agg_rest = dict()
        self.knn_text_sector_agg_sims_rest = dict()
        for param, clusters in self.clusters.items():
            embeddings, embedding_locs = get_clusters_embeddings(
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
                embeddings,
                faiss_index,
                k=self.k,
                chunk_size=INDEX_SEARCH_CHUNK_SIZE,
            )

            logger.info(
                (
                    "Iteratively reassigning companies to nearest neighbour "
                    f"clusters for {param}"
                )
            )
            reassigned, agg_clusters, agg_cluster_sims = reassign_clustered(
                knn,
                clusters,
                min_sim_threshold=0.6,
                n_iter=20,
                epsilon=0.05,
            )

            output = knn_output(
                np.array(glass_org_ids)[embedding_locs],
                clusters,
                reassigned,
                knn,
            )

            self.knn_org_ids[param] = output["knn_org_ids"]
            self.knn_original_text_sectors[param] = output["knn_original_text_sectors"]
            self.knn_sims[param] = output["knn_sims"]
            self.assigned_text_sector[param] = output["assigned_text_sector"]
            self.original_text_sector[param] = output["original_text_sector"]
            self.knn_assigned_text_sectors[param] = output["knn_assigned_text_sectors"]
            self.org_ids[param] = output["org_ids"]
            self.knn_text_sector_agg[param] = agg_clusters
            self.knn_text_sector_agg_sims[param] = agg_cluster_sims

            logger.info(f"Assigning left over companies for {param}")
            embeddings_rest, locs_rest = get_non_clusters_embeddings(
                embedding_locs,
                glass_embeddings,
            )
            # nrows = 20_000 if self.test_mode and not current.is_production else None
            # locs_rest = locs_rest[:nrows]
            # embeddings_rest = embeddings_rest[:nrows]

            logger.info(f"Finding left over nearest neighbours for {param}")
            knn_rest = find_knn(
                embeddings_rest,
                faiss_index,
                k=self.k,
                chunk_size=INDEX_SEARCH_CHUNK_SIZE,
            )

            logger.info(f"Assigning left over companies for {param}")
            (
                assigned_rest,
                agg_clusters_rest,
                agg_cluster_sims_rest,
            ) = assign_non_clustered(
                knn_rest,
                glass_org_ids,
                locs_rest,
                clusters,
                min_sim_threshold=0.6,
            )
            # self.assigned_rest[param] = assigned_rest

            output_rest = knn_output(
                np.array(glass_org_ids)[locs_rest],
                clusters,
                reassigned,
                knn_rest,
                assigned_rest,
                rest=True,
            )

            self.knn_org_ids_rest[param] = output_rest["knn_org_ids"]
            self.knn_original_text_sectors_rest[param] = output_rest[
                "knn_original_text_sectors"
            ]
            self.knn_assigned_text_sectors_rest[param] = output_rest[
                "knn_assigned_text_sectors"
            ]
            self.knn_sims_rest[param] = output_rest["knn_sims"]
            self.assigned_text_sector_rest[param] = output_rest["assigned_text_sector"]
            self.original_text_sector_rest[param] = output_rest["original_text_sector"]
            self.org_ids_rest[param] = output_rest["org_ids"]
            self.knn_text_sector_agg_rest[param] = agg_clusters_rest
            self.knn_text_sector_agg_sims_rest[param] = agg_cluster_sims_rest

            self.knn[param] = knn

        self.next(self.entropy)

    @conda(
        python="3.8",
        libraries={
            "scikit-bio": "0.5.6",
        },
    )
    @step
    def entropy(self):
        """Calculates Shannon diversity index of clusters of the K nearest
        neighbours before and after reassignment."""

        from skbio.diversity.alpha import shannon
        from sklearn.feature_extraction.text import CountVectorizer

        def do_nothing(text):
            """Dummy tokenizer and preprocessor."""
            return text

        cv = CountVectorizer(tokenizer=do_nothing, preprocessor=do_nothing)

        self.entropy_before = {}
        self.entropy_after = {}
        for param, knn in self.knn.items():
            logger.info(f"Calculating entropy for companies in {param}")
            index_id_cluster_lookup = np.array(self.original_text_sector[param])
            # index_id_cluster_lookup = np.array(
            #     [c[0] for c in self.original_text_sector[param]]
            # )

            knn_ids = np.array([k[0][1:] for k in knn])
            knn_cluster_labels = index_id_cluster_lookup[knn_ids]

            counts_before = cv.fit_transform(knn_cluster_labels).todense()
            entropy_before = [shannon(np.array(c)[0]) for c in counts_before]
            self.entropy_before[param] = list(
                # zip([c[1] for c in self.clusters[param]], entropy_before)
                zip(self.org_ids[param], entropy_before)
            )

            index_id_reassigned_lookup = np.array(
                self.assigned_text_sector[param]
                # self.reassigned[param]["assigned_text_sector"]
            )
            knn_reassigned_labels = index_id_reassigned_lookup[knn_ids]

            counts_after = cv.fit_transform(knn_reassigned_labels).todense()
            entropy_after = [shannon(np.array(c)[0]) for c in counts_after]
            self.entropy_after[param] = list(
                zip(self.org_ids[param], entropy_after)
                # zip([c[1] for c in self.clusters[param]], entropy_after)
            )

        self.next(self.end)

    @step
    def end(self):
        """No-op."""
        pass


if __name__ == "__main__":
    ClusterReassignFlow()
