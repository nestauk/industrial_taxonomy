from _typeshed import StrOrBytesPath
from typing import List

from metaflow import (
    conda,
    conda_base,
    current,
    FlowSpec,
    Parameter,
    project,
    step,
    Run,
    resources,
)

from industrial_taxonomy.getters.glass_embed import (
    glass_description_embeddings,
    org_ids,
)
from industrial_taxonomy.getters.cluster import (
    cluster_ids,
)  # TODO change this to match Juan's code


@conda_base(python="3.8")
@project(name="industrial_taxonomy")
class ClusterReassignFlow(FlowSpec):
    """Reassign companies to clusters based on their K nearest neighbours.

    Attributes:
    """

    k = Parameter(
        "k",
        help="Number of nearest neighbours to use.",
        type=int,
        default=50,
    )

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    @step
    def start(self):
        """Load data"""
        self.glass_org_embeddings = glass_description_embeddings()
        self.glass_org_ids = org_ids()
        self.clusters = glass_cluster_ids()
        pass

    @conda(libraries={"pytorch::faiss": "1.7.2"})
    @step
    def generate_index(self):
        """Generates a FAISS index"""
        import faiss

        dimensions = self.glass_org_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimensions)
        index.add(self.glass_org_embeddings)

        self.next(self.reassign_companies)

    def reassign_companies(self):
        """Queries FAISS index"""
        pass

    def end(self):
        pass
