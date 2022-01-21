"""Flow to cluster glass companies into text clusters
"""
from typing import Dict, List, Tuple
from metaflow import FlowSpec, JSONType, current, project, step, Parameter
from industrial_taxonomy.pipeline.glass_clusters.sbmtm import sbmtm


@project(name="industrial_taxonomy")
class ClusterGlass(FlowSpec):
    """We select sectors above a minimum doc (company) size and
    cluster the company inside them based on their descriptions using
    the stochastic block-model topic model.

    Attributes:
        min_sector_size: minimum sector size
        assigned_shares: number / share of companies to assign to cluster
        sectors: sectors we train the model on
        sectors_corpora: tokenised descriptions by sector
        models: topsbm models we will use for robustness / sanity checks etc
        clusters: lookup between assignment strategy and cluster assignments.
            The cluster assignments are a list with company ids and
                the clusters they are assigned to.
            Clusters are named as {SIC_4 the company belongs to}_{topsbm cluster id}.
                This is main output we will be using downstream
    """

    min_sector_size: int
    assigned_shares: List[float, str]
    sectors_corpora: Dict[str, Dict[int, List[str]]]
    sectors: List[str]
    clusters: Dict[str, List[Tuple[str, str]]]
    models: Dict[str, sbmtm]

    min_sector_size = Parameter(
        "min-sector-size", help="minimum sector size", default=1000
    )

    test_mode = Parameter(
        "test-mode",
        help="Run in test mode",
        default=True,
    )

    assigned_shares = Parameter(
        "assigned-shares",
        help="number or share of companies to assign to clusters",
        type=JSONType,
        default='[10,100,500,0.5,"all"]',
    )

    @step
    def start(self):
        """Start flow"""

        self.next(self.make_sector_corpora)

    @step
    def make_sector_corpora(self):
        """Make the sector corpora"""
        from toolz import take
        from utils import (
            make_sector_corpora,
        )
        from industrial_taxonomy.getters.glass_house import glass_sic4_lookup

        all_sectors_corpora = make_sector_corpora(
            glass_sic4_lookup(), min_sector_size=self.min_sector_size
        )

        if self.test_mode is True and not current.is_production:
            self.sectors_corpora = dict(take(2, all_sectors_corpora.items()))
        else:
            self.sectors_corpora = all_sectors_corpora

        self.next(self.fit_topic_models)

    @step
    def fit_topic_models(self):
        """Fits the topic model for each sector"""

        from topic_utils import (
            fit_model_sector,
        )

        self.models = {
            sect: fit_model_sector(corp) for sect, corp in self.sectors_corpora.items()
        }
        self.sectors = list(self.sectors_corpora.keys())

        self.next(self.cluster_glass_descriptions, foreach="assigned_shares")

    @step
    def cluster_glass_descriptions(self):
        """Cluster glass descriptions using topsbm
        for each value of the assigned_shares parametre"""

        from topic_utils import extract_clusters
        from itertools import chain

        self.clusters = {
            f"assigned_{self.input}": list(
                chain(
                    *[
                        extract_clusters(
                            self.models[sect], sect, docs_to_assign=self.input
                        )
                        for sect in self.sectors
                    ]
                )
            )
        }

        self.next(self.join_clusters)

    @step
    def join_clusters(self, inputs):
        """Combine branched cluster assignments"""

        self.merge_artifacts(inputs, include=["models", "sectors"])

        self.clusters = {
            assg_n: dict_
            for input in inputs
            for assg_n, dict_ in input.clusters.items()
        }

        self.next(self.end)

    @step
    def end(self):
        """End flow"""
        pass


if __name__ == "__main__":
    ClusterGlass()
