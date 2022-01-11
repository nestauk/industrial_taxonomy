"""Flow to cluster glass companies into text clusters
"""

from typing import Dict, List, Tuple
from metaflow import FlowSpec, JSONType, project, step, Parameter
from industrial_taxonomy.pipeline.glass_clusters.hSBM_Topicmodel.sbmtm import sbmtm


@project(name="industrial_taxonomy")
class ClusterGlass(FlowSpec):
    """Create sectoral corpora of glass descriptions
    and cluster them into text sectors

    Attributes:
        min_sector_size: minimum sector size
        assigned_shares = share of companies to assign to cluster
        sectors: sectors we train the model on
        sectors_corpora: tokenised descriptions by sector
        models: topsbm models we will use for robustness / sanity checks etc
        clusters: lookup between company ids and clusters they are assigned to
            clusters are named as {SIC_4 the company belongs to}_{topsbm cluster id}
    """

    min_sector_size: int
    assigned_shares: List[float]
    sectors_corpora: List[Dict[str, Dict[str, List[str]]]]
    sectors: List[str]
    clusters: List[Tuple[int, str]]
    models: Dict[str, Dict[str, sbmtm]]

    min_sector_size = Parameter(
        "min-sector-size", help="minimum sector size", default=1000
    )

    test_mode = Parameter("test-mode", help="Run in test mode", default=True)

    assigned_shares = Parameter(
        "assigned-shares",
        help="share of companies to assign",
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
        from industrial_taxonomy.pipeline.glass_clusters.utils import (
            make_sector_corpora,
        )

        sector_corpora = make_sector_corpora(min_sector_size=self.min_sector_size)

        all_sectors_corpora = [
            [sector, corp]
            for sector, corp in zip(
                list(sector_corpora.keys()), sector_corpora.values()
            )
        ]

        if self.test_mode is True:
            self.sectors_corpora = all_sectors_corpora[:3]
        else:
            self.sectors_corpora = all_sectors_corpora

        self.next(self.start_parallel_clustering)

    @step
    def start_parallel_clustering(self):
        """Creates a branch for every assignment share value"""

        self.next(self.cluster_glass_descriptions, foreach="assigned_shares")

    @step
    def cluster_glass_descriptions(self):
        """Cluster glass descriptions using topsbm"""

        from industrial_taxonomy.pipeline.glass_clusters.topic_utils import (
            fit_model_sector,
            extract_clusters,
        )

        sectors = [corp[0] for corp in self.sectors_corpora]
        models = [fit_model_sector(corp[1]) for corp in self.sectors_corpora]
        clusters = [
            extract_clusters(mod, sect, docs_to_assign=self.input)
            for sect, mod in zip(sectors, models)
        ]

        self.outputs = [sectors, models, clusters]

        self.next(self.join_clusters)

    @step
    def join_clusters(self, inputs):
        """Combine branched cluster assignments"""
        from itertools import chain

        # Sectors are the same for all assignment strategies
        self.sectors = set(chain(*[input.outputs[0] for input in inputs]))

        self.models = {
            f"assigned_{str(param)}": {
                sector: model
                for sector, model in zip(input.outputs[0], input.outputs[1])
            }
            for param, input in zip(self.assigned_shares, inputs)
        }

        self.clusters = {
            f"assigned_{str(param)}": list(chain(*[cl for cl in input.outputs[2]]))
            for param, input in zip(self.assigned_shares, inputs)
        }

        self.next(self.end)

    @step
    def end(self):
        """Save outputs"""
        pass


if __name__ == "__main__":
    ClusterGlass()
