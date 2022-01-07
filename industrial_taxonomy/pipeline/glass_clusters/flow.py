"""Flow to cluster glass companies into text clusters
"""

from typing import List, Tuple, Union
from metaflow import FlowSpec, project, step, Parameter
from industrial_taxonomy.pipeline.glass_clusters.hSBM_Topicmodel.sbmtm import sbmtm


@project(name="industrial_taxonomy")
class ClusterGlass(FlowSpec):
    """Create sectoral corpora of glass descriptions
    and cluster them into text sectors

    Attributes:
        min_sector_size: minimum sector size
        cluster_n: number of companies to cluster
        sector_corpora: tokenised descriptions by sector
        text_sectors: lookup between company ids and clusters
        models: topsbm models we will use for robustness tests etc
    """

    min_sector_size: int
    cluster_n: Union[str, int]
    sectors: List
    corpora: List
    text_sectors: List[Tuple[int, str]]
    model: List[sbmtm]

    min_sector_size = Parameter(
        "min-sector-size", help="minimum sector size", default=1000
    )

    test_mode = Parameter("test-mode", help="Run in test mode", default=True)

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

        self.sectors_corpora = [
            [sector, corp]
            for sector, corp in zip(
                list(sector_corpora.keys()), sector_corpora.values()
            )
        ]

        if self.test_mode is True:
            self.sectors_corpora = self.sectors_corpora[-3:]

        self.next(self.cluster_glass_descriptions, foreach="sectors_corpora")

    @step
    def cluster_glass_descriptions(self):
        """Cluster glass descriptions using topsbm"""

        from industrial_taxonomy.pipeline.glass_clusters.topic_utils import (
            fit_model_sector,
            extract_clusters,
        )

        sector = self.input[0]
        model = fit_model_sector(self.input[1])
        clusters = extract_clusters(model, sector, 1)
        self.outputs = [sector, model, clusters]

        self.next(self.join)

    @step
    def join(self, inputs):
        """Combine previous results"""
        from itertools import chain

        self.models = {input.outputs[0]: input.outputs[1] for input in inputs}
        self.clusters = list(chain(*[input.outputs[2] for input in inputs]))

        self.next(self.end)

    @step
    def end(self):
        """Save outputs"""
        pass


if __name__ == "__main__":
    ClusterGlass()
