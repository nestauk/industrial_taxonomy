# from typing import Dict, List
from metaflow import FlowSpec, project, step, current, Parameter

# import pandas as pd
import numpy as np
from toolz import pipe
from functools import partial


@project(name="industrial_taxonomy")
class SecondBenchmark(FlowSpec):
    """Calculates silhouette scores on secondary data
    for different clustering strategies

    Attributes:
        spec_profile: specialisation profile for LADs
        search_params: search parameters
        clustering_params: parameters for clustering
        secondary_data: tables with secondary info
            we want to calculate the sil scores for
        outputs: outputs

    """

    test_mode = Parameter("test-mode", help="Runs the flow on test mode", default=True)

    @step
    def start(self):
        """Starts the flow and load the input data"""

        from itertools import product
        from sklearn.cluster import KMeans, AffinityPropagation
        from sklearn.mixture import GaussianMixture
        from industrial_taxonomy.getters.official import local_benchmarking
        from utils import (
            process_benchmark_data,
            lad_code_name_lookup,
            geo_distribution,
            make_sector_assignments,
            glass_la,
        )

        self.spec_profile = pipe(
            make_sector_assignments(only_clustered=False),
            partial(
                geo_distribution, sector_shares="assigned_10", glass_lads=glass_la()
            ),
        ).reset_index(level=1, drop=True)

        self.clustering_params = [
            [KMeans, ["n_clusters", range(20, 50, 3)]],
            [AffinityPropagation, ["damping", np.arange(0.5, 0.91, 0.1)]],
            [GaussianMixture, ["n_components", range(20, 50, 5)]],
        ]

        self.secondary_data = ons, beis = process_benchmark_data(
            local_benchmarking(), lad_code_name_lookup()
        )

        if (self.test_mode is False) & (current.is_production is True):
            self.clustering_params = [
                [KMeans, ["n_clusters", range(20, 50, 3)]],
                [AffinityPropagation, ["damping", np.arange(0.5, 0.91, 0.1)]],
                [GaussianMixture, ["n_components", range(20, 50, 5)]],
            ]
            self.search_params = product(range(5, 55, 5), np.arange(0.4, 1.1, 0.1))

        else:
            self.clustering_params = [
                [KMeans, ["n_clusters", range(20, 50, 10)]],
                [GaussianMixture, ["n_components", range(20, 50, 10)]],
            ]

            self.search_params = product(range(5, 55, 15), np.arange(0.4, 1.1, 0.2))

        self.next(self.make_silhouettes)

    @step
    def make_silhouettes(self):
        """Makes the silhouettes"""
        from utils import silhouette_pipeline

        sil_outputs = []

        for par_set in self.search_params:

            sil_outputs.append(
                silhouette_pipeline(
                    assign=self.spec_profile,
                    pca=par_set[0],
                    comm_resolution=par_set[1],
                    secondary=self.secondary_data,
                    clustering_options=self.clustering_params,
                )
            )
        self.sil_outputs = sil_outputs

        self.next(self.end)

    @step
    def end(self):
        """no op"""


if __name__ == "__main__":
    SecondBenchmark()
