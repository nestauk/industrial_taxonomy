from typing import Dict, List
from metaflow import FlowSpec, project, step, current, Parameter
import pandas as pd
import logging


@project(name="industrial_taxonomy")
class LocalBenchmark(FlowSpec):
    """This flow evaluates the predictive performance of different
    versions of the taxonomy on a collection of secondary local indicators.

    Attributes:
         text_sector_assgn: text sector assignments by strategy
         sector_spec: Sector specialisation profiles by strategy
         assign_shares: different cluster assignment strategies
         beis: beis benchmarking data
         ons: ons benchmarking data
         results_raw: grid search outputs
         results_table: dataframe with results

    """

    text_sector_assgn: Dict
    sector_spec: List
    assign_shares: List
    secondary: List
    secondary_names: List
    results_raw: List
    results_table: pd.DataFrame

    test_mode = Parameter("test-mode", help="run the flow in test model", default=True)

    @step
    def start(self):
        """Read the text sector assignment, ONS and model config data"""

        self.assign_shares = [
            "assigned_500",
            "assigned_100",
            "assigned_0.5",
            "assigned_10",
            "assigned_all",
        ]

        self.secondary_names = ["ons", "beis"]

        if (self.test_mode) is True and current.is_production is False:
            self.param_grid = {
                "min_samples_split": [2, 10],
                "min_samples_leaf": [10],
            }

        # TODO: Read this from a yaml file
        else:
            self.param_grid = {
                "min_samples_split": [2, 10, 50],
                "min_samples_leaf": [1, 5, 10],
                "max_depth": [3, 5, 10],
                "max_features": ["auto", "sqrt"],
                "loss": ["squared_error", "huber"],
            }

        from utils import make_sector_assignments
        from utils import (
            glass_la,
            geo_distribution,
            ASSGN_KEYS,
            process_benchmark_data,
            lad_code_name_lookup,
        )
        from industrial_taxonomy.getters.official import local_benchmarking

        glass_la_lookup = glass_la()

        self.text_sector_assgn = make_sector_assignments()
        self.sector_distrs = [
            geo_distribution(self.text_sector_assgn, reassgn_strat, glass_la_lookup)
            for reassgn_strat in ASSGN_KEYS
        ]
        ons, beis = process_benchmark_data(local_benchmarking(), lad_code_name_lookup())

        self.secondary_data = [ons, beis]

        self.next(self.predictive_modelling, foreach="secondary_data")

    @step
    def predictive_modelling(self):
        """Predictive modelling of secondary data using the sector distributions"""

        from industrial_taxonomy.analysis.modelling import local_predictive
        from sklearn.ensemble import GradientBoostingRegressor

        self.modelling_results = [
            local_predictive(
                self.input,
                distr,
                GradientBoostingRegressor,
                self.param_grid,
                test=self.test_mode,
            )
            for distr in self.sector_distrs
        ]

        self.next(self.join_results)

    @step
    def join_results(self, inputs):
        """Combine results"""

        from modelling import modelling_results

        self.merge_artifacts(inputs, include=["secondary_names", "assign_shares"])

        self.results_raw = [res.modelling_results for res in inputs]

        self.results_table = pd.concat(
            [
                pd.concat(
                    [
                        modelling_results(result, strategy, "gboost")
                        for result, strategy in zip(sec_results, self.assign_shares)
                    ]
                ).assign(name=sec_name)
                for sec_results, sec_name in zip(self.results_raw, self.secondary_names)
            ]
        ).reset_index(drop=True)

        self.next(self.end)

    @step
    def end(self):
        """no op"""


if __name__ == "__main__":
    LocalBenchmark()
