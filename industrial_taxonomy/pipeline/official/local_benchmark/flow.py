from typing import Dict, List
from metaflow import FlowSpec, Parameter, project, step, current
from pandas import DataFrame


@project(name="industrial_taxonomy")
class SecondaryIndicators(FlowSpec):
    """Flow that collects and cleans up secondary data

    Attributes:
        nuts_lad_lookup: lookup between nuts codes and lad codes
        beis_indicators: data source and name for BEIS indicators
        ons_raw: ONS level up raw data
        beis_raw: Nesta / beis raw innovation dashboard data
        ons_clean: ONS level up processed and standardised data
        bveis_clean: BEIS innovation processed and standardised
        secondary_table: Combined secondary data
        beis_schema: Schemas for the nesta/beis data
    """

    nuts_lad_lookup: Dict
    beis_indicators: List
    ons_raw: DataFrame
    beis_raw: List
    ons_clean: DataFrame
    beis_clean: DataFrame
    secondary_table: DataFrame
    beis_schema: DataFrame

    test_mode = Parameter(
        "test-mode",
        help="Run in test mode",
        default=True,
    )

    @step
    def start(self):
        """Start the flow"""

        from utils import nuts_lad_lookup, BEIS_INDICATORS

        self.nuts_lad_lookup = nuts_lad_lookup()

        if self.test_mode is True and not current.is_production:
            self.beis_indicators = BEIS_INDICATORS[:5]

        else:
            self.beis_indicators = BEIS_INDICATORS

        self.next(self.fetch_ons)

    @step
    def fetch_ons(self):
        """Fetch ONS data"""

        from utils import fetch_ons

        self.ons_raw = fetch_ons()

        self.next(self.process_ons)

    @step
    def process_ons(self):
        """Process ONS data"""

        from utils import standardise_ons

        self.ons_clean = standardise_ons(self.ons_raw)

        self.next(self.fetch_beis)

    @step
    def fetch_beis(self):
        """Fetch BEIS data"""

        from utils import fetch_beis_table, fetch_schema

        self.beis_raw = [
            fetch_beis_table(indicator) for indicator in self.beis_indicators
        ]

        self.beis_schema = {
            indicator: fetch_schema(indicator) for indicator in self.beis_indicators
        }

        self.next(self.process_beis)

    @step
    def process_beis(self):
        """process BEIS data"""

        from utils import standardise_beis, indicator_name
        import pandas as pd

        self.beis_clean = pd.concat(
            [
                standardise_beis(
                    beis_table=table,
                    var_name=indicator_name(indicator),
                    nuts_lad_lookup=self.nuts_lad_lookup,
                )
                for table, indicator in zip(self.beis_raw, self.beis_indicators)
            ]
        )

        self.next(self.merge_data)

    @step
    def merge_data(self):
        """Combine the ONS and BEIS data"""
        import pandas as pd

        self.secondary_table = pd.concat([self.ons_clean, self.beis_clean])

        self.next(self.end)

    @step
    def end(self):
        """End flow, nothing to see here"""


if __name__ == "__main__":
    SecondaryIndicators()
