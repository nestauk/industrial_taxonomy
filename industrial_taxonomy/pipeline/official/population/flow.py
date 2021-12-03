"""Flow to fetch population data"""

from metaflow import FlowSpec, project, step

POP_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
    "populationandmigration/populationestimates/"
    "datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/"
    "mid2020/ukpopestimatesmid2020on2021geography.xls"
)


@project(name="industrial_taxonomy")
class PopEstimateData(FlowSpec):
    """Fetch and process population estimate data from the ONS website

    Attributes:
        url: location of the original file
        clean_pop_est: clean table
        pop_test: population est dict
    """

    @step
    def start(self):
        """Fetch population data from the ONS"""

        import pandas as pd
        from industrial_taxonomy.pipeline.official.utils import get

        self.url = POP_URL
        self._raw_pop_test = pd.read_excel(
            get(self.url).content, sheet_name=6, skiprows=7
        )

        self.next(self.transform)

    @step
    def transform(self):
        """Transform the population table"""

        from industrial_taxonomy.pipeline.official.population.utils import clean_popest

        self.clean_popest = clean_popest(self._raw_pop_test)

        self.next(self.end)

    @step
    def end(self):
        """Save the population estimate as a dict"""

        self.pop_est = self.clean_popest.set_index("geo_code")["pop_2020"].to_dict()


if __name__ == "__main__":
    PopEstimateData()
