"""Flow to collect NOMIS data with the exception of BRES"""
from io import BytesIO
import pandas as pd
from metaflow import FlowSpec, project, step
from typing import List, Dict, Union

_APS_URL = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv?"
    "geography=1811939329...1811939332,1811939334...1811939336,1811939338..."
    "1811939497,1811939499...1811939501,1811939503,"
    "1811939505...1811939507,1811939509...1811939517,"
    "1811939519,1811939520,1811939524...1811939570,1811939575...1811939599,"
    "1811939601...1811939628,1811939630...1811939634,1811939636...1811939647"
    "1811939649,1811939655...1811939664,1811939667...1811939680"
    ",1811939682,1811939683,1811939685,1811939687...1811939704,1811939707,1811939708"
    ",1811939710,1811939712...1811939717,1811939719,1811939720,1811939722..."
    "1811939730&date=2019-12&variable=18,45,290,335,344"
    "&measures=20599,21001,21002,21003"
)

_ASHE_URL = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_30_1.data.csv?"
    "geography=1811939329...1811939332,1811939334...1811939336,1811939338..."
    "1811939497,1811939499...1811939501,1811939503,1811939505..."
    "1811939507,1811939509...1811939517,1811939519,1811939520,1811939524..."
    "1811939570,1811939575...1811939599,1811939601...1811939628,1811939630..."
    "1811939634,1811939636...1811939647,1811939649,1811939655...1811939664,"
    "1811939667...1811939680,1811939682,1811939683,1811939685,1811939687..."
    "1811939704,1811939707,1811939708,1811939710,1811939712...1811939717,"
    "1811939719,1811939720,1811939722...1811939730&date=latest&sex=8&item="
    "2&pay=7&measures=20100,20701"
)
_APS_PARAMS = {
    "indicator_name": "Variable",
    "value_column": "VARIABLE_NAME",
    "source": "aps",
}
_ASHE_PARAMS = {"indicator_name": "Value", "value_column": "PAY_NAME", "source": "ashe"}


@project(name="industrial_taxonomy")
class NomisTables(FlowSpec):
    """Flow to collect APS / ASHE data from NOMIS

    Attributes:
        url_list: list of urls to collect and process
        params_list: list of parameters to use when collecting and processing the data
        nomis_table: clean dataset combining all nomis data
        nomis_dict: dictionary with the nomis data
    """

    # Type hints
    url_list: list
    nomis_table: pd.DataFrame
    nomis_dict: List[Dict[str, Union[str, float]]]

    @step
    def start(self):
        """Read the urls and parameters for fetching and processing"""

        self.urls = [_APS_URL, _ASHE_URL]
        self.params = [_APS_PARAMS, _ASHE_PARAMS]

        self.next(self.fetch_process)

    @step
    def fetch_process(self):
        """Fetch and process the data"""

        import pandas as pd
        from utils import process_nomis
        from industrial_taxonomy.pipeline.official.utils import get

        self.nomis_table = pd.concat(
            [
                process_nomis(pd.read_csv(BytesIO(get(url).content)), **params)
                for url, params in zip(self.urls, self.params)
            ]
        )

        self.next(self.end)

    @step
    def end(self):
        """Save nomis table as a dict"""
        self.nomis_dict = self.nomis_table.to_dict(orient="records")


if __name__ == "__main__":
    NomisTables()
