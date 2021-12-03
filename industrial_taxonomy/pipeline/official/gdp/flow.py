"""Fetch GDP data"""

from typing import Dict, List, Union
from metaflow import FlowSpec, project, step

try:  # Hack for type-hints on attributes
    from pandas import DataFrame
except ImportError:
    pass

GDP_URL = (
    "https://www.ons.gov.uk/file?uri=/economy/grossdomesticproductgdp/datasets/"
    "regionalgrossdomesticproductlocalauthorities/1998to2019/"
    "regionalgrossdomesticproductlocalauthorities.xlsx"
)
# Excel spreadsheets with the data we are interested in
SHEETS = [6, 7]


@project(name="industrial_taxonomy")
class LocalGdpData(FlowSpec):
    """Fetch local GDP (including population and GVA) data from the ONS website

    Attributes:
        url: location of the original file
        pop_clean: population table
        gva_clean: GVA table
        gva: gva dict
        pop: pop dict
    """

    url: str
    gva_clean: "DataFrame"
    pop_clean: "DataFrame"
    pop: List[Dict[str, Union[str, float]]]
    gva: List[Dict[str, Union[str, float]]]

    @step
    def start(self):
        """Fetch the GDP data from the ONS"""
        import pandas as pd
        from industrial_taxonomy.pipeline.official.utils import get

        self.url = GDP_URL
        gdp_table = get(self.url).content

        # Create dfs for the sheets with relevant information (population and GVA)
        self._gva_raw, self._pop_raw = [
            pd.read_excel(gdp_table, sheet_name=sh, skiprows=1) for sh in SHEETS
        ]

        self.next(self.transform)

    @step
    def transform(self):
        """Clean up the data"""
        from industrial_taxonomy.pipeline.official.gdp.utils import process_gdp_table

        self.pop_clean = process_gdp_table(self._pop_raw)
        self.gva_clean = process_gdp_table(self._gva_raw)

        self.next(self.end)

    @step
    def end(self):
        """Save the tables as dicts"""

        for table, name in zip([self.gva_clean, self.pop_clean], ["gva", "pop"]):

            table_dict = table.to_dict(orient="records")
            setattr(self, name, table_dict)


if __name__ == "__main__":
    LocalGdpData()
