"""Fetch and save lookups between SIC taxonomy codes and names"""
from typing import Dict, List, Union

from metaflow import FlowSpec, project, step

## Declare constants in all caps at the top.
URL = (
    "http://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/"
    "ukstandardindustrialclassificationofeconomicactivities/uksic2007/"
    "sic2007summaryofstructurtcm6.xls"
)

# Type aliases
field_name = str
code_name = str
code = str


@project(name="industrial_taxonomy")
class Sic2007Structure(FlowSpec):
    """SIC 2007 taxonomy structure.

    Includes extra subclasses used by companies house:
    - 74990 - Non-trading company
    - 98000 - Residents property management
    - 99999 - Dormant company

    Attributes:
        url: Source excel file
        records: Taxonomy structure as list of records.
        section_lookup: Lookup from code to name
        division_lookup: Lookup from code to name
        group_lookup: Lookup from code to name
        class_lookup: Lookup from code to name
        subclass_lookup: Lookup from code to name
    """

    url: str
    records: List[Dict[field_name, Union[code, code_name]]]
    section_lookup: Dict[code, code_name]
    division_lookup: Dict[code, code_name]
    group_lookup: Dict[code, code_name]
    class_lookup: Dict[code, code_name]
    subclass_lookup: Dict[code, code_name]

    @step
    def start(self):
        """Fetch Excel spreadsheet containing taxonomy structure from ONS website."""
        from utils import get, excel_to_df

        self.url = URL

        self._raw_data = excel_to_df(get(self.url))

        self.next(self.transform)

    @step
    def transform(self):
        """Transform data.

        Make implicit entries explicit at row-level (fill), normalise SIC
        codes, and add extra codes specific to Companies House.
        """
        from utils import companies_house_extras, fill, normalise_codes

        self._data = (
            self._raw_data.pipe(fill)
            .pipe(normalise_codes)
            .append(companies_house_extras())
        )

        self.next(self.end)

    @step
    def end(self):
        """Generate lookups at each level in the SIC hierarchy."""
        from utils import LEVELS

        for level in LEVELS:
            lookup: Dict[code, code_name] = (
                self._data[[level, f"{level}_name"]]
                .set_index(level)
                .dropna()
                .to_dict()[f"{level}_name"]
            )
            setattr(self, f"{level}_lookup", lookup)

        self.records = self._data.to_dict(orient="records")


if __name__ == "__main__":

    Sic2007Structure()
