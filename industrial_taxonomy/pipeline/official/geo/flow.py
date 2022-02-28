from metaflow import FlowSpec, step, Parameter, project


BOUND_URL = (
    "https://opendata.arcgis.com/datasets/db23041df155451b9a703494854c18c4_0.geojson"
)


@project(name="industrial_taxonomy")
class FetchBound(FlowSpec):
    """Fetch a boundary

    Attributes:
        boundary_url: url to fetch the data from
        boundary: boundary file
    """

    boundary_url: str
    boundary: dict

    boundary_url = Parameter(
        "boundary-url", help="url for the boundary file", default=BOUND_URL
    )

    @step
    def start(self):
        """start the flow"""

        self.next(self.fetch_boundary)

    @step
    def fetch_boundary(self):
        """Fetch the boundary"""
        from utils import fetch_boundary

        self.boundary = fetch_boundary(self.boundary_url)

        self.next(self.end)

    @step
    def end(self):
        """End the flow, nothing to see here"""


if __name__ == "__main__":
    FetchBound()
