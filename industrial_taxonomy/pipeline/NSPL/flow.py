from typing import Dict

from metaflow import (
    current,
    FlowSpec,
    Parameter,
    project,
    step,
)


GEOPORTAL_URL_PREFIX = "https://geoportal.statistics.gov.uk/datasets"

# Type aliases
postcode = str
laua_name = str
laua_code = str


@project(name="industrial_taxonomy")
class NsplLookup(FlowSpec):
    """Lookups from postcode to Local Authority Districts and lat-long.
    Uses National Statistics Postcode Lookup (NSPL).
    Excludes postcodes that do not have an assigned output area (OA),
    and therefore no Local Authority District coding.
    Postcodes in the Channel Islands and Isle of Man both have lat-longs of
    `(99.999999, 0.000000)` and have psuedo Local Authority district codes of
    "L99999999" and "M99999999" respectively, this does not match up with
    other datasets and therefore have been excluded.
    Attributes:
        geoportal_url: Full URL of dataset in gov.uk geoportal
        download_url: Download URL of dataset
        laua_names: Lookup from laua code to name
        pcd_laua: Lookup from postcode to LAUA
        pcd_latlong: Lookup from postcode to latitude and longitude
    """

    geoportal_dataset = Parameter(
        "geoportal-dataset",
        help=f"Name of dataset in URL at {GEOPORTAL_URL_PREFIX}",
        required=True,
        type=str,
        default="national-statistics-postcode-lookup-february-2021",
    )

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode",
        type=bool,
        default=lambda _: not current.is_production,
    )

    geoportal_url: str
    download_url: str
    laua_names: Dict[laua_code, laua_name]
    pcd_laua: Dict[postcode, laua_code]

    @step
    def start(self):
        """Get dynamically rendered download link."""
        from utils import (
            chrome_driver,
            find_download_url,
        )

        self.geoportal_url = f"{GEOPORTAL_URL_PREFIX}/{self.geoportal_dataset}"
        with chrome_driver() as driver:
            self.download_url = find_download_url(driver, self.geoportal_url)

        self.next(self.get_data)

    @step
    def get_data(self):
        """Download zipped NSPL collection, extracting main lookup & LAUA names."""
        import logging

        import requests_cache

        from utils import (
            download_zip,
            filter_nspl_data,
            read_nspl_data,
            read_laua_names,
            nspl_joined,
        )

        if self.test_mode and not current.is_production:
            nrows = 1_000
            logging.warning(f"TEST MODE: Constraining to first {nrows} rows!")
        else:
            nrows = None

        if not current.is_production:
            requests_cache.install_cache("nspl_zip_cache")

        with download_zip(self.download_url) as zipfile:

            # Load main postcode lookup
            self.nspl_data = read_nspl_data(zipfile, nrows).pipe(filter_nspl_data)
            # LAUA lookup from codes to names
            self.laua_names = read_laua_names(zipfile)
            # joining nspl_names with nspl_data on corresponding laua_code
            self.nspl_linked = nspl_joined(self.nspl_data, self.laua_names)

        self.next(self.data_quality)

    @step
    def data_quality(self):
        """Data quality checks."""

        # Null checks
        has_nulls = self.nspl_data.isnull().sum().sum() > 0
        if has_nulls:
            raise AssertionError("Nulls detected")

        # Postcode validity
        # Choose very simple postcode verification as NSPL is a fairly
        # authoritative source that may update faster than a precise regex
        postcode_regex = r"^([A-Z]{1,2}[A-Z\d]{0,2}? ?\d[A-Z]{2})$"
        valid_pcds = self.nspl_data.pcds.str.match(postcode_regex)
        if not valid_pcds.all():
            raise AssertionError(
                "Invalid postcodes detected: "
                f"{self.nspl_data.loc[~valid_pcds].pcds.values}"
            )

        # Check we have names for all laua codes
        nspl_laua_cds = set(self.nspl_data.index.dropna())
        laua_names_cds = set(self.laua_names.LAD20CD)
        laua_diff = nspl_laua_cds - laua_names_cds
        if len(laua_diff) > 0:
            raise AssertionError(f"LAUA do not match: {laua_diff}")

        self.next(self.end)

    @step
    def end(self):
        """Setting index to pcds and converting dataframe to dict where keys are set to the current index. self.nspl_datframe printed as
        {'AB1 0AA': {'laua_code': 'S12000033', 'lat': 57.101474, 'long': -2.242851, 'laua_name': 'Aberdeen City'},
        'AB1 0AB': {'laua_code': 'S12000033', 'lat': 57.102554, 'long': -2.246308, 'laua_name': 'Aberdeen City'} ... for first two pcds"""

        self.nspl_dataframe = self.nspl_linked.to_dict(orient="index")


if __name__ == "__main__":
    NsplLookup()
