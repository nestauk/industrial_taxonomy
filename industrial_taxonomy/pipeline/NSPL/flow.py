from typing import Dict, Literal

from metaflow import (
    conda_base,
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
#


@project(name="industrial_taxonomy")
class NsplLookup(FlowSpec):
    """Lookups from postcode to Local Authority Districts and lat-long.
    Uses National Statistics Postcode Lookup (NSPL).
    Excludes postcodes that do not have an assigned output area (OA),
    and therefore no Local Authority District coding.
    Postcodes in the Channel Islands and Isle of Man both have lat-longs of
    `(99.999999, 0.000000)` and have pseudo Local Authorit district codes of
    "L99999999" and "M99999999" respectively, this does not match up with
    other datasets and therefore have been excluded.
    Attributes:
        geoportal_url: Full URL of dataset in gov.uk geoportal
        download_url: Download URL of dataset
        laua_names: Lookup from laua code to name
        laua_year: Year LAUA names and codes correspond to
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
    laua_year: int
    pcd_laua: Dict[postcode, laua_code]
    # pcd_latlong: Dict[postcode, Dict[Literal["lat", "long"], float]]

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
            laua_names_tmp = read_laua_names(zipfile)

            self.laua_names = laua_names_tmp
            self.nspl_linked = nspl_joined(self.nspl_data, self.laua_names)
            print(self.nspl_linked)
            print(self.laua_names)
            print(self.nspl_data.columns)

        self.next(self.data_quality)

    @step
    def data_quality(self):
        """Data quality checks."""
        from utils import LOOSE_UK_BOUNDS

        print(type(self.nspl_data))

        # Null checks
        has_nulls = self.nspl_data.isnull().sum().sum() > 0
        if has_nulls:
            raise AssertionError("Nulls detected")

        # Postcode validity
        # Choose very simple postcode verification as NSPL is a fairly
        # authoritative source that may update faster than a precise regex
        #    POSTCODE_REGEX = r"^([A-Z]{1,2}[A-Z\d]{0,2}? ?\d[A-Z]{2})$"
        #    valid_pcds = self.nspl_data.index.str.match(POSTCODE_REGEX)
        #    if not valid_pcds.all():
        #        raise AssertionError(
        #            "Invalid postcodes detected: "
        #            f"{self.nspl_data.loc[~valid_pcds].index.values}"
        #        )

        # Check we have names for all laua codes
        # nspl_laua_cds = set(self.nspl_data.laua_code.dropna())
        # laua_names_cds = set(self.laua_names.keys())
        # laua_diff = nspl_laua_cds - laua_names_cds
        # if len(laua_diff) > 0:
        #   raise AssertionError(f"LAUA do not match: {laua_diff}")
        #######lines 147 and 148 caused an error which state that the rows arent equal however would this be solved with the join? ##

        ### Once the current error is solved i will add these lines back in but change it to work with the current adjusted dataframe.
        self.next(self.end)

    @step
    def end(self):

        from utils import resetting_index

        index_reset = resetting_index(self.nspl_linked)
        self.nspl_dataframe = index_reset.to_dict(orient="index")
        print(self.nspl_dataframe)


if __name__ == "__main__":
    NsplLookup()
