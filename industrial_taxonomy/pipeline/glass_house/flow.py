from typing import List

from metaflow import (
    conda_base,
    current,
    FlowSpec,
    Parameter,
    resources,
    pip,
    project,
    Run,
    step,
    S3,
)
from metaflow.datatools.s3 import MetaflowS3NotFound

from industrial_taxonomy.getters.glass_house import FullMatchResult

try:  # Hack for type-hints on attributes
    from pandas import DataFrame
except ImportError:
    pass


CHUNKSIZE = 100_000  # Chunksize of matches output by Jacchammer and persisted to S3


@conda_base(python="3.9")  # Need >=3.8 on Batch for TypedDict of FullMatchResult
@project(name="industrial_taxonomy")
class JacchammerFlow(FlowSpec):
    """Fuzzy match Companies House and Glass AI names using Jacchammer.

    Method overview:
    - Use two methods to generate pairs of names likely to be similar without
      realising all pairwise combinations which is prohibitively expensive.
      1. MinHash + Locality Sensitive Hashing
      2. Top-k Cosine similarity of TF-IDF
    - For the pairs in the previous step calculate the exactly similarity via.
      multiple methods.
      1. Jaccard Similarity of k-shingles
      2. Levenshtein score
      3. Cosine similarity of TF-IDF
    - Return Companies House name with highest mean match score for each Glass name

    Attributes:
        run_id: The origin run ID of the flow
        ch_names: Companies House company names
        glass_names: Glass AI organisation names
        clean_ch_names: Pre-processed Companies House company names
        clean_glass_names: Pre-processed Glass AI organisation names
        full_top_matches: Full match results
    """

    drop_matches_below = Parameter(
        "drop-matches-below",
        help=(
            "Internal optimisation to reduce number of matches persisted."
            "Should still be low enough to make a domain decision about what "
            "score constitutes a good match."
        ),
        type=int,
        default=50,
    )

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )
    should_clean_names = Parameter(
        "clean-names",
        help="Whether to clean names using Jacchammer default `preproc_names`",
        type=bool,
        default=False,
    )

    ch_names: "DataFrame"
    glass_names: "DataFrame"
    clean_ch_names: "DataFrame"
    clean_glass_names: "DataFrame"
    run_id: int
    full_top_matches: List[FullMatchResult]  # TODO: py37 error

    @step
    def start(self):
        """Load raw data."""
        from industrial_taxonomy.getters import glass, companies_house

        self.run_id = current.origin_run_id or current.run_id

        nrows = 10_000 if (self.test_mode and not current.is_production) else None

        self.ch_names = companies_house.get_name().name.head(nrows)
        self.glass_names = glass.get_organisation().name.head(nrows)

        self.next(self.clean_names)

    @pip(path="requirements.txt")
    @step
    def clean_names(self):
        """Pre-process names if `self.should_clean_names`."""
        from jacc_hammer.name_clean import preproc_names

        self.clean_ch_names = (
            preproc_names(self.ch_names) if self.should_clean_names else self.ch_names
        )
        self.clean_glass_names = (
            preproc_names(self.glass_names)
            if self.should_clean_names
            else self.glass_names
        )

        self.next(self.match)

    @resources(memory=64_000)
    @pip(path="requirements.txt")
    @step
    def match(self):
        """The core fuzzy matching algorithm."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from jacc_hammer.fuzzy_hash import match_names_stream

        from utils import optimise_chunk

        tmp_dir = Path(TemporaryDirectory(dir=".").name)
        tmp_dir.mkdir()

        # When Jacchammer generates candidate pairs likely to be similar,
        # it vastly decreases the number of pairwise comparisons necessary such
        # that the matching task becomes computationally feasible; however the
        # number of candidate pairs is still very large!
        # Persisting the full list of matches as a regular Metaflow artifact
        # is not feasible therefore we persist smaller chunks of matches to S3
        # (Jacchammer can output chunks of matches to slightly ease memory usage.)
        chunks = match_names_stream(
            [self.clean_ch_names.tolist(), self.clean_glass_names.tolist()],
            threshold=self.drop_matches_below,
            chunksize=CHUNKSIZE,
            tmp_dir=tmp_dir,
        )
        self._chunk_ids = [
            self._persist_chunk(id, optimise_chunk(chunk))
            for id, chunk in enumerate(chunks)
        ]

        self.next(self.extract_top_matches)

    @resources(memory=64_000)
    @step
    def extract_top_matches(self):
        """Extract top matches."""
        from pandas import concat
        from utils import get_top_matches

        self.top_matches = get_top_matches(
            concat(map(self._get_chunk, self._chunk_ids)).reset_index(drop=True)
        )

        print("Number of matches: ", self.top_matches.shape[0])
        self.next(self.end)

    @step
    def end(self):
        """Merge names and id's back in and convert to dict."""

        self.full_top_matches = (
            self.top_matches.merge(
                self.glass_names.reset_index().rename(columns={"name": "org_name"}),
                left_on="y",
                right_index=True,
                validate="1:1",
            )
            .merge(
                self.ch_names.reset_index().rename(columns={"name": "company_name"}),
                left_on="x",
                right_index=True,
                validate="m:1",
            )
            .drop(["x", "y"], axis=1)
            .to_dict("records")
        )
        del self.top_matches

    def _persist_chunk(self, chunk_id: int, chunk: "DataFrame") -> int:
        key = f"match-chunks-{chunk_id}"
        with S3(run=self) as s3:
            s3.put(key, chunk.to_csv())
        print(f"Persisted chunk {chunk_id} of shape {chunk.shape}")
        return chunk_id

    def _get_chunk(self, chunk_id: int) -> "DataFrame":
        from io import StringIO
        from pandas import read_csv

        key = f"match-chunks-{chunk_id}"
        try:
            with S3(run=self) as s3:
                return read_csv(StringIO(s3.get(key).text))
        except MetaflowS3NotFound:  # If flow resumed, could be under `origin_run_id`
            with S3(run=Run(f"{current.flow_name}/{current.origin_run_id}")) as s3:
                return read_csv(StringIO(s3.get(key).text))


if __name__ == "__main__":
    JacchammerFlow()
