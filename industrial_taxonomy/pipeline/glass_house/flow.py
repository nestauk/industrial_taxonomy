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


@conda_base(python="3.9")  # Need >=3.8 on Batch for TypedDict of FullMatchResult
@project(name="industrial_taxonomy")
class JacchammerFlow(FlowSpec):
    """Fuzzy match Companies House and Glass AI names using Jacchammer.

    Attributes:
        run_id: The origin run ID of the flow
        names_x: Companies House company names
        names_y: Glass AI organisation names
        clean_names_x: Pre-processed Companies House company names
        clean_names_y: Pre-processed Glass AI organisation names
        full_top_matches: Full match results
    """

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

    names_x: "DataFrame"
    names_y: "DataFrame"
    clean_names_x: "DataFrame"
    clean_names_y: "DataFrame"
    run_id: int
    full_top_matches: List[FullMatchResult]  # TODO: py37 error

    @step
    def start(self):
        """Load raw data"""
        from industrial_taxonomy.getters.companies_house import (
            get_name as get_companies_house_names,
        )
        from industrial_taxonomy.getters.glass import (
            get_organisation as get_glass_organisation,
        )

        self.run_id = current.origin_run_id or current.run_id

        nrows = 10_000 if (self.test_mode and not current.is_production) else None

        self.names_x = get_companies_house_names().name.head(nrows)
        self.names_y = get_glass_organisation().name.head(nrows)

        self.next(self.clean_names)

    @pip(path="requirements.txt")
    @step
    def clean_names(self):
        """Pre-process names if `self.should_clean_names`."""
        from jacc_hammer.name_clean import preproc_names

        self.clean_names_x = (
            preproc_names(self.names_x) if self.should_clean_names else self.names_x
        )
        self.clean_names_y = (
            preproc_names(self.names_y) if self.should_clean_names else self.names_y
        )

        self.next(self.match)

    @resources(memory=64_000)
    @pip(path="requirements.txt")
    @step
    def match(self):
        """The core fuzzy matching algorithm"""
        from itertools import starmap as starmap_
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from jacc_hammer.fuzzy_hash import (
            Cos_config,
            Fuzzy_config,
            match_names_stream,
        )
        from pandas import DataFrame
        from toolz.curried import curry, pipe
        from toolz.curried import map as tmap

        from utils import optimise_chunk

        starmap = curry(starmap_)

        def persist_chunk(chunk_id: int, chunk: DataFrame) -> int:
            key = f"match-chunks-{chunk_id}"
            with S3(run=self) as s3:
                s3.put(key, chunk.to_csv())
            print(f"Persisted chunk {chunk_id} of shape {chunk.shape}")
            return chunk_id

        tmp_dir = Path(TemporaryDirectory(dir=".").name)
        tmp_dir.mkdir()

        cos_config = Cos_config()
        fuzzy_config = Fuzzy_config(num_perm=128)
        match_config = dict(
            threshold=50,
            chunksize=100_000,
            cos_config=cos_config,
            fuzzy_config=fuzzy_config,
            tmp_dir=tmp_dir,
        )

        self._chunk_ids: List[int] = pipe(
            match_names_stream(
                [self.clean_names_x.tolist(), self.clean_names_y.tolist()],
                **match_config,
            ),
            tmap(optimise_chunk),
            enumerate,  # Give chunks ID's
            starmap(persist_chunk),
            list,  # Realise computation
        )
        self.next(self.extract_top_matches)

    @resources(memory=64_000)
    @step
    def extract_top_matches(self):
        """Extract top matches."""
        from io import StringIO
        from pandas import concat, DataFrame, read_csv
        from utils import get_top_matches

        def get_chunk(chunk_id: int) -> DataFrame:
            key = f"match-chunks-{chunk_id}"
            try:
                with S3(run=self) as s3:
                    return read_csv(StringIO(s3.get(key).text))
            except MetaflowS3NotFound:  # In case of Flow resumption
                with S3(run=Run(f"{current.flow_name}/{current.origin_run_id}")) as s3:
                    return read_csv(StringIO(s3.get(key).text))

        self.top_matches = get_top_matches(
            concat(map(get_chunk, self._chunk_ids)).reset_index(drop=True)
        )

        print("Number of matches: ", self.top_matches.shape[0])
        self.next(self.end)

    @step
    def end(self):
        """Merge names and id's back in and convert to dict."""

        self.full_top_matches = (
            self.top_matches.merge(
                self.names_y.reset_index().rename(columns={"index": "index_y"}),
                left_on="y",
                right_index=True,
                validate="1:1",
            )
            .merge(
                self.names_x.reset_index().rename(columns={"index": "index_x"}),
                left_on="x",
                right_index=True,
                validate="m:1",
            )
            .rename(columns={"name_x": "company_name", "name_y": "org_name"})
            .drop(["x", "y"], axis=1)
            .to_dict("records")
        )
        del self.top_matches


if __name__ == "__main__":
    JacchammerFlow()
