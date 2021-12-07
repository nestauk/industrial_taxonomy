from typing import Dict, Generator, List

from metaflow import current, FlowSpec, step, Parameter, project, JSONType, conda_base


@conda_base(
    libraries={
        "spacy": ">=3.0",
        "toolz": ">0.11",
        "gensim": ">=4.0",
        "spacy-model-en_core_web_lg": ">=3.0",
        "beautifulsoup4": "4.10.0",
        "lxml": ">=4.6",
    },
    python="3.8",
)
@project(name="industrial_taxonomy")
class GlassNlpFlow(FlowSpec):
    """Process Glass descriptions into n-grammed tokens.

    Steps:
    - Run documents through a spacy pipeline
      (large english model with entity merging)
    - Convert to bag of words
      (using either lemmatisation or by remapping certain entities to a common
      token configured by `entity_mappings` param)
    - Filter low frequency words
      (note: purely done for computational efficiency to avoid explosion of n-grams)
    - Generate n-grams using statistical co-occurrence (gensim)
    - Filter low and high frequency n-grams
    - Filter very short n-grams or bi-grams that are purely stop words

    Attributes:
        keys: Document identifiers
        raw_documents: Raw document descriptions [MUTATES]
        documents: Processed document descriptions
    """

    keys: List[str]
    raw_documents: List[str]
    documents: Dict[str, List[str]]

    n_process = Parameter(
        "n-process",
        help="The number of processes to use with spacy (default: 1)",
        type=int,
        default=1,
    )
    n_gram = Parameter(
        "n-gram",
        help="The `N` in N-gram",
        type=int,
        default=2,
    )
    entity_mappings = Parameter(
        "entity-mappings",
        help="JSON string, mapping spacy entity types to the phrase to substitute"
        "their lemmatisation with",
        type=JSONType,
        default=None,
    )
    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    def pop_documents(self) -> Generator[str, None, None]:
        """Destructively yield from `self.documents`."""
        self.raw_documents.reverse()
        n = len(self.raw_documents)
        i = 0
        while i < n:
            yield self.raw_documents.pop()
            i += 1

    @step
    def start(self):
        """Load data."""
        from industrial_taxonomy.getters.glass import get_organisation_description
        from industrial_taxonomy.getters.glass_house import glass_companies_house_lookup

        matched_glass_org_ids = glass_companies_house_lookup().keys()
        nrows = 1_000 if self.test_mode and not current.is_production else None
        data = (
            get_organisation_description()
            .loc[lambda x: x.index.isin(matched_glass_org_ids), "description"]
            .head(nrows)
            .to_dict()
        )
        self.raw_documents = list(data.values())
        self.keys = list(data.keys())
        print(f"Received {len(data)} documents")

        self.next(self.process)

    @step
    def process(self):
        """Run the NLP pipeline, returning tokenised documents."""
        from bs4 import BeautifulSoup
        from toolz.curried import map as mapc, pipe
        from nlp_utils import spacy_pipeline, ngram_pipeline, spacy_to_tokens

        def _strip_html(document: str) -> str:
            # noqa: DAR
            """Strips lingering HTML tags and parses HTML escape characters.

            Doesn't remove partial tags (which do exist in the Glass data), e.g.
            - 'p> This is a description</p>' => 'p> This is a description'
            - 'a href="http://foo.bar"><img src="foo' => 'a href="http://foo.bar">'

            There is a small change of what could be seen as a false removal, e.g.
            - 'something <word-without-space-before' => something '
            """
            return BeautifulSoup(document, "lxml").text

        tokens = pipe(
            self.pop_documents(),
            mapc(_strip_html),
            # Step: Spacy
            lambda docs: spacy_pipeline().pipe(docs, n_process=self.n_process),
            # Step: Spacy -> (ordered) Bag of words
            mapc(spacy_to_tokens(entity_mappings=None)),  # pylint: disable=E1120
            # Step: construct n-grams
            list,
            ngram_pipeline(n=self.n_gram),  # pylint: disable=E1120
        )

        self.documents = dict(zip(self.keys, tokens))
        print(f"Processed {len(self.documents)} documents")
        assert len(self.documents) == len(self.keys), (
            "Number of document ID's and processed documents does not match... "
            f"{len(self.keys)} != {len(self.documents)}"
        )
        self.next(self.end)

    @step
    def end(self):
        """No-op."""
        pass


if __name__ == "__main__":
    GlassNlpFlow()
