"""Tokenises and ngrams documents.

- Run documents through a spacy pipeline
  (large english model with entity merging)
- Convert to bag of words
  (using either lemmatisation or by remapping certain entities to a common token
   - configured by `entity_mappings` param)
- Filter low frequency words
  (note: purely done for computational efficiency to avoid explosion of n-grams)
- Generate n-grams using statistical co-occurrence (gensim)
- Filter low and high frequency n-grams
- Filter very short n-grams or bi-grams that are purely stop words
"""
import json
from typing import Generator

import toolz.curried as t
from metaflow import FlowSpec, step, Parameter, IncludeFile, JSONType, conda_base

from nlp_utils import spacy_pipeline, ngram_pipeline, spacy_to_tokens


@conda_base(
    libraries={
        "spacy": ">=3.0",
        "toolz": ">0.11",
        "gensim": ">=4.0",
        "spacy-model-en_core_web_lg": ">=3.0",
    },
    python="3.8",
)
class EscoeNlpFlow(FlowSpec):
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
    input_file = IncludeFile(
        "input-file",
        help="JSON file, mapping document id to document text."
        " Structure: Dict[str: str]",
    )

    def pop_documents(self) -> Generator[str, None, None]:
        """Destructively yield from `self.documents`."""
        self.documents.reverse()
        n = len(self.documents)
        i = 0
        while i < n:
            yield self.documents.pop()
            i += 1

    @step
    def start(self):
        """Load data and run the NLP pipeline, returning tokenised documents."""

        data = json.loads(self.input_file)
        self.documents = list(data.values())
        self.keys = list(data.keys())
        print(f"Received {len(data)} documents")

        spacyify = t.curry(spacy_pipeline().pipe, n_process=self.n_process)

        tokens = t.pipe(
            self.pop_documents(),
            # TODO Step: Remove HTML ?
            # Step: Spacy
            spacyify,
            # Step: Spacy -> (ordered) Bag of words
            t.map(spacy_to_tokens(entity_mappings=None)),
            # Step: construct n-grams
            list,
            ngram_pipeline(n=self.n_gram),
        )

        self.documents = dict(zip(self.keys, tokens))
        print(f"Processed {len(self.documents)} documents")
        print(len(self.documents), len(self.keys))
        assert len(self.documents) == len(self.keys), (
            "Number of document ID's and processed documents does not match... "
            f"{len(self.keys)} != {len(self.documents)}"
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    EscoeNlpFlow()
