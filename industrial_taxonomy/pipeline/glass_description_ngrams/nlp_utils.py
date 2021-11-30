"""Core NLP pipeline functionality for `nlp_flow.EscoeNlpFlow`."""
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Set

import spacy
import toolz.curried as t
from gensim import models
from gensim.corpora import Dictionary
from spacy.lang.en import STOP_WORDS
from spacy.tokens import Doc, Token
from spacy.language import Language

SECOND_ORDER_STOP_WORDS: Set[str] = t.pipe(
    STOP_WORDS,
    lambda stops: product(stops, stops),
    t.map(lambda x: f"{x[0]}_{x[1]}"),
    set,
    lambda stops: stops | STOP_WORDS,
)


def build_ngrams(
    documents: List[List[str]], n: int = 2, phrase_kws: Optional[Dict[str, Any]] = None
) -> List[List[str]]:
    """Create ngrams using Gensim's phrases.

    Args:
        documents: List of tokenised documents.
        n: The `n` in n-gram.
        phrase_kws: Passed to `gensim.models.Phrases`.

    Returns:
        List of n-grammed documents.
    """
    if n < 2:
        return documents

    def_phrase_kws = {
        "scoring": "npmi",
        "threshold": 0.25,
        "min_count": 2,
        "delimiter": "_",
    }
    phrase_kws = t.merge(def_phrase_kws, phrase_kws or {})

    step = 1
    while step < n:
        phrases = models.Phrases(documents, **phrase_kws)
        bigram = models.phrases.Phraser(phrases)
        del phrases
        tokenised = bigram[documents]
        step += 1

    return list(tokenised)


def _token_filter(tokens: Iterable[str]) -> Iterable[str]:
    """Filter short n-grams and bi-grams that are just stopwords."""
    return filter(
        lambda token: (
            # No short words
            (not len(token) <= 2)
            # No stopwords
            and (token not in SECOND_ORDER_STOP_WORDS)
        ),
        tokens,
    )


@t.curry
def _filter_frequency(
    documents: List[str], kwargs: Optional[Dict[str, Any]] = None
) -> Iterable[str]:
    """Filter `documents` based on token frequency corpus."""
    dct = Dictionary(documents)

    default_kwargs = dict(no_below=10, no_above=0.9, keep_n=1_000_000)
    if kwargs is None:
        kwargs = default_kwargs
    else:
        kwargs = t.merge(default_kwargs, kwargs)

    dct.filter_extremes(**kwargs)
    return t.pipe(
        documents,
        t.map(lambda document: [token for token in document if token in dct.token2id]),
    )


def spacy_pipeline() -> Language:
    """Spacy NLP pipeline - large english model with entity merging.

    Returns:
        A spacy language model.
    """
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("merge_entities")
    return nlp


@t.curry
def spacy_to_tokens(
    doc: Doc, entity_mappings: Optional[Dict[str, str]] = None
) -> List[str]:
    """Convert spacy document to list of tokens.

    Args:
        doc: Spacy document.
        entity_mappings: Mapping between spacy entity types and the
            phrase to use in place of the lemmatised token.

    Returns:
        List of tokens based on the spacy Doc.

        Filters tokens that are stopwords, punctuation, and whitespace.

        If a detected entity of a type in `entity_mappings`,
        replace token with the value from that mapping;
        otherwise extracts the lemmatisation of a token.
    """

    entity_mappings_: Dict[str, str] = entity_mappings or {
        "CARDINAL": "CARDINAL",
        "DATE": "DATE",
        "GPE": "GPE",
        "LOC": "LOC",
        "MONEY": "MONEY",
        "NORP": "NORP",  # ?
        "ORDINAL": "ORDINAL",
        "ORG": "ORG",
        "PERCENT": "PERCENT",
        "PERSON": "PERSON",
        "QUANTITY": "QUANTITY",
        "TIME": "TIME",
        # Undesirable / empricially too inaccurate:
        #   "EVENT",
        #   "FAC",
        #   "LANGUAGE",
        #   "LAW",
        #   "PRODUCT",  # Loses too much info, we're interested in industry
        #   "WORK_OF_ART",
    }

    def extract_text(token: Token) -> str:
        """Extract text from a spacy token."""
        if token.ent_type_ in entity_mappings_:
            return entity_mappings_[token.ent_type_]
        else:
            return token.lemma_

    return t.pipe(
        doc,
        t.filter(lambda x: not (x.is_stop or x.is_punct or x.is_space)),
        t.map(extract_text),
        list,
    )


@t.curry
def ngram_pipeline(docs: List[List[str]], n: int) -> List[List[str]]:
    """Pipeline to turn tokens into a list of n-grams.

    Args:
        docs: List of document tokens.
        n: the n in n-gram.

    Returns:
        List of n-grammed documents.
    """
    return t.pipe(
        docs,
        # Filter low frequency terms (want to keep high frequency terms)
        _filter_frequency(kwargs={"no_above": 1}),
        list,
        # N-gram based on statistical co-occurrence
        t.curry(build_ngrams, n=n),
        # Filter ngrams: combination of stopwords, e.g. `of_the`
        t.map(t.compose(list, _token_filter)),
        list,
        # Filter ngrams:  low (and very high) frequency terms
        _filter_frequency,
        list,
    )
