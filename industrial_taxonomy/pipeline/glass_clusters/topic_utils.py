# Utilities to fit topics models and cluster documents

from typing import List, Tuple, Dict, Union
from industrial_taxonomy.pipeline.glass_clusters.sbmtm import sbmtm

DocId = int
Token = List[str]


def fit_model_sector(sector_corpus: Dict[DocId, Token]) -> sbmtm:
    """Fits the model taking the sector corpus data structure as an input

    Args:
        sector_corpus: a dict where keys are doc ids and
            values are the tokenised descriptions

    Returns:
        The model
    """

    model = sbmtm()
    model.make_graph(list(sector_corpus.values()), documents=list(sector_corpus.keys()))
    model.fit()
    return model


def get_n_docs_to_assign(docs_to_assign: Union[int, float, str], corpus_n: int) -> int:
    """Converts a parameter with the number of docs to extract into an integer

    Args:
        docs_to_assign: value we want to extract
        corpus_n: number of documents that have been topic modelled

    Returns:
        number of documents to put in each cluster

    Raises:
        ValueError: if docs_to_assign is not in a suitable type
    """
    if type(docs_to_assign) is int:
        return docs_to_assign
    elif type(docs_to_assign) is float:
        return int(corpus_n * docs_to_assign)
    elif type(docs_to_assign) is str:
        return corpus_n
    else:
        raise ValueError("The input needs to be integer, float or str")


def extract_clusters(
    model: sbmtm,
    sector: str,
    docs_to_assign: Union[int, float, str] = 10,
    cl_level: int = 0,
) -> List[Tuple[int, str]]:
    """Extracts clusters from the model

    Args:
        model: the topic model
        sector: the sector label
        docs_to_assign: number / share of clustered docs we want to keep.
            If 'all', assign all documents
        cl_level: level of clustering  (0 is doc level)

    Returns:
        A tuple with doc ids and their cluster names (SIC label + cluster id number)
    """
    extract_n = get_n_docs_to_assign(docs_to_assign, len(model.documents))

    cluster_assignment = model.clusters(l=cl_level, n=extract_n)

    return [(f"{sector}_{k}", el[0]) for k, v in cluster_assignment.items() for el in v]
