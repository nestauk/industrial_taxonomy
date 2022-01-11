# Utilities to fit topics models and

from typing import List, Tuple, Dict, Union
from industrial_taxonomy.pipeline.glass_clusters.hSBM_Topicmodel.sbmtm import sbmtm


def fit_model(corpus: List[List[str]], doc_ids: List) -> sbmtm:
    """Trains top sbm model on tokenised corpus

    Args:
        corpus: list of tokenised documents
        doc_ids: their ids

    Returns:
        A topic model fit on the corpus
    """

    model = sbmtm()
    model.make_graph(corpus, documents=doc_ids)
    model.fit()
    return model


def fit_model_sector(sector_corpus: Dict[str, Dict[int, List[str]]]) -> sbmtm:
    """Fits the model taking the sector corpus data structure as an input

    Args:
        sector_corpus: a dict where keys are doc ids and
            values are the tokenised descriptions

    Returns:
        The model
    """

    return fit_model(
        corpus=list(sector_corpus.values()), doc_ids=list(sector_corpus.keys())
    )


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

    if type(docs_to_assign) is int:
        extract_n = docs_to_assign
    elif type(docs_to_assign) is float:
        extract_n = int(len(model.documents) * docs_to_assign)
    elif type(docs_to_assign) is str:
        extract_n = len(model.documents)

    cluster_assignment = model.clusters(l=cl_level, n=extract_n)

    return [(f"{sector}_{k}", el[0]) for k, v in cluster_assignment.items() for el in v]
