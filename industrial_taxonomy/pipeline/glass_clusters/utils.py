"""Utils to create a sector corpus
and implement diagnostics of clustering homogeneity"""
from toolz import pipe
from functools import partial
from typing import Dict, List, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from industrial_taxonomy.pipeline.glass_clusters.sbmtm import sbmtm
from industrial_taxonomy.pipeline.glass_clusters.topic_utils import extract_topic_mix

NE_CODES = {
    "CARDINAL",
    "DATE",
    "GPE",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERSON",
    "QUANTITY",
    "TIME",
}

token_descr = List[str]
org_id = str
sic_4 = str


def strip_nes(tokenised_dict: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Removes named entities from company tokenised descriptions"""

    return {
        k: [t for t in v if all(ne not in t for ne in NE_CODES)]
        for k, v in tokenised_dict.items()
    }


def filter_non_matched_comps(
    tokenised: Dict[org_id, token_descr], matched_ids: set
) -> Dict[org_id, token_descr]:
    """Removes tokenised descriptions of glass companies that were not matched with CH

    Args:
        tokenised: lookup between company ids and tokenised descriptions
        matched_ids: ids from glass companies matched with CH

    Returns:
        filtered tokenised descriptions dict
    """

    return {id_: tok for id_, tok in tokenised.items() if id_ in matched_ids}


def big_sector_tokens_lookup(
    tokenised: Dict[org_id, token_descr], gl_sic4: Dict[org_id, sic_4], big_sectors: set
) -> Dict[sic_4, Dict[org_id, token_descr]]:
    """Creates a dict where keys are (big) sectors and
    values the tokenised descriptions of their companies.

    Args:
        tokenised: lookup between company ids and tokenised descriptions
        gl_sic4: lookup between company ids and SIC4s
        big_sectors: sectors above a certain size threshold

    Returns:
        dict with sectors and tokenised descriptions

    """

    return {
        sector: {id_: tok for id_, tok in tokenised.items() if gl_sic4[id_] == sector}
        for sector in big_sectors
    }


def make_sector_corpora(
    glass_sic4: Dict[int, str],
    token_descr: Dict[org_id, token_descr],
    min_sector_size: int = 1000,
) -> Dict[str, Dict[int, List[str]]]:
    """Creates a dict of sectors and the tokenised descriptions of their companies

    Args:
        glass_sic4: lookup between glass ids and sic4s
        token_descr: lookup between glass ids and tokenised descriptions
        min_sector_size: minimum sector size

    Returns:
        dict with sectors and tokenised descriptions for their companies
    """
    selected_sectors = set(
        sector
        for sector, sector_n in pd.Series(glass_sic4).value_counts().items()
        if sector_n > min_sector_size
    )

    return pipe(
        token_descr,
        strip_nes,
        partial(filter_non_matched_comps, matched_ids=set(glass_sic4.keys())),
        partial(
            big_sector_tokens_lookup, gl_sic4=glass_sic4, big_sectors=selected_sectors
        ),
    )


def get_sector_companies(
    text_sectors: List[Tuple[str, str]], sic4: str
) -> Dict[str, str]:
    """Creates lookup between company ids and text sectors inside a sic4

    Args:
        text_sectors: company ids and their text sectors
        sic4: sic code

    Returns:
        Lookup between company id and text sectors within a sic4
    """

    return {
        assign[1]: assign[0] for assign in filter(lambda x: sic4 in x[0], text_sectors)
    }


def get_topics_labels(
    sector_companies: Dict[str, str], topic_mix: pd.DataFrame
) -> List[np.array]:
    """Get topic mix and sector labels for the silhouette analysis"""

    tm_filt = topic_mix.assign(
        sector=lambda df: df.index.map(sector_companies)
    ).dropna()

    tm_filt_x = tm_filt[topic_mix.columns].to_numpy()
    tm_filt_labs = tm_filt["sector"].to_numpy()

    return [tm_filt_x, tm_filt_labs]


def get_silouhette_scores(
    x_labs: List[np.array], dist_metric: str = "cosine"
) -> pd.DataFrame:
    """Calculate aggregate and cluster-level silouhette scores

    Args:
        x_labs: sector label and topic mix for each company
        dist_metric: distance measure to calculate the silhouette score

    Returns:
        sic level and text sector level silhouette scores

    """

    X, labs = x_labs

    s_score = silhouette_score(X, labs, metric=dist_metric)

    s_cluster = (
        pd.DataFrame(
            {"score": silhouette_samples(X, labs, metric=dist_metric), "cluster": labs}
        )
        .groupby("cluster")
        .mean()
    )

    return {"local_score": s_score, "cluster_scores": s_cluster}


def text_sector_silhouette(
    companies_sector_all: List[Tuple[str, str]],
    sector: str,
    models: Dict[str, sbmtm],
) -> Dict[str, Union[float, pd.DataFrame]]:
    """Return silouhette scores for companies in a sector

    Args:
        companies_sector_all: companies with text sector labels
        models: tobsbm models
        sector: focal sector

    Returns:
        Sector (SIC) and silhouette scores
    """

    return pipe(
        companies_sector_all,
        partial(get_sector_companies, sic4=sector),
        partial(get_topics_labels, topic_mix=extract_topic_mix(models[sector])),
        get_silouhette_scores,
    )


def filter_clusters(clusters_silh_scores: pd.DataFrame, q: float) -> dict:
    """Filters clusters based on the position of their silhouette score in the
        distribution

    Args:
        clusters_silh_scores: cluster level silhouette scores
        q: silhouette threshold

    Returns:
        Lookup between assignment strategies and filtered sectors
    """

    return {
        assg_share: pipe(
            clusters_silh_scores.query(f"assign_strategy=='{assg_share}'"),
            lambda df: df.loc[df["score"] > df["score"].quantile(q)]["cluster"],
            set,
        )
        for assg_share in clusters_silh_scores["assign_strategy"].unique()
    }
