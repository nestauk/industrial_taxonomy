from collections import defaultdict
import numpy as np
from operator import itemgetter


def split_dict(d, params):
    """Splits dict into key and value lists to pass through `foreach` step."""
    d = {k: v for k, v in d.items() if k in params}
    return list(d.keys()), list(d.values())


def get_locs(ids, vector_index):
    """Gets the locations of some elements as they appear in an index."""
    id_to_loc = dict(zip(vector_index, range(len(vector_index))))
    locs = np.array(itemgetter(*ids)(id_to_loc))
    return locs


def sic4_lookups(text_sector_embedding_lookup):
    """Generates a lookup between SIC4 codes, text sectors and Glass
    organisation IDs."""
    sic4_embedding_lookup = defaultdict(list)
    sic4_text_sector_lookup = defaultdict(list)
    for label, org_ids in text_sector_embedding_lookup.items():
        sic4_embedding_lookup[label[:4]].extend(org_ids)
        sic4_text_sector_lookup[label[:4]].append(label)
        
    return sic4_embedding_lookup, sic4_text_sector_lookup


def sector_org_ids_lookup(sectors):
    """Creates a lookup between text sector labels and the Glass organisation
    IDs for the companies in that text sector."""
    lookup = defaultdict(list)
    for label, org_id in sectors:
        lookup[label].append(org_id)
    return lookup


def get_clusters_embeddings(
    clusters,
    org_ids,
    embeddings,
):
    """Returns the embeddings for a set of clusters.

    Args:
        clusters: List of cluster ID + org ID pairs.
        org_ids: List of Glass org IDs corresponding to embedding rows.
        embeddings: 2-D array of Glass org embeddings.

    Returns:
        Embeddings for the organisations in clusters, in the order that
            they were passed.
    """
    glass_org_ids_clusters = [c[1] for c in clusters]
    glass_org_id_to_loc = dict(zip(org_ids, range(len(org_ids))))
    embedding_locs = np.array(itemgetter(*glass_org_ids_clusters)(glass_org_id_to_loc))
    return embeddings[embedding_locs], embedding_locs


def tfidf_vectors(texts, vectorizer, min_df=3, ngram_range=(2, 3)):
    """Generates tf-idf vectors from a set of documents with some fixed
    parameters."""
    tfidf_vectorizer = vectorizer(
        token_pattern=r"[A-Za-z]+",
        analyzer="word",
        ngram_range=ngram_range,
        stop_words="english",
        min_df=min_df,
    )
    return tfidf_vectorizer.fit_transform(texts), tfidf_vectorizer


def top_tfidf_terms(tfidf_vectors, tfidf_vectorizer, topn=20):
    """Sums the tf-idf values for each term across documents and sorts them by
    score. Returns the `topn` with the highest tf-idf values."""
    summed_tfidf = tfidf_vectors.sum(axis=0)

    summed_tfidf_list = summed_tfidf.tolist()[0]
    ngram_list = tfidf_vectorizer.get_feature_names()
    ngram_scores = list(zip(summed_tfidf_list, ngram_list))
    sorted_ngram_scores = sorted(
        ngram_scores,
        key=lambda tup: tup[0],
        reverse=True,
    )

    return [ngram[1] for ngram in sorted_ngram_scores[:topn]]


def central_ngrams(terms, encoder, sector_embeddings, dist_func, topn=3):
    """Returns the `topn` terms that have the highest average pairwise cosine
    similarity with the documents they were generated from."""
    ngram_embeddings = encoder.encode(terms)

    mean_sims = dist_func(
        ngram_embeddings,
        sector_embeddings,
        metric="cosine",
    ).mean(axis=1)
    top_sim_locs = np.argsort(mean_sims)[:topn]

    return [terms[i] for i in top_sim_locs]
