import altair as alt
import logging
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from collections import Counter
from toolz import pipe, keymap
from itertools import combinations
from community import community_louvain
from metaflow import Run
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP
from industrial_taxonomy.getters.glass import get_address
from industrial_taxonomy.getters.text_sectors import (
    org_ids_reassigned,
    assigned_text_sector,
    original_text_sector,
    knn_text_sector_agg_sims,
)
from industrial_taxonomy.utils.metaflow import get_run
from industrial_taxonomy.getters.sic import level_lookup
from scipy.stats import iqr


clustering_param = str
sector_id = str
glass_id = str

ASSGN_KEYS = [
    "assigned_500",
    "assigned_100",
    "assigned_0.5",
    "assigned_10",
    "assigned_all",
]


def combine_reassgn_outputs(output_list: list) -> dict:
    """Combines the text assignment outputs for originally
    clustered and unclustered organisations
    """

    return {
        assgn: np.concatenate([output[assgn] for output in output_list])
        for assgn in ASSGN_KEYS
    }


def make_reassgn_lookup(id_lookup: dict, text_assgn_lookup: dict) -> dict:
    """Combines org id and text assignments in a single lookup"""

    return {
        assgn: [
            (text_sector, str(_id))
            for _id, text_sector in zip(id_lookup[assgn], text_assgn_lookup[assgn])
        ]
        for assgn in ASSGN_KEYS
    }


def make_sector_assignments(only_clustered: bool = False):
    """Creates combined org id and text sector dicts (if necessary) and
    combines them into a single output

    Args:
        only_clustered: only considers originally clustered organisations
    """

    if only_clustered is True:
        logging.info("reading data")
        org_ids = org_ids_reassigned()
        final_text_sect = assigned_text_sector()

        logging.info("making id - reassignments")
        return make_reassgn_lookup(org_ids, final_text_sect)

    if only_clustered is False:
        logging.info("reading data")
        org_ids_list = [org_ids_reassigned(clustered=_bool) for _bool in [True, False]]
        final_text_sect_list = [
            assigned_text_sector(clustered=_bool) for _bool in [True, False]
        ]

        logging.info("combining data")
        org_ids_combined, final_text_combined = [
            combine_reassgn_outputs(_list)
            for _list in [org_ids_list, final_text_sect_list]
        ]
        logging.info("making id - reassignments")
        return make_reassgn_lookup(org_ids_combined, final_text_combined)


def text_sectors(
    filtered: bool = True,
    run: Optional[Run] = None,
) -> Dict[clustering_param, List[Tuple[sector_id, glass_id]]]:
    """Gets text sector assignments from the topic modelling

    Args:
        filtered: filters low silhouette score clusters
        run: what ClusterGlass flow run to use.
            Defaults to the latest production run

    Returns:
        A lookup between cluster assignment parameters (how many
        companies do we tell topsbm to assign to each cluster based on
        their representativeness) and the actual cluster assignments
        (a tuple with sector id and company id)
    !FIXUP: This should be in getters
    """
    run = run or get_run("ClusterGlass")

    if filtered is True:
        clusters = run.data.clusters
        filtered_clusters = get_run("FilterClusters").data.text_sectors_filtered

        return {
            assign_share: [
                text_s
                for text_s in sectors
                if text_s[0] in filtered_clusters[assign_share]
            ]
            for assign_share, sectors in clusters.items()
        }

    else:
        return run.data.clusters


def fetch_combine(getter):
    """Gets and combines clustered / non clustered outputs"""

    return combine_reassgn_outputs([getter(clustered=True), getter(clustered=False)])


def reassgn_analysis_df(assn="assigned_10"):
    """Creates a dataframe that we can use for a variety of downstrea
    analyses and validations"""

    org_ids = fetch_combine(org_ids_reassigned)[assn]
    old_sectors = fetch_combine(original_text_sector)[assn]
    new_sectors = fetch_combine(assigned_text_sector)[assn]
    max_sims = [max(sim) for sim in fetch_combine(knn_text_sector_agg_sims)[assn]]
    mean_sims = [np.mean(sim) for sim in fetch_combine(knn_text_sector_agg_sims)[assn]]

    return (
        pd.DataFrame(
            {
                name: data
                for data, name in zip(
                    [org_ids, old_sectors, new_sectors, max_sims, mean_sims],
                    ["org_id", "source_sector", "end_sector", "max_sim", "mean_sim"],
                )
            }
        )
        .assign(sic4_source=lambda df: df["source_sector"].str.split("_").str[0])
        .assign(sic4_target=lambda df: df["end_sector"].str.split("_").str[0])
        .replace({None: np.nan})
    )


def check_companies(reassign_df, descr_lookup, sic4_lookup, n=10):
    """Returns example companies to check outputs"""

    selected = reassign_df.query("max_sim>0.6").sample(n)

    for _id, row in selected.iterrows():

        print(row["org_id"])
        print(row["end_sector"])
        print(row["text_name"])
        try:
            print(sic4_lookup[row["sic4_target"]])
        except:
            print("no SIC for this company")
        print(row["max_sim"])
        print(descr_lookup[str(row["org_id"])])
        print("\n")


def text_sector_names_reassigned(
    run: Optional[Run] = None,
) -> Dict:
    """Gets text sector names."""
    run = run or get_run("TextSectorName")
    return run.data.sector_names


def postcode_la_lookup() -> dict:
    """Return lookup between postcode and local authority
    #FIXUP: This should be a getter
    """

    la_nspl = Run("NsplLookup/3021")

    return la_nspl.data.nspl_data[["pcds", "laua_code"]].assign(
        laua_name=lambda df: df.index.map(la_nspl.data.laua_names.squeeze())
    )


def lad_code_name_lookup() -> dict:
    """Return lookup between postcode and local
    #FIXUP This should be a getter
    """
    la_nspl = Run("NsplLookup/3021")

    return la_nspl.data.laua_names.squeeze().to_dict()


def glass_la() -> pd.DataFrame:
    """dataframe with glass id to glass la
    #FIXUP This should be a getter
    """

    return (
        get_address()
        .reset_index(drop=False)
        .query("rank==1")
        .merge(postcode_la_lookup(), left_on="postcode", right_on="pcds")
    )


def create_lq(
    X: pd.DataFrame, threshold: int = 1, binary: bool = False
) -> pd.DataFrame:
    """Calculate the location quotient.
    Divides the share of activity in a location by the share of activity in
    the UK total.

    Args:
        X: Rows are locations, columns are sectors,
        threshold: Binarisation threshold.
        binary: If True, binarise matrix at `threshold`.
            and values are activity in a given sector at a location.

    Returns:
        pandas.DataFrame
    """

    xm = X.values
    with np.errstate(invalid="ignore"):  # Accounted for divide by zero
        x = pd.DataFrame(
            (xm * xm.sum()) / (xm.sum(1)[:, np.newaxis] * xm.sum(0)),
            index=X.index,
            columns=X.columns,
        ).fillna(0)

    return (x > threshold).astype(float) if binary else x


def low_similarity_comps(reassign_df: pd.DataFrame, sim_quantile: float):
    """Filters companies if their similarity score to their
    closest text sector is too low"""

    return pipe(
        reassign_df.query(f"max_sim<{reassign_df['max_sim'].quantile(sim_quantile)}")[
            "org_id"
        ].astype(str),
        set,
    )


def filter_assignments(assignm: Dict, to_drop: Dict):
    """Filters out selected companies e.g. low similarity companies"""

    return {k: v for k, v in assignm.items() if k not in to_drop}


def geo_distribution(
    sector_assignments: Dict[str, Tuple[str, str]],
    sector_shares: str,
    glass_lads: pd.DataFrame,
    specialisation: bool = True,
    binarise=False,
    drop_companies: set = None,
) -> pd.DataFrame:
    """Produces a geographical distribution of activity by sector

    Args:
        sector_assignments: lookup between org ids and sectors
        sector_shares: assignment strategy to focus on
        glass_lads: lookup between glass id and LAD
        drop_companies: companies to drop e.g. low similarities
        specialisation: if we want LQs

    Returns:
        dataframe with geographical specialisation profile (based on LQd)

    """

    # Select sector assignments eg all
    text_sectors = pipe(
        sector_assignments[sector_shares],
        lambda sector_ass_list: {el[1]: el[0] for el in sector_ass_list},
    )

    if drop_companies is not None:

        text_sectors = filter_assignments(text_sectors, drop_companies)

    return pipe(
        glass_lads.assign(
            text_sector=lambda df: df["org_id"].astype(str).map(text_sectors)
        )
        .dropna(axis=0, subset=["text_sector"])
        .groupby(["text_sector", "laua_code", "laua_name"])
        .size()
        .reset_index(name="count")
        .pivot_table(
            index=["laua_code", "laua_name"], columns="text_sector", values="count"
        )
        .fillna(0),
        lambda df: create_lq(df, binarise) if specialisation is True else df,
    )


def reduce_dim(
    text_geo: pd.DataFrame, n_components_pca: int = 50, n_components_umap: int = 2
) -> pd.DataFrame:
    """Reduce dimensionality of sectoral distribution first via PCA and then via UMAP"""

    pca = PCA(n_components=n_components_pca)

    return pipe(
        text_geo,
        lambda df: pd.DataFrame(pca.fit_transform(text_geo), index=text_geo.index),
        lambda df: pd.DataFrame(
            UMAP(n_components=n_components_umap).fit_transform(df),
            index=df.index,
            columns=["x", "y"],
        ),
    )

    return pd.DataFrame(pca.fit_transform(text_geo), index=text_geo.index)


def build_cluster_graph(
    vectors: pd.DataFrame,
    clustering_algorithms: list,
    n_runs: int = 10,
    sample: int = None,
):
    """Builds a cluster network based on observation co-occurrences
    in a clustering output

    Args:
        vectors: vectors to cluster
        clustering_algorithms: a list where the first element is the clustering
            algorithm and the second element are the parameter names and sets
        n_runs: number of times to run a clustering algorithm
        sample: size of the vector to sample.

    Returns:
        A network where the nodes are observations and their edges number
            of co-occurrences in the clustering

    #FIXUP: remove the nested loops
    """

    clustering_edges = []

    index_to_id_lookup = {n: ind for n, ind in enumerate(vectors.index)}

    logging.info("Running cluster ensemble")
    for cl in clustering_algorithms:

        logging.info(cl[0])

        algo = cl[0]

        parametres = [{cl[1][0]: v} for v in cl[1][1]]

        for par in parametres:

            logging.info(f"running {par}")

            for _ in range(n_runs):

                cl_assignments = algo(**par).fit_predict(vectors)
                index_cluster_pair = {n: c for n, c in enumerate(cl_assignments)}

                indices = range(0, len(cl_assignments))

                pairs = combinations(indices, 2)

                for p in pairs:

                    if index_cluster_pair[p[0]] == index_cluster_pair[p[1]]:
                        clustering_edges.append(frozenset(p))

    edges_weighted = Counter(clustering_edges)

    logging.info("Building cluster graph")

    edge_list = [(list(fs)[0], list(fs)[1]) for fs in edges_weighted.keys()]
    cluster_graph = nx.Graph()
    cluster_graph.add_edges_from(edge_list)

    for ed in cluster_graph.edges():

        cluster_graph[ed[0]][ed[1]]["weight"] = edges_weighted[
            frozenset([ed[0], ed[1]])
        ]

    return cluster_graph, index_to_id_lookup


def extract_communities(
    cluster_graph: nx.Graph, resolution: float, index_lookup: dict
) -> list:
    """Extracts community from the cluster graph and names them

    Args:
        cluster_graph: network object
        resolution: resolution for community detection
        index_lookup: lookup between integer indices and project ids

    Returns:
        a lookup between communities and the projects that belong to them
    """
    logging.info("Extracting communities")
    comms = community_louvain.best_partition(cluster_graph, resolution=resolution)

    comm_assignments = {
        comm: [index_lookup[k] for k, v in comms.items() if v == comm]
        for comm in set(comms.values())
    }

    return {
        lad: cluster
        for cluster, lad_list in comm_assignments.items()
        for lad in lad_list
    }


def umap_plot(umap_df: pd.DataFrame):
    """Plots a umap projection of LAD data"""

    no_labels = alt.Axis(labels=False, ticks=False)

    return (
        alt.Chart(umap_df)
        .mark_point(filled=True)
        .encode(
            x=alt.X(
                "x", scale=alt.Scale(zero=False), axis=no_labels, title="Dimension 1"
            ),
            y=alt.Y(
                "y", scale=alt.Scale(zero=False), axis=no_labels, title="Dimension 2"
            ),
            color=alt.Color(
                "la_cluster:N", scale=alt.Scale(scheme="tableau20"), title="LA cluster"
            ),
            tooltip=["la_name", "la_cluster"],
        )
    )


def lad_nuts1_lookup(year: int = 2019) -> dict:
    """Read a lookup between local authorities and NUTS"""

    if year == 2019:
        lu_df = pd.read_csv(
            "https://opendata.arcgis.com/datasets/3ba3daf9278f47daba0f561889c3521a_0.csv"
        )
        return lu_df.set_index("LAD19CD")["RGN19NM"].to_dict()
    else:
        lu_df = pd.read_csv(
            "https://opendata.arcgis.com/datasets/054349b09c094df2a97f8ddbd169c7a7_0.csv"
        )
        return lu_df.set_index("LAD20CD")["RGN20NM"].to_dict()


def assign_lad_to_nuts1(code: str, lad_nuts_lookup: dict):
    """Assigns lad to NUTS including scotland, wales and NI"""

    return (
        np.nan
        if (code[0] == "E") & (code not in lad_nuts_lookup.keys())
        else lad_nuts_lookup[code]
        if code[0] == "E"
        else "Scotland"
        if code[0] == "S"
        else "Wales"
        if code[0] == "W"
        else "Northern Ireland"
    )


def add_names_clusters(
    df: pd.DataFrame, cluster_lu: dict, name_lu: dict
) -> pd.DataFrame:

    return df.assign(la_name=df["la_code"].map(name_lu)).assign(
        sector_cluster=df["la_code"].map(cluster_lu)
    )


def process_benchmark_data(
    local_benchmark_data: pd.DataFrame,
    name_lu: dict,
    last_year: bool = True,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Split the levelling up data and the Nesta innovation data"""

    ons, beis = [
        (local_benchmark_data.query(f"source=='{source}'").reset_index(drop=True))
        for source in ["ons_levelling_up", "nesta_beis"]
    ]

    beis = beis.query("indicator != 'house_price_normalised'")

    if last_year is True:

        beis = (
            beis.groupby(["indicator"])
            .apply(lambda df: (df.query(f"period=={df['period'].max()}")))
            .reset_index(drop=True)
        )

    return ons, beis


def second_bench_plot(second_df: pd.DataFrame) -> alt.Chart:
    """Benchmark clusters against secondary data"""

    median_scores = (
        second_df.groupby(["indicator", "sector_cluster"])["zscore"]
        .median()
        .reset_index(drop=False)
    )

    indicator_var = (
        second_df.groupby("indicator")["zscore"]
        .apply(lambda x: iqr(x))
        .sort_values(ascending=False)
        .index.tolist()
    )

    return (
        alt.Chart(median_scores)
        .mark_point(filled=True, size=80, stroke="black", strokeWidth=0.5)
        .encode(
            y=alt.Y("indicator", sort=indicator_var, title=None),
            x="zscore",
            tooltip=["sector_cluster"],
            color=alt.Color("sector_cluster:N", scale=alt.Scale(scheme="tableau20")),
        )
        .configure_axis(labelFontSize=12, titleFontSize=14, labelLimit=400)
        .configure_legend(labelFontSize=12, titleFontSize=14)
        .properties(width=300, height=600)
    )


def lad_by_clusters(lad_cluster_lookup: dict, lad_name_lookup: dict) -> dict:

    return {
        k: ", ".join(
            [lad_name_lookup[n] for n, v in lad_cluster_lookup.items() if n if v == k]
        )
        for k in set(lad_cluster_lookup.values())
    }


def section_code_lookup() -> Dict[str, str]:
    """Returns lookup from 2-digit SIC code to SIC section letter."""

    def _dictrange(key_range, value) -> dict:
        return {i: value for i in key_range}

    return keymap(
        lambda i: str(i).zfill(2),
        {
            **_dictrange([1, 2, 3], "A"),
            **_dictrange(range(5, 10), "B"),
            **_dictrange(range(10, 34), "C"),
            35: "D",
            **_dictrange(range(36, 40), "E"),
            **_dictrange(range(41, 44), "F"),
            **_dictrange(range(45, 48), "G"),
            **_dictrange(range(49, 54), "H"),
            **_dictrange([55, 56], "I"),
            **_dictrange(range(58, 64), "J"),
            **_dictrange([64, 65, 66], "K"),
            68: "L",
            **_dictrange(range(69, 76), "M"),
            **_dictrange(range(77, 83), "N"),
            84: "O",
            85: "P",
            **_dictrange([86, 87, 88], "Q"),
            **_dictrange(range(90, 94), "R"),
            **_dictrange([94, 95, 96], "S"),
            **_dictrange([97, 98], "T"),
            99: "U",
        },
    )


def sic4_division_lu():
    """Returns a lookup between sic4s and SIC 4 codes and names"""

    sic4_codes = set(get_run("Sic2007Structure").data.class_lookup.keys())

    section_to_division = section_code_lookup()
    division_to_name = level_lookup(1)

    return {
        sic4: ": ".join(
            [
                section_to_division[sic4[:2]],
                division_to_name[section_to_division[sic4[:2]]],
            ]
        )
        for sic4 in sic4_codes
    }


def make_sic4_specialisation(glass_la, sic4_lookup, specialisation: bool = True):
    """Calculates a local authority SIC4 specialisation based on SIC codes"""

    return pipe(
        (
            glass_la.assign(sic4=lambda df: df["org_id"].map(sic4_lookup))
            .groupby(["laua_code", "sic4"])
            .size()
            .unstack()
            .fillna(0)
        ),
        lambda df: create_lq(df) if specialisation is True else df,
    )


def make_secondary_silhouette(secondary, clusters):
    """Calculates the silouhette score for secondary data
    based on clusters
    """

    second_long = secondary.pivot_table(
        index="la_code", columns="indicator", values="zscore"
    ).dropna(axis=0)

    second_w_clusters = second_long.assign(
        cluster=lambda df: df.index.map(clusters)
    ).dropna(axis=0)

    return silhouette_score(
        second_w_clusters[second_long.columns], second_w_clusters["cluster"]
    )


def extract_clusters(
    assign: pd.DataFrame, pca: int, comm_resolution: float, clustering_options: dict
):
    """Function to extract cluster lookups and positions"""
    geo_dim = reduce_dim(assign, n_components_pca=pca)
    clustering, indices = build_cluster_graph(geo_dim, clustering_options)
    lad_cluster_lookup = extract_communities(clustering, comm_resolution, indices)

    umap_df = (
        geo_dim.assign(la_cluster=lambda df: df.index.map(lad_cluster_lookup))
        .assign(la_name=lambda df: df.index.map(lad_code_name_lookup()))
        .reset_index(drop=False)
    )

    return umap_df, lad_cluster_lookup


def silhouette_pipeline(
    assign: pd.DataFrame,
    pca: int,
    comm_resolution: float,
    secondary: list,
    clustering_options,
):
    """Pipeline that calculates silouhette scores for clusters across different
    datasets and parameter values
    """

    umap_df, cluster_lookup = extract_clusters(
        assign, pca, comm_resolution, clustering_options
    )

    # Calculate silhouette scores
    sil_scores = []

    for sec in secondary:
        sil = make_secondary_silhouette(sec, cluster_lookup)
        sil_scores.append(sil)

    return {
        "pca": pca,
        "comm_resolution": comm_resolution,
        "num_clusters": len(set(cluster_lookup.values())),
        "sil_ons": sil_scores[0],
        "sil_beis": sil_scores[1],
    }
