import altair as alt
import numpy as np
import pandas as pd
import geopandas as gp
import json


def process_corr(long_df, x, y, value):
    """Processes a correlation df to make it easier to visualise"""

    return (
        long_df.copy()
        .assign(
            value=lambda df: [
                r[value] if r[x] != r[y] else np.nan for _id, r in df.iterrows()
            ]
        )
        .assign(
            text_value=lambda df: [
                str(r[value])[:4] if not np.isnan(r[value]) else ""
                for _id, r in df.iterrows()
            ]
        )
    )


def plot_corr_mat(
    long_corr: pd.DataFrame, x: str, y: str, value: str = "value"
) -> alt.Chart:
    """Plots a correlation matrix"""

    proc_corr = process_corr(long_corr, x, y, value)

    base = alt.Chart(proc_corr).encode(x=f"{x}:N", y=f"{y}:N")

    heat = base.mark_rect().encode(
        color=alt.Color(
            f"{value}:Q", sort="descending", scale=alt.Scale(scheme="redblue")
        )
    )

    val = base.mark_text().encode(
        text=alt.Text("text_value:N"),
        color=alt.condition(
            (alt.datum.value > 0.6) | (alt.datum.value < 0.2),
            alt.value("white"),
            alt.value("black"),
        ),
    )

    return heat + val


def compare_predictions(result_df: pd.DataFrame, bench_var: str, perf_var: str):
    """Compares predictive performance for different assignment strategies"""

    return (
        alt.Chart(result_df)
        .mark_point(filled=True)
        .encode(
            y=alt.Y(
                "indicator",
                sort=alt.EncodingSortField(perf_var, "mean", order="ascending"),
            ),
            x="r2_test",
            color=f"{bench_var}:N",
        )
    )


def plot_model_outputs(
    gs_outputs: pd.DataFrame, source: str, strat_var: str, bench_var: str
) -> alt.Chart:
    """Plots model outputs"""
    return (
        alt.Chart(gs_outputs.query(f"name=='{source}'"))
        .mark_circle()
        .encode(
            y=alt.Y(
                "indicator",
                title=None,
                sort=alt.EncodingSortField(bench_var, op="mean", order="descending"),
            ),
            x=alt.X(bench_var, scale=alt.Scale(type="symlog")),
            shape="name",
            color=strat_var,
        )
        .configure_axis(labelLimit=300)
    )


def boxplot_performance(
    gs_outputs: pd.DataFrame, bench_var: str, strat_var: str
) -> alt.Chart:
    """Compares model performance across multiple variables"""

    return (
        alt.Chart(gs_outputs)
        .mark_boxplot()
        .encode(
            row=alt.Row(
                strat_var,
                sort=alt.EncodingSortField(bench_var, "median", order="descending"),
                header=alt.Header(labelOrient="top"),
            ),
            color=alt.Color("name", title="Indicator source"),
            y=alt.Y("name", title=None),
            x=alt.X(bench_var, scale=alt.Scale(type="symlog")),
        )
    )


def choro_plot(
    geo_df: gp.GeoDataFrame,
    count_var: str,
    count_var_name: str,
    region_name: str = "region",
    scheme: str = "spectral",
    scale_type: str = "linear",
) -> alt.Chart:
    """Plot an altair choropleth

    Args:
        geo_df: geodf
        count_var: name of the variable we are plotting
        count_var_name: clean name for the count variable
        region_name is the name of the region variable
        scheme: is the colour scheme. Defaults to spectral
        scale_type: is the type of scale we are using. Defaults to log

    Returns:
        An altair chart

    """

    base_map = (  # Base chart with outlines
        alt.Chart(alt.Data(values=json.loads(geo_df.to_json())["features"]))
        .project(type="mercator")
        .mark_geoshape(filled=False, stroke="gray")
    )

    choropleth = (  # Filled polygons and tooltip
        base_map.transform_calculate(region=f"datum.properties.{region_name}")
        .mark_geoshape(filled=True, stroke="darkgrey", strokeWidth=0.2)
        .encode(
            size=f"properties.{count_var}:N",
            color=alt.Color(
                f"properties.{count_var}:N",
                title=count_var_name,
                scale=alt.Scale(scheme=scheme, type=scale_type),
                sort="ascending",
            ),
            tooltip=[
                "region:N",
                alt.Tooltip(f"properties.{count_var}:Q", format="1.2f"),
            ],
        )
    )

    return base_map + choropleth


def nuts_plot(lad_clusters: pd.DataFrame, color="nuts") -> alt.Chart:
    """Plots the distribution of LADs by sector and NUTS
    User can choose whether they want to colour by nuts or cluste
    """

    if color == "nuts":
        nuts_1_distr = (
            lad_clusters.groupby("la_cluster")["nuts1"]
            .value_counts(normalize=True)
            .reset_index(name="share")
        )

        return (
            alt.Chart(nuts_1_distr)
            .mark_bar()
            .encode(
                y="la_cluster:N",
                x="share",
                color=alt.Color("nuts1", scale=alt.Scale(scheme="tableau20")),
                tooltip=["la_cluster", "nuts1"],
            )
        )
    else:
        la_cluster_distr = (
            lad_clusters.groupby("nuts1")["la_cluster"]
            .value_counts(normalize=True)
            .reset_index(name="share")
        )

        return (
            alt.Chart(la_cluster_distr)
            .mark_bar()
            .encode(
                y="nuts1:N",
                x="share",
                color=alt.Color("la_cluster:N", scale=alt.Scale(scheme="tableau20")),
                tooltip=["la_cluster", "nuts1"],
            )
        )
