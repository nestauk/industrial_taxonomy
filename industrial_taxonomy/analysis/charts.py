import altair as alt
import numpy as np
import pandas as pd


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
            (alt.datum.value > 0.6) | (alt.datum.value < 0.1),
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
