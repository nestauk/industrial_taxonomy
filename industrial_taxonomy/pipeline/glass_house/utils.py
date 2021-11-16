from pandas import DataFrame


def get_top_matches(df: DataFrame) -> DataFrame:
    """For each y get the x and sim_mean corresponding to highest sim_mean."""
    return df.loc[df.groupby("y").sim_mean.idxmax()]


def _filter_cols(df: DataFrame) -> DataFrame:
    # Optimisation - keep used cols only
    return df.loc[:, ("x", "y", "sim_mean")]


def _drop_poor_matches(df: DataFrame) -> DataFrame:
    return df.pipe(lambda x: x.loc[x.groupby("y").sim_mean.idxmax()])


def optimise_chunk(df: DataFrame) -> DataFrame:
    """Don't persist data we won't use - select minimal cols and drop poor matches."""
    return df.pipe(_filter_cols).set_index("y").pipe(_drop_poor_matches)
