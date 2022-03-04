import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from industrial_taxonomy.utils.metaflow import get_run


def local_predictive(
    secondary_table: pd.DataFrame,
    sector_distr: pd.DataFrame,
    model: BaseEstimator,
    param_grid: dict,
    cv: int = 3,
    test: bool = True,
):
    """Calculate predictive performance on secondary data of
    sectoral distributions based on different taxonomies /
    implementations of the taxonomy

    Args:
        secondary_table: table with secondary data
        sector_distr: sector distribution according to taxonomy
        model: sklearn model we are using
        param_grid: parameters to perform grid search on
        cv: cross validation strategy
        test: run this for a few indicators

    Returns:
        Table with cross-validation and test results
    """

    results = []

    for n, b in enumerate(secondary_table["indicator"].unique()):
        logging.info(b)

        data = secondary_table.query(f"indicator=='{b}'").merge(
            sector_distr.reset_index(drop=False),
            left_on="la_code",
            right_on="laua_code",
        )

        y = data["zscore"]

        x = data[sector_distr.columns]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        scorer = make_scorer(mean_squared_error)

        reg = model()

        logging.info("Grid searching")
        clf = GridSearchCV(reg, param_grid=param_grid, cv=cv, scoring=scorer)
        clf.fit(x_train, y_train)

        best_score = clf.best_score_
        best_params = clf.best_params_
        best_model = clf.best_estimator_

        test_score = mean_squared_error(y_test, best_model.predict(x_test))
        test_r2 = r2_score(y_test, best_model.predict(x_test))

        results.append([b, best_score, best_params, test_score, test_r2])

        if test is True:
            if n == 3:
                break

    return results


def modelling_results(
    result_table: list, assign_share: str, modelling_approach: str
) -> pd.DataFrame:
    """Transforms modelling outputs into a pandas table"""

    return (
        pd.DataFrame(
            result_table,
            columns=["indicator", "cross_val_score", "params", "mse_test", "r2_test"],
        )
        .drop(axis=1, labels=["params"])
        .assign(share=assign_share)
        .assign(model=modelling_approach)
    )


def modelling_outputs():
    """Get modelling results"""

    return get_run("LocalBenchmark").data.results_table
