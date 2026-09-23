import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def load_csv(file):
    return pd.read_csv(file)


def get_basic_summary(df):
    return df.describe(include="all")


def get_numeric_columns(df):
    return (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )


def get_correlation_matrix(df):
    numeric_df = df.select_dtypes(include="number")

    return numeric_df.corr()


def get_target_correlations(df, target_column):
    numeric_df = df.select_dtypes(include="number")

    if target_column not in numeric_df.columns:
        return None

    correlations = (
        numeric_df.corr()[target_column]
        .drop(target_column)
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    return correlations


def run_linear_regression(
    df,
    target_column,
    feature_columns
):
    data = df[
        feature_columns + [target_column]
    ].dropna()

    X = data[feature_columns]
    y = data[target_column]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    predictions = model.predict(
        X
    )

    r2 = r2_score(
        y,
        predictions
    )

    mse = mean_squared_error(
        y,
        predictions
    )

    coefficients = pd.Series(
        model.coef_,
        index=feature_columns,
        name="coefficient"
    )

    results = pd.DataFrame({
        "actual": y,
        "predicted": predictions,
        "residual": y - predictions
    })

    return {
        "model": model,
        "r2": r2,
        "mse": mse,
        "intercept": model.intercept_,
        "coefficients": coefficients,
        "results": results
    }