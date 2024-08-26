import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://homl.info/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")

    return [
        pd.read_csv(Path("datasets/titanic/train.csv")),
        pd.read_csv(Path("datasets/titanic/test.csv")),
    ]


def preprocess_feats(feats: pd.DataFrame):
    feats = feats.set_index("passengerid")

    def categorize_column(df, column="sex"):
        return pd.get_dummies(df[column], drop_first=True)

    feats["sex"] = categorize_column(feats, "sex")

    feats.loc[feats.cabin.isna(), "cabin"] = "no_cabin"

    oh_encoder = OneHotEncoder()
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    cat_pipeline = Pipeline([("oh", oh_encoder), ("imputer", cat_imputer)])
    num_pipeline = Pipeline([("imputer", num_imputer), ("scaler", scaler)])

    categorical_columns = feats.select_dtypes(include=[object, bool]).columns
    numerical_columns = feats.columns.difference(categorical_columns)

    col_transformer = ColumnTransformer(
        [
            ("categorical", cat_pipeline, categorical_columns),
            ("numerical", num_pipeline, numerical_columns),
        ]
    )

    preprocessed_feats = col_transformer.fit_transform(feats)

    return preprocessed_feats
