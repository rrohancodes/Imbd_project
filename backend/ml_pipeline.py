import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# Custom transformer for director frequency
class DirectorFrequencyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel())

        counts = s.value_counts()

        self.freq_map_ = (counts / len(s)).to_dict()

        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())

        return (
            s.map(self.freq_map_)
            .fillna(0.0)
            .values.reshape(-1, 1)
        )


# Convert overview column into text
def select_text_column(X):

    if isinstance(X, pd.DataFrame):
        s = X.iloc[:, 0]

    elif isinstance(X, pd.Series):
        s = X

    else:
        arr = np.asarray(X)

        if arr.ndim == 2:
            s = pd.Series(arr[:, 0])
        else:
            s = pd.Series(arr.ravel())

    return s.fillna("").astype(str)


# Build preprocessing pipeline
def make_preprocessor():

    numeric_features = [
        "Meta_score",
        "No_of_Votes",
        "Gross",
        "Runtime",
        "Released_Year"
    ]

    categorical_features = [
        "Genre",
        "Certificate"
    ]

    director_feature = ["Director"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    director_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq", DirectorFrequencyEncoder())
    ])

    text_transformer = Pipeline(steps=[
        ("select", FunctionTransformer(
            select_text_column,
            validate=False
        )),
        ("tfidf", TfidfVectorizer(
            max_features=300,
            stop_words="english"
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("dir", director_transformer, director_feature),
            ("text", text_transformer, ["Overview"])
        ]
    )

    return preprocessor, numeric_features, categorical_features