import os
import joblib
import kagglehub
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from ml_pipeline import make_preprocessor


# ==================================
# PATH CONFIGURATION
# ==================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ==================================
# LOAD DATASET
# ==================================

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"
)

df = pd.read_csv(path + "/imdb_top_1000.csv")

df.columns = df.columns.str.strip()


# ==================================
# SELECT FEATURES
# ==================================

features = [
    "Meta_score",
    "No_of_Votes",
    "Gross",
    "Runtime",
    "Released_Year",
    "Genre",
    "Certificate",
    "Director",
    "Overview"
]

target = "IMDB_Rating"

df = df[features + [target]].copy()


# ==================================
# DATA CLEANING
# ==================================

df["Gross"] = (
    df["Gross"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

df["Runtime"] = (
    df["Runtime"]
    .astype(str)
    .str.replace(" min", "", regex=False)
)

df["Runtime"] = pd.to_numeric(df["Runtime"], errors="coerce")

df["Released_Year"] = pd.to_numeric(
    df["Released_Year"], errors="coerce"
)

df["Meta_score"] = pd.to_numeric(
    df["Meta_score"], errors="coerce"
)

df["No_of_Votes"] = pd.to_numeric(
    df["No_of_Votes"], errors="coerce"
)


# Apply log transformation
df["No_of_Votes"] = np.log1p(df["No_of_Votes"])
df["Gross"] = np.log1p(df["Gross"])

df = df.dropna().copy()


# ==================================
# CREATE CLASSIFICATION TARGET
# ==================================

threshold = df["IMDB_Rating"].median()

df["Rating_Class"] = np.where(
    df["IMDB_Rating"] > threshold,
    "High",
    "Low"
)

X = df.drop(columns=["IMDB_Rating", "Rating_Class"])

y_class = df["Rating_Class"]
y_reg = df["IMDB_Rating"]


# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train_class, y_test_class = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

y_train_reg = y_reg.loc[X_train.index]
y_test_reg = y_reg.loc[X_test.index]


# ==================================
# CLASSIFICATION MODEL
# ==================================

print("\nTraining classification model...")

classification_preprocessor, _, _ = make_preprocessor()

classification_model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

classification_pipeline = Pipeline(steps=[
    ("preprocess", classification_preprocessor),
    ("model", classification_model)
])

classification_pipeline.fit(X_train, y_train_class)


# ==================================
# REGRESSION MODEL
# ==================================

print("\nTraining regression model...")

regression_preprocessor, _, _ = make_preprocessor()

regression_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

regression_pipeline = Pipeline(steps=[
    ("preprocess", regression_preprocessor),
    ("model", regression_model)
])

regression_pipeline.fit(X_train, y_train_reg)


# ==================================
# MODEL EVALUATION
# ==================================

print("\n========== CLASSIFICATION RESULTS ==========")

y_pred_class = classification_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class))


print("\n========== REGRESSION RESULTS ==========")

y_pred_reg = regression_pipeline.predict(X_test)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==================================
# SAVE MODELS
# ==================================

joblib.dump(
    classification_pipeline,
    MODEL_DIR / "classification_model.joblib"
)

joblib.dump(
    regression_pipeline,
    MODEL_DIR / "regression_model.joblib"
)

joblib.dump(
    {"threshold": float(threshold)},
    MODEL_DIR / "model_metadata.joblib"
)

print("\nModels saved successfully!")