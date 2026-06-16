"""Leakage-safe industrial ML pipeline for predicting white_invert.

Run from the repository root or NN directory:
    python NN/industrial_white_invert_pipeline.py

Outputs are written to NN/reports/industrial_white_invert/:
    - monthly_trend_white_invert.png
    - model_performance_comparison.png
    - feature_importance_white_invert.png
    - model_performance.csv
    - feature_importance.csv
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow this file to run both as ``python NN/script.py`` and from inside NN.
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from read_data import safe_numeric_conversion  # noqa: E402
from refinery_variables import filter_allowed_model_inputs, validate_model_inputs  # noqa: E402

RANDOM_STATE = 42
TARGET = "white_invert"
REPORT_DIR = MODULE_DIR / "reports" / "industrial_white_invert"
SOURCE_EXCEL = MODULE_DIR / "convert" / "gozaresh.xlsx"
DATASET_CSV = MODULE_DIR / "convert" / "sugar_all.csv"


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def load_refinery_data() -> pd.DataFrame:
    """Load the sugar-refinery data, rebuilding the CSV from Excel if necessary."""
    if DATASET_CSV.exists():
        return pd.read_csv(DATASET_CSV)

    # convert2.py is intentionally verbose; capture debug logs to keep this
    # industrial pipeline output readable and sectioned.
    import importlib.util

    spec = importlib.util.spec_from_file_location("sugar_report_converter", MODULE_DIR / "convert" / "convert2.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load NN/convert/convert2.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    with contextlib.redirect_stdout(io.StringIO()):
        data = converter.extract_excel_to_dataframe(SOURCE_EXCEL)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATASET_CSV, index=False)
    return data


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create allowed calendar features known before final QC is measured."""
    out = df.copy()
    out["year"] = pd.to_numeric(out.get("report_date_year"), errors="coerce")
    out["month"] = pd.to_numeric(out.get("report_date_month"), errors="coerce")
    out["day"] = pd.to_numeric(out.get("report_date_day"), errors="coerce")
    # A deterministic ordinal is sufficient for chronological splitting/trends;
    # the source calendar is Jalali, so we avoid pretending it is Gregorian.
    out["process_ordinal"] = out["year"].fillna(0) * 10_000 + out["month"].fillna(0) * 100 + out["day"].fillna(0)
    out["day_of_week"] = (out["process_ordinal"].rank(method="dense").astype(int) - 1) % 7
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["day_sin"] = np.sin(2 * np.pi * out["day"] / 31)
    out["day_cos"] = np.cos(2 * np.pi * out["day"] / 31)
    out["month_label"] = out["year"].fillna(0).astype(int).astype(str) + "-" + out["month"].fillna(0).astype(int).astype(str).str.zfill(2)
    return out


def prepare_leakage_safe_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    if TARGET not in df.columns:
        raise ValueError(f"Required target column not found: {TARGET}")
    data = add_calendar_features(df)
    y = pd.to_numeric(data[TARGET], errors="coerce")
    feature_names = filter_allowed_model_inputs(data.columns, output_features=[TARGET])
    validate_model_inputs(feature_names, output_features=[TARGET])
    X_raw = data[feature_names].copy()
    numeric_cols = [c for c in X_raw.columns if c != "shift_name"]
    X_raw[numeric_cols] = safe_numeric_conversion(X_raw[numeric_cols])
    modeling = pd.concat([X_raw, y.rename(TARGET), data[["process_ordinal", "month_label"]]], axis=1)
    modeling = modeling.dropna(subset=[TARGET]).sort_values(["process_ordinal", "shift_name"], kind="mergesort")
    X = modeling[feature_names]
    y = modeling[TARGET]
    metadata = modeling[["process_ordinal", "month_label"]]
    return X, y, metadata, feature_names


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [c for c in X.columns if X[c].dtype == "object" or c == "shift_name"]
    numeric_features = [c for c in X.columns if c not in categorical_features]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # Older sklearn compatibility.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_candidates() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=20_000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, l2_regularization=0.1, random_state=RANDOM_STATE),
    }


def chronological_split(X: pd.DataFrame, y: pd.Series, test_fraction: float = 0.2) -> SplitData:
    test_size = max(8, int(np.ceil(len(X) * test_fraction)))
    if len(X) <= test_size + 10:
        raise ValueError("Not enough non-null target rows for a leakage-safe chronological train/test split.")
    return SplitData(X.iloc[:-test_size], X.iloc[-test_size:], y.iloc[:-test_size], y.iloc[-test_size:])


def evaluate_models(split: SplitData) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows = []
    fitted = {}
    n_splits = min(5, max(2, len(split.X_train) // 12))
    cv = TimeSeriesSplit(n_splits=n_splits)
    for name, estimator in model_candidates().items():
        pipe = Pipeline([("preprocess", make_preprocessor(split.X_train)), ("model", estimator)])
        cv_scores = cross_validate(
            pipe,
            split.X_train,
            split.y_train,
            cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            error_score="raise",
        )
        pipe.fit(split.X_train, split.y_train)
        pred = pipe.predict(split.X_test)
        fitted[name] = pipe
        rows.append({
            "model": name,
            "cv_r2_mean": np.mean(cv_scores["test_r2"]),
            "cv_rmse_mean": -np.mean(cv_scores["test_rmse"]),
            "cv_mae_mean": -np.mean(cv_scores["test_mae"]),
            "holdout_r2": r2_score(split.y_test, pred),
            "holdout_rmse": float(np.sqrt(mean_squared_error(split.y_test, pred))),
            "holdout_mae": mean_absolute_error(split.y_test, pred),
        })
    results = pd.DataFrame(rows).sort_values(["holdout_rmse", "holdout_mae"], ascending=True)
    return results, fitted


def plot_monthly_trend(df: pd.DataFrame) -> Path:
    trend = df.dropna(subset=[TARGET]).groupby("month_label", as_index=False)[TARGET].mean()
    path = REPORT_DIR / "monthly_trend_white_invert.png"
    plt.figure(figsize=(10, 5))
    plt.plot(trend["month_label"], trend[TARGET], marker="o", linewidth=2)
    plt.title("Monthly Trend of white_invert")
    plt.xlabel("Report month")
    plt.ylabel("Mean white_invert")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_model_performance(results: pd.DataFrame) -> Path:
    path = REPORT_DIR / "model_performance_comparison.png"
    ordered = results.sort_values("holdout_rmse", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.barh(ordered["model"], ordered["holdout_rmse"], color="#4472C4")
    plt.title("Model Performance Comparison (lower RMSE is better)")
    plt.xlabel("Chronological holdout RMSE")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_feature_importance(best_pipeline: Pipeline, split: SplitData) -> tuple[Path, pd.DataFrame]:
    importance = permutation_importance(
        best_pipeline,
        split.X_test,
        split.y_test,
        n_repeats=25,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    table = pd.DataFrame({
        "feature": split.X_test.columns,
        "importance_mean_rmse_reduction": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean_rmse_reduction", ascending=False)
    path = REPORT_DIR / "feature_importance_white_invert.png"
    top = table.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance_mean_rmse_reduction"], xerr=top["importance_std"], color="#70AD47")
    plt.title("Permutation Feature Importance for white_invert")
    plt.xlabel("RMSE reduction when feature is available")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path, table


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("1) Load refinery data")
    raw = load_refinery_data()
    print(f"Rows: {len(raw):,} | Columns: {len(raw.columns):,}")

    print_section("2) Leakage-safe feature selection")
    X, y, metadata, feature_names = prepare_leakage_safe_xy(raw)
    print(f"Target: {TARGET}")
    print(f"Usable target rows: {len(y):,}")
    print(f"Approved input features ({len(feature_names)}): {', '.join(feature_names)}")
    print("Leakage policy: X contains only early/process calendar variables and operator controls; final white QC, averages, future-quality outputs, and the target itself are excluded.")

    plot_df = pd.concat([y.rename(TARGET), metadata], axis=1)
    monthly_plot = plot_monthly_trend(plot_df)

    print_section("3) Chronological validation and sklearn pipelines")
    split = chronological_split(X, y)
    print(f"Train rows: {len(split.X_train):,} | Holdout rows: {len(split.X_test):,}")
    print("Validation: TimeSeriesSplit on training data + final chronological holdout test.")
    results, fitted = evaluate_models(split)
    results_path = REPORT_DIR / "model_performance.csv"
    results.to_csv(results_path, index=False)
    print(results.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
    perf_plot = plot_model_performance(results)

    print_section("4) Best model and feature importance")
    best_name = results.iloc[0]["model"]
    best_pipeline = fitted[best_name]
    importance_plot, importance_table = plot_feature_importance(best_pipeline, split)
    importance_path = REPORT_DIR / "feature_importance.csv"
    importance_table.to_csv(importance_path, index=False)
    print(f"Best model: {best_name}")
    print(importance_table.head(12).to_string(index=False, float_format=lambda v: f"{v:0.6f}"))

    print_section("5) Output files")
    for label, path in [
        ("Monthly trend plot", monthly_plot),
        ("Model performance plot", perf_plot),
        ("Feature importance plot", importance_plot),
        ("Model performance table", results_path),
        ("Feature importance table", importance_path),
    ]:
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
