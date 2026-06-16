"""Preprocessing helpers for refinery ML pipelines."""

import inspect
import logging
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

LOGGER = logging.getLogger(__name__)

_NUMERIC_TOKEN_RE = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def coerce_refinery_numeric_series(series, column_name=None, logger=None, max_logged_rows=20):
    """Convert refinery values to floats with robust numeric extraction.

    Persian/other text-only values such as ``بومه شیرآهک`` become NaN. Mixed
    strings with numbers keep the first numeric token, and simple numeric ranges
    such as ``10-12`` are converted to their midpoint.
    """
    logger = logger or LOGGER
    column_label = column_name or getattr(series, "name", None) or "<unnamed>"
    problematic = []

    def _convert(index, value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        text = str(value).strip()
        if not text:
            return np.nan

        normalized = (
            text.replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace("٫", ".")
            .replace("٬", "")
            .replace(",", "")
            .strip()
        )
        # Common spreadsheet/export artifacts: [1.366046E1], (1.23), etc.
        if re.fullmatch(r"\[\s*[^\[\]]+\s*\]", normalized):
            normalized = normalized[1:-1].strip()
        negative_parenthesized = re.fullmatch(r"\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)", normalized)
        if negative_parenthesized:
            return -float(negative_parenthesized.group(1))
        direct_value = pd.to_numeric(normalized, errors="coerce")
        if pd.notna(direct_value):
            return float(direct_value)

        range_match = re.fullmatch(
            r"\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*",
            normalized,
        )
        if range_match:
            low, high = map(float, range_match.groups())
            return (low + high) / 2.0

        tokens = _NUMERIC_TOKEN_RE.findall(normalized)
        if tokens:
            return float(tokens[0])

        problematic.append((index, value))
        return np.nan

    converted = pd.Series(
        (_convert(index, value) for index, value in series.items()),
        index=series.index,
        name=series.name,
        dtype="float64",
    )
    if problematic:
        preview = problematic[:max_logged_rows]
        logger.warning(
            "Column %s: converted %s non-numeric value(s) to NaN. Sample rows: %s",
            column_label,
            len(problematic),
            preview,
        )
    return converted


def coerce_refinery_numeric_frame(dataframe, logger=None):
    """Convert every dataframe column with the refinery numeric parser."""
    logger = logger or LOGGER
    return pd.DataFrame(
        {
            column: coerce_refinery_numeric_series(dataframe[column], column_name=column, logger=logger)
            for column in dataframe.columns
        },
        index=dataframe.index,
    )


def make_median_imputer():
    """Create a median imputer that preserves empty columns when supported."""
    kwargs = {"strategy": "median"}
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        kwargs["keep_empty_features"] = True
    return SimpleImputer(**kwargs)


def median_impute_train_test(X_train, X_test, logger=None):
    """Fit median imputation on training data and guarantee no NaNs remain."""
    logger = logger or LOGGER
    X_train_numeric = coerce_refinery_numeric_frame(X_train, logger=logger).replace([np.inf, -np.inf], np.nan)
    X_test_numeric = coerce_refinery_numeric_frame(X_test, logger=logger).replace([np.inf, -np.inf], np.nan)
    X_test_numeric = X_test_numeric.reindex(columns=X_train_numeric.columns)

    all_missing = X_train_numeric.columns[X_train_numeric.isna().all()].tolist()
    if all_missing:
        logger.warning(
            "Columns with all training values missing will be filled with 0.0 before median imputation: %s",
            all_missing,
        )
        X_train_numeric.loc[:, all_missing] = 0.0
        X_test_numeric.loc[:, all_missing] = X_test_numeric.loc[:, all_missing].fillna(0.0)

    imputer = make_median_imputer()
    train_arr = imputer.fit_transform(X_train_numeric)
    test_arr = imputer.transform(X_test_numeric)
    X_train_imputed = pd.DataFrame(train_arr, columns=X_train_numeric.columns, index=X_train_numeric.index)
    X_test_imputed = pd.DataFrame(test_arr, columns=X_train_numeric.columns, index=X_test_numeric.index)

    remaining_train = int(X_train_imputed.isna().sum().sum())
    remaining_test = int(X_test_imputed.isna().sum().sum())
    if remaining_train or remaining_test:
        logger.warning(
            "Median imputation left NaNs (train=%s, test=%s); filling remaining values with 0.0.",
            remaining_train,
            remaining_test,
        )
        X_train_imputed = X_train_imputed.fillna(0.0)
        X_test_imputed = X_test_imputed.fillna(0.0)

    return X_train_imputed.astype(float), X_test_imputed.astype(float), imputer


def _offending_rows_dataframe(df, mask, max_rows=5):
    rows = df.loc[mask].head(max_rows).copy()
    rows.insert(0, "__row_index__", rows.index)
    return rows


def validate_dataframe(df, name, require_numeric=True, max_rows=5):
    """Validate a DataFrame and report shape, dtypes, NaN/inf counts, and offenders."""
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    numeric_df = frame.select_dtypes(include=[np.number])
    non_numeric_columns = frame.columns.difference(numeric_df.columns).tolist()
    nan_mask = frame.isna().any(axis=1)
    inf_counts = pd.Series(dtype=int)
    inf_mask = pd.Series(False, index=frame.index)
    if not numeric_df.empty:
        inf_frame = np.isinf(numeric_df.to_numpy(dtype=float))
        inf_counts = pd.Series(inf_frame.sum(axis=0), index=numeric_df.columns)
        inf_counts = inf_counts[inf_counts > 0]
        inf_mask = pd.Series(inf_frame.any(axis=1), index=frame.index)
    report = {
        "name": name,
        "shape": frame.shape,
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "nan_count": int(frame.isna().sum().sum()),
        "inf_count": int(inf_counts.sum()) if not inf_counts.empty else 0,
        "non_numeric_columns": non_numeric_columns,
        "sample_offending_rows": _offending_rows_dataframe(frame, nan_mask | inf_mask, max_rows=max_rows),
    }
    LOGGER.info(
        "Validation %s: shape=%s nan_count=%s inf_count=%s non_numeric_columns=%s",
        name, report["shape"], report["nan_count"], report["inf_count"], non_numeric_columns,
    )
    if not report["sample_offending_rows"].empty:
        LOGGER.warning("Sample offending rows for %s:\n%s", name, report["sample_offending_rows"].to_string(index=False))
    if require_numeric and non_numeric_columns:
        raise ValueError(f"{name} contains non-numeric columns: {non_numeric_columns}")
    if report["nan_count"]:
        raise ValueError(f"{name} contains {report['nan_count']} NaN value(s).")
    if report["inf_count"]:
        raise ValueError(f"{name} contains {report['inf_count']} infinite value(s).")
    return report


def validate_numpy(arr, name, max_rows=5):
    """Validate a numpy-like array and report shape, dtype, NaN/inf counts, and offenders."""
    array = np.asarray(arr)
    numeric = np.issubdtype(array.dtype, np.number)
    coerced = array.astype(float) if numeric else pd.DataFrame(array.reshape(array.shape[0], -1) if array.ndim else [array]).apply(pd.to_numeric, errors="coerce").to_numpy()
    nan_mask_flat = pd.isna(coerced)
    inf_mask_flat = np.isinf(coerced) if numeric or coerced.dtype.kind in "fc" else np.zeros_like(nan_mask_flat, dtype=bool)
    LOGGER.info("Validation %s: shape=%s dtype=%s nan_count=%s inf_count=%s", name, array.shape, array.dtype, int(nan_mask_flat.sum()), int(inf_mask_flat.sum()))
    if (not numeric) or nan_mask_flat.any() or inf_mask_flat.any():
        rows = np.unique(np.where((nan_mask_flat | inf_mask_flat).reshape(coerced.shape))[0])[:max_rows].tolist() if coerced.ndim else [0]
        raise ValueError(f"{name} invalid: numeric={numeric}, offending row indexes={rows}")
    return {"name": name, "shape": array.shape, "dtype": str(array.dtype), "nan_count": int(nan_mask_flat.sum()), "inf_count": int(inf_mask_flat.sum()), "non_numeric_columns": []}


def validate_tensor(tensor, name, max_rows=5):
    """Validate a torch tensor and report shape, dtype, NaN/inf counts, and offenders."""
    import torch
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name} is not a torch tensor.")
    finite = torch.isfinite(tensor) if tensor.is_floating_point() or tensor.is_complex() else torch.ones_like(tensor, dtype=torch.bool)
    bad_count = int((~finite).sum().item())
    LOGGER.info("Validation %s: shape=%s dtype=%s nan_or_inf_count=%s", name, tuple(tensor.shape), tensor.dtype, bad_count)
    if bad_count:
        bad_rows = torch.unique(torch.nonzero(~finite, as_tuple=False)[:, 0])[:max_rows].cpu().tolist() if tensor.ndim else [0]
        raise ValueError(f"{name} contains NaN or infinite values; offending row indexes={bad_rows}")
    return {"name": name, "shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "nan_count": bad_count, "inf_count": bad_count, "non_numeric_columns": []}
