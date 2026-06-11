import os
import json
import re
import importlib.util
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from refinery_variables import (
    filter_allowed_model_inputs,
    refinery_variable_group_metadata,
    validate_model_inputs,
)

ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _normalize_numeric_text(value):
    """Return cleaned numeric text while preserving invalid values for NaN coercion."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    if not text:
        return pd.NA

    normalized = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("٫", ".")
        .replace(",", "")
        .strip()
    )

    bracket_match = re.fullmatch(r"\[\s*(.*?)\s*\]", normalized)
    if bracket_match:
        normalized = bracket_match.group(1).strip()

    range_match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*-\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*",
        normalized,
    )
    if range_match:
        low, high = map(float, range_match.groups())
        return (low + high) / 2.0

    return normalized


def safe_numeric_conversion(df):
    """Safely convert a dataframe to numeric values without changing column names."""
    numeric_df = pd.DataFrame(index=df.index, columns=df.columns)
    converted_columns = []
    failed_columns = []

    for column in df.columns:
        original = df[column]
        normalized = original.map(_normalize_numeric_text)
        converted = pd.to_numeric(normalized, errors="coerce")
        numeric_df[column] = converted

        original_non_null = original.notna()
        failed_mask = original_non_null & converted.isna()
        failed_count = int(failed_mask.sum())
        if failed_count:
            examples = original[failed_mask].astype(str).str.strip().drop_duplicates().head(3).tolist()
            failed_columns.append(f"{column}: {failed_count} ({examples})")

        if not pd.api.types.is_numeric_dtype(original) or failed_count:
            changed_mask = original_non_null & (
                original.astype(str).str.strip() != converted.astype(str).str.strip()
            )
            if int(changed_mask.sum()) or not pd.api.types.is_numeric_dtype(original):
                converted_columns.append(column)

    if converted_columns:
        print(
            "Converted columns to numeric values "
            "(scientific notation, whitespace, commas, bracketed values, and ranges handled): "
            + ", ".join(converted_columns)
        )
    if failed_columns:
        print(
            f"{ANSI_YELLOW}WARNING: failed numeric conversions converted to NaN: "
            + "; ".join(failed_columns)
            + f"{ANSI_RESET}"
        )

    return numeric_df


def _coerce_numeric_table(df, name):
    """Return a numeric copy of a prep-data table and report non-numeric cleanups."""
    numeric_df = safe_numeric_conversion(df)
    return numeric_df


def resolve_relative_path(path, base_dir=MODULE_DIR):
    """Resolve relative data paths from CWD first, then the NN module directory."""
    path_text = os.fspath(path)
    if os.path.isabs(path_text):
        return os.path.abspath(path_text)

    cwd_candidate = os.path.abspath(path_text)
    module_candidate = os.path.abspath(os.path.join(base_dir, path_text))

    if os.path.exists(cwd_candidate):
        return cwd_candidate
    if os.path.exists(module_candidate):
        return module_candidate
    return cwd_candidate


def _load_converter_module(script_name="convert2.py"):
    """Load a sugar Excel converter module without requiring package imports."""
    converter_path = Path(MODULE_DIR) / "convert" / script_name
    if not converter_path.is_file():
        raise FileNotFoundError(f"Sugar converter script was not found: {converter_path}")

    spec = importlib.util.spec_from_file_location("sugar_report_converter", converter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sugar converter script: {converter_path}")

    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    return converter


def ensure_dataset_csv_exists(dataset_path, source_excel_path=None):
    """Return an existing raw CSV path, rebuilding the default sugar CSV when possible.

    The project does not track generated CSV files, so a fresh checkout may have
    ``convert/gozaresh.xlsx`` but not ``convert/sugar_all.csv``.  When the
    default sugar dataset is requested and missing, rebuild it from the Excel
    report using the maintained converter before the prep-data rebuild reads it.
    """
    resolved_dataset_path = resolve_relative_path(dataset_path)
    if os.path.isfile(resolved_dataset_path):
        return resolved_dataset_path

    requested_name = os.path.basename(os.fspath(dataset_path))
    if requested_name != "sugar_all.csv":
        raise FileNotFoundError(f"Dataset file '{resolved_dataset_path}' not found.")

    using_default_excel_source = source_excel_path is None
    if source_excel_path is None:
        source_excel_path = os.path.join("convert", "gozaresh.xlsx")
    resolved_excel_path = resolve_relative_path(source_excel_path)
    if using_default_excel_source:
        resolved_dataset_path = os.path.join(os.path.dirname(resolved_excel_path), requested_name)
    if not os.path.isfile(resolved_excel_path):
        raise FileNotFoundError(
            f"Dataset file '{resolved_dataset_path}' not found, and source Excel "
            f"file '{resolved_excel_path}' was not found to rebuild it."
        )

    print(
        "[path-debug] Dataset CSV is missing; rebuilding default sugar dataset "
        f"from Excel report: {resolved_excel_path}"
    )

    try:
        converter = _load_converter_module("convert2.py")
        data = converter.extract_excel_to_dataframe(Path(resolved_excel_path))
    except ImportError as err:
        raise ImportError(
            "Dataset CSV is missing and could not be rebuilt because an Excel "
            "dependency is unavailable. Install project requirements, including "
            "openpyxl, or create convert/sugar_all.csv manually."
        ) from err

    os.makedirs(os.path.dirname(resolved_dataset_path), exist_ok=True)
    data.to_csv(resolved_dataset_path, index=False)
    print(f"[path-debug] Rebuilt dataset CSV: {resolved_dataset_path}")
    return resolved_dataset_path


def print_path_debug(label, path, resolved_path=None):
    """Print debug information for a path that may depend on the working directory."""
    path_text = os.fspath(path)
    if resolved_path is None:
        resolved_path = resolve_relative_path(path_text)
    cwd_candidate = path_text if os.path.isabs(path_text) else os.path.abspath(path_text)
    module_candidate = (
        path_text if os.path.isabs(path_text) else os.path.abspath(os.path.join(MODULE_DIR, path_text))
    )
    print(f"[path-debug] {label}")
    print(f"[path-debug]   cwd: {os.getcwd()}")
    print(f"[path-debug]   module_dir: {MODULE_DIR}")
    print(f"[path-debug]   requested: {path_text}")
    print(f"[path-debug]   cwd candidate: {cwd_candidate} (exists={os.path.exists(cwd_candidate)})")
    print(f"[path-debug]   module candidate: {module_candidate} (exists={os.path.exists(module_candidate)})")
    print(f"[path-debug]   resolved: {resolved_path} (exists={os.path.exists(resolved_path)})")
    return resolved_path


def _normalize_output_features(output_feature, all_columns):
    """Normalize output feature(s) to a validated list."""
    if output_feature is None:
        return [all_columns[-1]]

    if isinstance(output_feature, str):
        output_features = [output_feature]
    else:
        output_features = list(output_feature)

    missing_outputs = [col for col in output_features if col not in all_columns]
    if missing_outputs:
        raise ValueError(
            f"Target column(s) '{missing_outputs}' not found in dataset. "
            f"Available columns: {list(all_columns)}"
        )

    if len(output_features) == 0:
        raise ValueError("At least one output feature must be selected.")

    return output_features


def resolve_data_columns(data, output_feature=None, input_features=None, optional_future_quality_inputs=None):
    """Resolve targets and enforce refinery-safe input columns."""
    output_features = _normalize_output_features(output_feature, data.columns)

    if input_features is None:
        input_features = filter_allowed_model_inputs(
            data.columns,
            output_features=output_features,
            optional_future_quality_inputs=optional_future_quality_inputs,
        )
    else:
        input_features = list(input_features)

    missing_inputs = [col for col in input_features if col not in data.columns]
    if missing_inputs:
        raise ValueError(
            f"Input columns not found in dataset: {missing_inputs}. "
            f"Available columns: {list(data.columns)}"
        )

    validate_model_inputs(input_features, output_features=output_features, optional_future_quality_inputs=optional_future_quality_inputs)

    if len(input_features) == 0:
        raise ValueError(
            "No refinery-safe input columns were selected. Models may only use "
            "EARLY_VARIABLES + CONTROL_VARIABLES; future quality variables are "
            "blocked to prevent target leakage."
        )

    return output_features, input_features


TEMPORAL_DERIVED_COLUMN_PATTERN = re.compile(
    r".+__(diff|lag|seqdiff|ratio|accel|normchg|roll_mean|roll_std|roll_min|roll_max|roll_range|roll_slope)_\d+$"
)


def split_base_and_derived_input_features(data_columns, input_features):
    """Split user-provided inputs into dataset columns and temporal-derived columns."""
    if input_features is None:
        return None, []

    base_input_features = [col for col in input_features if col in data_columns]
    derived_input_features = [col for col in input_features if col not in data_columns]

    invalid_derived = [
        col for col in derived_input_features if not TEMPORAL_DERIVED_COLUMN_PATTERN.match(col)
    ]
    if invalid_derived:
        raise ValueError(
            f"Input columns not found in dataset and not recognized as temporal-derived columns: "
            f"{invalid_derived}. Available columns: {list(data_columns)}"
        )

    return base_input_features, derived_input_features


def prep_data_file_paths(prep_folder="prep_data"):
    """Return canonical prep_data file paths."""
    resolved_prep_folder = resolve_relative_path(prep_folder)
    return {
        "X_train": os.path.join(resolved_prep_folder, "X_train.csv"),
        "X_test": os.path.join(resolved_prep_folder, "X_test.csv"),
        "y_train": os.path.join(resolved_prep_folder, "y_train.csv"),
        "y_test": os.path.join(resolved_prep_folder, "y_test.csv"),
        "metadata": os.path.join(resolved_prep_folder, "metadata.json"),
    }


def prep_data_exists(prep_folder="prep_data"):
    """Check whether all required prep_data CSV files exist."""
    paths = prep_data_file_paths(prep_folder)
    required = [paths["X_train"], paths["X_test"], paths["y_train"], paths["y_test"]]
    exists = all(os.path.isfile(path) for path in required)
    print(f"[path-debug] prep_data folder resolved to: {os.path.dirname(paths['X_train'])}")
    print(f"[path-debug] prep_data required files exist: {exists}")
    if not exists:
        missing = [path for path in required if not os.path.isfile(path)]
        print(f"[path-debug] prep_data missing required files: {missing}")
    return exists


def save_prep_metadata(prep_folder="prep_data", metadata=None):
    """Save prep-data configuration metadata as JSON."""
    if metadata is None:
        metadata = {}
    metadata_path = prep_data_file_paths(prep_folder)["metadata"]
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)


def read_prep_metadata(prep_folder="prep_data"):
    """Read prep-data metadata JSON if available."""
    metadata_path = prep_data_file_paths(prep_folder)["metadata"]
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)





def _append_engineered_input(resolved_inputs, column_name):
    """Append a generated feature name once while preserving feature order."""
    if column_name not in resolved_inputs:
        resolved_inputs.append(column_name)


def _safe_divide(numerator, denominator):
    """Divide process signals while treating zero denominators as missing values."""
    return numerator / denominator.mask(denominator == 0)


def _normalize_sequence_features(data, input_features, sequential_features=None, sequential_groups=None):
    """Return valid row-sequence features and ordered process groups.

    ``sequential_features`` represent observations that have meaningful previous
    rows (batch/shift/day history). ``sequential_groups`` represent ordered
    refinery process locations such as raw syrup -> carbonated -> sulphited.
    Both are restricted to currently selected, refinery-safe model inputs.
    """
    resolved_input_set = set(input_features)
    sequential_features = [
        col for col in (sequential_features or [])
        if col in data.columns and col in resolved_input_set
    ]

    normalized_groups = []
    for group in sequential_groups or []:
        normalized_group = [
            col for col in group
            if col in data.columns and col in resolved_input_set
        ]
        if len(normalized_group) >= 2:
            normalized_groups.append(normalized_group)

    return sequential_features, normalized_groups


def create_lag_window_features(data, resolved_inputs, sequential_features, lag_steps=(1, 2, 3)):
    """Create lag-window memory features such as ``feature__lag_1``.

    Lag features let tabular models see recent process history. The default
    window explicitly supports lag_1, lag_2, and lag_3 as requested.
    """
    transformed = data.copy()
    for feature in sequential_features:
        for lag_step in sorted({int(step) for step in lag_steps if int(step) > 0}):
            lag_col = f"{feature}__lag_{lag_step}"
            transformed[lag_col] = transformed[feature].shift(lag_step)
            _append_engineered_input(resolved_inputs, lag_col)
    return transformed, resolved_inputs


def create_time_sequence_process_features(
    data,
    resolved_inputs,
    sequential_features,
    sequential_groups,
    add_differences=False,
    difference_order=1,
    add_ratio_features=False,
    add_acceleration_features=False,
    add_normalized_change_features=False,
):
    """Create row-to-row and ordered-process sequence features.

    The row-to-row features learn process movement over time. The grouped
    features learn how a measurement changes between ordered refinery stages.
    """
    transformed = data.copy()

    if add_differences:
        if difference_order < 1:
            raise ValueError("difference_order must be >= 1.")
        for feature in sequential_features:
            source_col = feature
            for order in range(1, difference_order + 1):
                diff_col = f"{feature}__diff_{order}"
                transformed[diff_col] = transformed[source_col].diff()
                source_col = diff_col
                _append_engineered_input(resolved_inputs, diff_col)
        for group in sequential_groups:
            for prev_feature, curr_feature in zip(group[:-1], group[1:]):
                source_col = transformed[curr_feature] - transformed[prev_feature]
                for order in range(1, difference_order + 1):
                    seq_diff_col = f"{curr_feature}__minus_{prev_feature}__seqdiff_{order}"
                    transformed[seq_diff_col] = source_col if order == 1 else transformed[
                        f"{curr_feature}__minus_{prev_feature}__seqdiff_{order - 1}"
                    ].diff()
                    _append_engineered_input(resolved_inputs, seq_diff_col)

    if add_ratio_features:
        for feature in sequential_features:
            ratio_col = f"{feature}__ratio_1"
            transformed[ratio_col] = _safe_divide(transformed[feature], transformed[feature].shift(1))
            _append_engineered_input(resolved_inputs, ratio_col)
        for group in sequential_groups:
            for prev_feature, curr_feature in zip(group[:-1], group[1:]):
                ratio_col = f"{curr_feature}__over_{prev_feature}__ratio_1"
                transformed[ratio_col] = _safe_divide(transformed[curr_feature], transformed[prev_feature])
                _append_engineered_input(resolved_inputs, ratio_col)

    if add_acceleration_features:
        for feature in sequential_features:
            accel_col = f"{feature}__accel_1"
            transformed[accel_col] = (
                transformed[feature]
                - 2 * transformed[feature].shift(1)
                + transformed[feature].shift(2)
            )
            _append_engineered_input(resolved_inputs, accel_col)
        for group in sequential_groups:
            for prev2_feature, prev1_feature, curr_feature in zip(group[:-2], group[1:-1], group[2:]):
                accel_col = f"{curr_feature}__accel_from_{prev1_feature}_{prev2_feature}__accel_1"
                transformed[accel_col] = (
                    transformed[curr_feature]
                    - 2 * transformed[prev1_feature]
                    + transformed[prev2_feature]
                )
                _append_engineered_input(resolved_inputs, accel_col)

    if add_normalized_change_features:
        for feature in sequential_features:
            normchg_col = f"{feature}__normchg_1"
            transformed[normchg_col] = _safe_divide(
                transformed[feature] - transformed[feature].shift(1),
                transformed[feature].shift(1),
            )
            _append_engineered_input(resolved_inputs, normchg_col)
        for group in sequential_groups:
            for prev_feature, curr_feature in zip(group[:-1], group[1:]):
                normchg_col = f"{curr_feature}__minus_{prev_feature}__normchg_1"
                transformed[normchg_col] = _safe_divide(
                    transformed[curr_feature] - transformed[prev_feature],
                    transformed[prev_feature],
                )
                _append_engineered_input(resolved_inputs, normchg_col)

    return transformed, resolved_inputs


def create_rolling_process_dynamics_features(data, resolved_inputs, sequential_features, rolling_window=3):
    """Create rolling dynamics features over recent refinery observations."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be >= 2.")

    transformed = data.copy()
    for feature in sequential_features:
        rolling = transformed[feature].rolling(window=rolling_window, min_periods=rolling_window)

        mean_col = f"{feature}__roll_mean_{rolling_window}"
        std_col = f"{feature}__roll_std_{rolling_window}"
        min_col = f"{feature}__roll_min_{rolling_window}"
        max_col = f"{feature}__roll_max_{rolling_window}"
        range_col = f"{feature}__roll_range_{rolling_window}"
        slope_col = f"{feature}__roll_slope_{rolling_window}"

        transformed[mean_col] = rolling.mean()
        transformed[std_col] = rolling.std()
        transformed[min_col] = rolling.min()
        transformed[max_col] = rolling.max()
        transformed[range_col] = transformed[max_col] - transformed[min_col]
        transformed[slope_col] = (transformed[feature] - transformed[feature].shift(rolling_window - 1)) / (rolling_window - 1)

        for column_name in (mean_col, std_col, min_col, max_col, range_col, slope_col):
            _append_engineered_input(resolved_inputs, column_name)

    return transformed, resolved_inputs


def apply_temporal_feature_engineering(
    data,
    input_features,
    sequential_features=None,
    sequential_groups=None,
    add_differences=False,
    difference_order=1,
    add_ratio_features=False,
    add_acceleration_features=False,
    add_normalized_change_features=False,
    create_rnn_windows=False,
    rnn_window_size=3,
    add_rolling_dynamics=False,
    rolling_window=3,
):
    """Create sequence-aware industrial learning columns for selected features."""
    transformed = data.copy()
    resolved_inputs = list(input_features)

    if not sequential_features and not sequential_groups:
        return transformed, resolved_inputs

    sequential_features, normalized_groups = _normalize_sequence_features(
        transformed,
        resolved_inputs,
        sequential_features=sequential_features,
        sequential_groups=sequential_groups,
    )

    transformed, resolved_inputs = create_time_sequence_process_features(
        data=transformed,
        resolved_inputs=resolved_inputs,
        sequential_features=sequential_features,
        sequential_groups=normalized_groups,
        add_differences=add_differences,
        difference_order=difference_order,
        add_ratio_features=add_ratio_features,
        add_acceleration_features=add_acceleration_features,
        add_normalized_change_features=add_normalized_change_features,
    )

    if create_rnn_windows:
        if rnn_window_size < 1:
            raise ValueError("rnn_window_size must be >= 1.")
        lag_steps = range(1, rnn_window_size + 1)
        transformed, resolved_inputs = create_lag_window_features(
            transformed,
            resolved_inputs,
            sequential_features,
            lag_steps=lag_steps,
        )

    if add_rolling_dynamics:
        transformed, resolved_inputs = create_rolling_process_dynamics_features(
            transformed,
            resolved_inputs,
            sequential_features,
            rolling_window=rolling_window,
        )

    transformed = transformed.dropna().reset_index(drop=True)
    return transformed, resolved_inputs

def read_data(default_train_file="data.csv"):
    """Interactive reader that stores selected split in prep_data/."""
    prep_data_dir = "prep_data"
    os.makedirs(prep_data_dir, exist_ok=True)

    train_file = input(
        f"Enter the path to the training file [{default_train_file}]: "
    ).strip()
    if not train_file:
        train_file = default_train_file
    if not os.path.isfile(train_file):
        raise FileNotFoundError("The specified training file does not exist.")

    train_data = pd.read_csv(train_file)

    test_file = input("Enter the path to the testing file (leave empty for auto split): ").strip()
    if test_file and os.path.isfile(test_file):
        test_data = pd.read_csv(test_file)
        auto_split = False
    else:
        test_data = None
        auto_split = True

    output_features, input_features = select_features(train_data)

    if auto_split:
        data_seed = 123
        seed = input(f"Enter data seed for splitting ({data_seed}):").strip()
        if seed:
            data_seed = int(seed)
        y = train_data[output_features]
        X = train_data[input_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=data_seed
        )
    else:
        y_train = train_data[output_features]
        X_train = train_data[input_features]
        y_test = test_data[output_features]
        X_test = test_data[input_features]

    save_data(prep_data_dir, X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def select_features(data):
    """Interactive target/input selection from dataset columns."""
    print("\nColumns in the dataset:")
    for idx, col in enumerate(data.columns, start=1):
        print(f"{idx}. {col}")

    output_answer = input(
        "\nEnter output (target) feature(s), comma-separated [last column]: "
    ).strip()
    if not output_answer:
        output_features = [data.columns[-1]]
    else:
        output_features = [col.strip() for col in output_answer.split(",") if col.strip()]

    default_inputs = [col for col in data.columns if col not in output_features]
    print("\nDefault input features (excluding selected output(s)):")
    print(", ".join(default_inputs))

    input_features = input(
        "Press Enter to use the default input features, or enter comma-separated feature names: "
    ).strip()

    if not input_features:
        input_features = default_inputs
    else:
        input_features = [col.strip() for col in input_features.split(",")]

    return output_features, input_features


def save_data(folder, X_train, X_test, y_train, y_test, metadata=None, after_prepare=False):
    """Save train/test splits to prep_data files."""
    folder = resolve_relative_path(folder)
    os.makedirs(folder, exist_ok=True)
    print(f"\nSaving processed data to '{folder}'...")
    X_train.to_csv(os.path.join(folder, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(folder, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(folder, "y_test.csv"), index=False)
    
    if after_prepare is True:
        pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(folder, "train.csv"), index=False
        )
        pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
            os.path.join(folder, "test.csv"), index=False
        )

    pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(
        os.path.join(folder, "Xy_train.csv"), index=False
    )
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        os.path.join(folder, "Xy_test.csv"), index=False
    )
    if metadata is not None:
        save_prep_metadata(folder, metadata)
    print("Data saved successfully!")


def prepare_data_from_file(
    dataset_path="data.csv",
    output_feature="rate",
    input_features=None,
    prep_folder="prep_data",
    test_size=0.2,
    random_state=123,
    sequential_features=None,
    sequential_groups=None,
    add_differences=False,
    difference_order=1,
    add_ratio_features=False,
    add_acceleration_features=False,
    add_normalized_change_features=False,
    create_rnn_windows=False,
    rnn_window_size=3,
    add_rolling_dynamics=False,
    rolling_window=3,
    optional_future_quality_inputs=None,
):
    """Prepare prep_data files directly from a raw CSV dataset."""
    dataset_path = print_path_debug("prepare_data_from_file dataset", dataset_path)
    dataset_path = ensure_dataset_csv_exists(dataset_path)

    print(f"[path-debug] Reading raw dataset CSV: {dataset_path}")
    data = pd.read_csv(dataset_path)
    base_input_features, requested_derived_features = split_base_and_derived_input_features(
        data.columns, input_features
    )
    output_features, input_features = resolve_data_columns(
        data,
        output_feature,
        base_input_features,
        optional_future_quality_inputs=optional_future_quality_inputs,
    )
    data, input_features = apply_temporal_feature_engineering(
        data=data,
        input_features=input_features,
        sequential_features=sequential_features,
        sequential_groups=sequential_groups,
        add_differences=add_differences,
        difference_order=difference_order,
        add_ratio_features=add_ratio_features,
        add_acceleration_features=add_acceleration_features,
        add_normalized_change_features=add_normalized_change_features,
        create_rnn_windows=create_rnn_windows,
        rnn_window_size=rnn_window_size,
        add_rolling_dynamics=add_rolling_dynamics,
        rolling_window=rolling_window,
    )
    if requested_derived_features:
        missing_requested = [col for col in requested_derived_features if col not in data.columns]
        if missing_requested:
            raise ValueError(
                f"Requested temporal-derived columns were not generated: {missing_requested}. "
                f"Generated columns: {list(data.columns)}"
            )
        input_features = list(base_input_features) + list(requested_derived_features)
    validate_model_inputs(input_features, output_features=output_features, optional_future_quality_inputs=optional_future_quality_inputs)

    X = data[input_features]
    y = data[output_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    metadata = {
        "dataset_path": dataset_path,
        "output_features": list(output_features),
        "input_features": list(input_features),
        "optional_future_quality_inputs": list(optional_future_quality_inputs or []),
        "sequential_features": list(sequential_features or []),
        "sequential_groups": [list(group) for group in (sequential_groups or [])],
        "add_differences": bool(add_differences),
        "difference_order": int(difference_order),
        "add_ratio_features": bool(add_ratio_features),
        "add_acceleration_features": bool(add_acceleration_features),
        "add_normalized_change_features": bool(add_normalized_change_features),
        "create_rnn_windows": bool(create_rnn_windows),
        "rnn_window_size": int(rnn_window_size),
        "add_rolling_dynamics": bool(add_rolling_dynamics),
        "rolling_window": int(rolling_window),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "refinery_variable_groups": refinery_variable_group_metadata(),
    }
    save_data(prep_folder, X_train, X_test, y_train, y_test, metadata=metadata, after_prepare = True)
    return X_train, X_test, y_train, y_test


def sync_prep_data_with_dataset(
    dataset_path="data.csv",
    prep_folder="prep_data",
    output_feature="rate",
    input_features=None,
    sequential_features=None,
    sequential_groups=None,
    add_differences=False,
    difference_order=1,
    add_ratio_features=False,
    add_acceleration_features=False,
    add_normalized_change_features=False,
    create_rnn_windows=False,
    rnn_window_size=3,
    add_rolling_dynamics=False,
    rolling_window=3,
    optional_future_quality_inputs=None,
):
    """Rebuild prep_data if files are missing or schema differs from dataset."""
    dataset_path = print_path_debug("sync_prep_data_with_dataset dataset", dataset_path)
    dataset_path = ensure_dataset_csv_exists(dataset_path)

    print(f"[path-debug] Reading raw dataset CSV for prep_data sync: {dataset_path}")
    data = pd.read_csv(dataset_path)
    base_input_features, requested_derived_features = split_base_and_derived_input_features(
        data.columns, input_features
    )
    output_features, input_features = resolve_data_columns(
        data,
        output_feature,
        base_input_features,
        optional_future_quality_inputs=optional_future_quality_inputs,
    )
    _, input_features = apply_temporal_feature_engineering(
        data=data,
        input_features=input_features,
        sequential_features=sequential_features,
        sequential_groups=sequential_groups,
        add_differences=add_differences,
        difference_order=difference_order,
        add_ratio_features=add_ratio_features,
        add_acceleration_features=add_acceleration_features,
        add_normalized_change_features=add_normalized_change_features,
        create_rnn_windows=create_rnn_windows,
        rnn_window_size=rnn_window_size,
        add_rolling_dynamics=add_rolling_dynamics,
        rolling_window=rolling_window,
    )
    if requested_derived_features:
        input_features = list(base_input_features) + list(requested_derived_features)
    validate_model_inputs(input_features, output_features=output_features, optional_future_quality_inputs=optional_future_quality_inputs)

    paths = prep_data_file_paths(prep_folder)
    expected_files = [paths["X_train"], paths["X_test"], paths["y_train"], paths["y_test"]]

    should_rebuild = not all(os.path.isfile(path) for path in expected_files)

    if not should_rebuild:
        current_x_train = pd.read_csv(expected_files[0], nrows=0)
        current_x_test = pd.read_csv(expected_files[1], nrows=0)
        current_y_train = pd.read_csv(expected_files[2], nrows=0)
        current_y_test = pd.read_csv(expected_files[3], nrows=0)

        x_train_columns = list(current_x_train.columns)
        x_test_columns = list(current_x_test.columns)
        y_train_columns = list(current_y_train.columns)
        y_test_columns = list(current_y_test.columns)

        should_rebuild = (
            x_train_columns != input_features
            or x_test_columns != input_features
            or y_train_columns != output_features
            or y_test_columns != output_features
        )

        if should_rebuild:
            print(
                "prep_data schema mismatch detected. "
                "Expected train/test feature and target columns to match current settings."
            )

    if should_rebuild:
        print("prep_data is missing or outdated. Rebuilding from dataset...")
        prepare_data_from_file(
            dataset_path=dataset_path,
            output_feature=output_features,
            input_features=input_features,
            prep_folder=prep_folder,
            sequential_features=sequential_features,
            sequential_groups=sequential_groups,
            add_differences=add_differences,
            difference_order=difference_order,
            add_ratio_features=add_ratio_features,
            add_acceleration_features=add_acceleration_features,
            add_normalized_change_features=add_normalized_change_features,
            create_rnn_windows=create_rnn_windows,
            rnn_window_size=rnn_window_size,
            add_rolling_dynamics=add_rolling_dynamics,
            rolling_window=rolling_window,
            optional_future_quality_inputs=optional_future_quality_inputs,
        )
    else:
        print("prep_data matches dataset columns. Reusing existing files.")


def read_prep_data(inputs=None, prep_folder="prep_data", optional_future_quality_inputs=None):
    """Read preprocessed train/test files, optionally filtering input columns."""
    prep_folder = print_path_debug("read_prep_data prep_folder", prep_folder)
    print(f"Reading data from '{prep_folder}'...")

    X_train_path = os.path.join(prep_folder, "X_train.csv")
    X_test_path = os.path.join(prep_folder, "X_test.csv")
    y_train_path = os.path.join(prep_folder, "y_train.csv")
    y_test_path = os.path.join(prep_folder, "y_test.csv")

    for file_path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Required file '{file_path}' not found in '{prep_folder}'.")

    print(f"[path-debug] Reading prep_data CSV: {X_train_path}")
    X_train_full = pd.read_csv(X_train_path)
    print(f"[path-debug] Reading prep_data CSV: {X_test_path}")
    X_test_full = pd.read_csv(X_test_path)
    print(f"[path-debug] Reading prep_data CSV: {y_train_path}")
    y_train = pd.read_csv(y_train_path)
    print(f"[path-debug] Reading prep_data CSV: {y_test_path}")
    y_test = pd.read_csv(y_test_path)

    if list(X_train_full.columns) != list(X_test_full.columns):
        train_only = [col for col in X_train_full.columns if col not in X_test_full.columns]
        test_only = [col for col in X_test_full.columns if col not in X_train_full.columns]
        raise ValueError(
            "prep_data schema mismatch between X_train and X_test. "
            f"Columns only in X_train: {train_only}. "
            f"Columns only in X_test: {test_only}. "
            "Rebuild prep_data to keep train/test schemas aligned."
        )

    if inputs is not None:
        print(f"Filtering inputs: {inputs}")
        if optional_future_quality_inputs is None:
            optional_future_quality_inputs = list(inputs)
        validate_model_inputs(
            inputs,
            output_features=y_train.columns.tolist(),
            optional_future_quality_inputs=optional_future_quality_inputs,
        )
        missing_inputs = [col for col in inputs if col not in X_train_full.columns]
        if missing_inputs:
            raise KeyError(
                f"Requested input columns are missing from prep_data: {missing_inputs}. "
                "Rebuild prep_data with the same feature-engineering settings used by the run."
            )
        X_train = X_train_full[inputs]
        X_test = X_test_full[inputs]
    else:
        print("No specific inputs provided; returning all refinery-safe columns.")
        safe_inputs = filter_allowed_model_inputs(
            X_train_full.columns,
            output_features=y_train.columns.tolist(),
            optional_future_quality_inputs=optional_future_quality_inputs,
        )
        validate_model_inputs(
            safe_inputs,
            output_features=y_train.columns.tolist(),
            optional_future_quality_inputs=optional_future_quality_inputs,
        )
        if not safe_inputs:
            raise ValueError(
                "prep_data does not contain any refinery-safe model inputs. "
                "Rebuild prep_data with EARLY_VARIABLES + CONTROL_VARIABLES."
            )
        X_train = X_train_full[safe_inputs]
        X_test = X_test_full[safe_inputs]

    def _nan_report(df, name):
        nan_counts = df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        if nan_counts.empty:
            print(f"{name}: no NaN values detected.")
            return

        total_nans = int(nan_counts.sum())
        print(
            f"{ANSI_RED}WARNING: {name} contains NaN values "
            f"(total: {total_nans}).{ANSI_RESET}"
        )
        for col, cnt in nan_counts.items():
            print(f"{ANSI_RED}  - {col}: {int(cnt)} NaN{ANSI_RESET}")

    X_train = _coerce_numeric_table(X_train, "X_train")
    X_test = _coerce_numeric_table(X_test, "X_test")
    y_train = _coerce_numeric_table(y_train, "y_train")
    y_test = _coerce_numeric_table(y_test, "y_test")

    _nan_report(X_train, "X_train")
    _nan_report(X_test, "X_test")
    _nan_report(y_train, "y_train")
    _nan_report(y_test, "y_test")

    # Fill missing values in inputs so sklearn regressors that do not support NaN can train.
    input_fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(input_fill_values)
    X_test = X_test.fillna(input_fill_values)
    remaining_train_nan = int(X_train.isna().sum().sum())
    remaining_test_nan = int(X_test.isna().sum().sum())
    if remaining_train_nan or remaining_test_nan:
        print(
            f"{ANSI_YELLOW}Some input NaNs remained after median-fill. "
            "Applying forward/backward fill as fallback for compatibility."
            f"{ANSI_RESET}"
        )
        X_train = X_train.ffill().bfill()
        X_test = X_test.ffill().bfill()

    print("Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read_data()

    print("\nData processed successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
