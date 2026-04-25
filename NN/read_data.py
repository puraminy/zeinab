import os
import pandas as pd
from sklearn.model_selection import train_test_split

ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"


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


def resolve_data_columns(data, output_feature=None, input_features=None):
    """Resolve and validate target/input columns."""
    output_features = _normalize_output_features(output_feature, data.columns)

    if input_features is None:
        input_features = [col for col in data.columns if col not in output_features]

    missing_inputs = [col for col in input_features if col not in data.columns]
    if missing_inputs:
        raise ValueError(
            f"Input columns not found in dataset: {missing_inputs}. "
            f"Available columns: {list(data.columns)}"
        )

    return output_features, input_features




def apply_temporal_feature_engineering(
    data,
    input_features,
    sequential_features=None,
    add_differences=False,
    difference_order=1,
    create_rnn_windows=False,
    rnn_window_size=3,
):
    """Create temporal columns (differences and lag windows) for selected features."""
    transformed = data.copy()
    resolved_inputs = list(input_features)

    if not sequential_features:
        return transformed, resolved_inputs

    sequential_features = [
        col for col in sequential_features if col in transformed.columns and col in resolved_inputs
    ]

    if add_differences:
        if difference_order < 1:
            raise ValueError("difference_order must be >= 1.")
        for feature in sequential_features:
            source_col = feature
            for order in range(1, difference_order + 1):
                diff_col = f"{feature}__diff_{order}"
                transformed[diff_col] = transformed[source_col].diff()
                source_col = diff_col
                if diff_col not in resolved_inputs:
                    resolved_inputs.append(diff_col)

    if create_rnn_windows:
        if rnn_window_size < 2:
            raise ValueError("rnn_window_size must be >= 2.")
        for feature in sequential_features:
            for lag_step in range(1, rnn_window_size):
                lag_col = f"{feature}__lag_{lag_step}"
                transformed[lag_col] = transformed[feature].shift(lag_step)
                if lag_col not in resolved_inputs:
                    resolved_inputs.append(lag_col)

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


def save_data(folder, X_train, X_test, y_train, y_test):
    """Save train/test splits to prep_data files."""
    os.makedirs(folder, exist_ok=True)
    print(f"\nSaving processed data to '{folder}'...")
    X_train.to_csv(os.path.join(folder, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(folder, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(folder, "y_test.csv"), index=False)
    print("Data saved successfully!")


def prepare_data_from_file(
    dataset_path="data.csv",
    output_feature="rate",
    input_features=None,
    prep_folder="prep_data",
    test_size=0.2,
    random_state=123,
    sequential_features=None,
    add_differences=False,
    difference_order=1,
    create_rnn_windows=False,
    rnn_window_size=3,
):
    """Prepare prep_data files directly from a raw CSV dataset."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    data = pd.read_csv(dataset_path)
    output_features, input_features = resolve_data_columns(data, output_feature, input_features)
    data, input_features = apply_temporal_feature_engineering(
        data=data,
        input_features=input_features,
        sequential_features=sequential_features,
        add_differences=add_differences,
        difference_order=difference_order,
        create_rnn_windows=create_rnn_windows,
        rnn_window_size=rnn_window_size,
    )

    X = data[input_features]
    y = data[output_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    save_data(prep_folder, X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def sync_prep_data_with_dataset(
    dataset_path="data.csv",
    prep_folder="prep_data",
    output_feature="rate",
    input_features=None,
    sequential_features=None,
    add_differences=False,
    difference_order=1,
    create_rnn_windows=False,
    rnn_window_size=3,
):
    """Rebuild prep_data if files are missing or schema differs from dataset."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    data = pd.read_csv(dataset_path)
    output_features, input_features = resolve_data_columns(data, output_feature, input_features)
    _, input_features = apply_temporal_feature_engineering(
        data=data,
        input_features=input_features,
        sequential_features=sequential_features,
        add_differences=add_differences,
        difference_order=difference_order,
        create_rnn_windows=create_rnn_windows,
        rnn_window_size=rnn_window_size,
    )

    expected_files = [
        os.path.join(prep_folder, "X_train.csv"),
        os.path.join(prep_folder, "X_test.csv"),
        os.path.join(prep_folder, "y_train.csv"),
        os.path.join(prep_folder, "y_test.csv"),
    ]

    should_rebuild = not all(os.path.isfile(path) for path in expected_files)

    if not should_rebuild:
        current_x = pd.read_csv(expected_files[0])
        current_y = pd.read_csv(expected_files[2])
        should_rebuild = (
            list(current_x.columns) != input_features
            or list(current_y.columns) != output_features
        )

    if should_rebuild:
        print("prep_data is missing or outdated. Rebuilding from dataset...")
        prepare_data_from_file(
            dataset_path=dataset_path,
            output_feature=output_features,
            input_features=input_features,
            prep_folder=prep_folder,
            sequential_features=sequential_features,
            add_differences=add_differences,
            difference_order=difference_order,
            create_rnn_windows=create_rnn_windows,
            rnn_window_size=rnn_window_size,
        )
    else:
        print("prep_data matches dataset columns. Reusing existing files.")


def read_prep_data(inputs=None, prep_folder="prep_data"):
    """Read preprocessed train/test files, optionally filtering input columns."""
    print(f"Reading data from '{prep_folder}'...")

    X_train_path = os.path.join(prep_folder, "X_train.csv")
    X_test_path = os.path.join(prep_folder, "X_test.csv")
    y_train_path = os.path.join(prep_folder, "y_train.csv")
    y_test_path = os.path.join(prep_folder, "y_test.csv")

    for file_path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Required file '{file_path}' not found in '{prep_folder}'.")

    X_train_full = pd.read_csv(X_train_path)
    X_test_full = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    if inputs is not None:
        print(f"Filtering inputs: {inputs}")
        X_train = X_train_full[inputs]
        X_test = X_test_full[inputs]
    else:
        print("No specific inputs provided; returning all columns.")
        X_train = X_train_full
        X_test = X_test_full

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
