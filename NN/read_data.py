import os
import pandas as pd
from sklearn.model_selection import train_test_split


def _resolve_columns(data, output_feature=None, input_features=None):
    """Resolve and validate target/input columns."""
    if output_feature is None:
        output_feature = "rate" if "rate" in data.columns else data.columns[-1]

    if output_feature not in data.columns:
        raise ValueError(f"Target column '{output_feature}' not found in dataset.")

    if input_features is None:
        input_features = [col for col in data.columns if col != output_feature]

    missing_inputs = [col for col in input_features if col not in data.columns]
    if missing_inputs:
        raise ValueError(
            f"Input columns not found in dataset: {missing_inputs}. "
            f"Available columns: {list(data.columns)}"
        )

    return output_feature, input_features


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

    output_feature, input_features = select_features(train_data)

    if auto_split:
        data_seed = 123
        seed = input(f"Enter data seed for splitting ({data_seed}):").strip()
        if seed:
            data_seed = int(seed)
        y = train_data[output_feature]
        X = train_data[input_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=data_seed
        )
    else:
        y_train = train_data[output_feature]
        X_train = train_data[input_features]
        y_test = test_data[output_feature]
        X_test = test_data[input_features]

    save_data(prep_data_dir, X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def select_features(data):
    """Interactive target/input selection from dataset columns."""
    print("\nColumns in the dataset:")
    for idx, col in enumerate(data.columns, start=1):
        print(f"{idx}. {col}")

    output_feature = input("\nEnter the name of the output (target) feature: ").strip()

    default_inputs = [col for col in data.columns if col != output_feature]
    print("\nDefault input features (excluding the output):")
    print(", ".join(default_inputs))

    input_features = input(
        "Press Enter to use the default input features, or enter comma-separated feature names: "
    ).strip()

    if not input_features:
        input_features = default_inputs
    else:
        input_features = [col.strip() for col in input_features.split(",")]

    return output_feature, input_features


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
):
    """Prepare prep_data files directly from a raw CSV dataset."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    data = pd.read_csv(dataset_path)
    output_feature, input_features = _resolve_columns(data, output_feature, input_features)

    X = data[input_features]
    y = data[output_feature]

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
):
    """Rebuild prep_data if files are missing or schema differs from dataset."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    data = pd.read_csv(dataset_path)
    output_feature, input_features = _resolve_columns(data, output_feature, input_features)

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
            or current_y.columns[0] != output_feature
        )

    if should_rebuild:
        print("prep_data is missing or outdated. Rebuilding from dataset...")
        prepare_data_from_file(
            dataset_path=dataset_path,
            output_feature=output_feature,
            input_features=input_features,
            prep_folder=prep_folder,
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
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    if inputs is not None:
        print(f"Filtering inputs: {inputs}")
        X_train = X_train_full[inputs]
        X_test = X_test_full[inputs]
    else:
        print("No specific inputs provided; returning all columns.")
        X_train = X_train_full
        X_test = X_test_full

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
