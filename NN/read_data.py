import os
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data():
    """
    Reads training and testing data from user-specified files or performs an automatic split.
    Saves the processed data into a folder for future use.
    
    Returns:
        X_train, X_test, y_train, y_test: Training and test datasets.
    """
    # Create prep_data folder if not exists
    prep_data_dir = "prep_data"
    os.makedirs(prep_data_dir, exist_ok=True)
    
    # Ask for train file
    train_file = input("Enter the path to the training file: ").strip()
    if not train_file or not os.path.isfile(train_file):
        raise FileNotFoundError("The specified training file does not exist.")
    
    # Load train data
    train_data = pd.read_csv(train_file)
    
    # Ask for test file
    test_file = input("Enter the path to the testing file (leave empty for auto split): ").strip()
    if test_file and os.path.isfile(test_file):
        test_data = pd.read_csv(test_file)
        auto_split = False
    else:
        test_data = None
        auto_split = True
    
    # Handle feature selection
    output_feature, input_features = select_features(train_data)
    
    if auto_split:
        # Automatic split
        data_seed = 123
        seed = input(f"Enter data seed for splitting ({data_seed}):").strip()
        if seed: data_seed = int(seed)
        y = train_data[output_feature]
        X = train_data[input_features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=data_seed)
    else:
        # Separate files for train and test
        y_train = train_data[output_feature]
        X_train = train_data[input_features]
        y_test = test_data[output_feature]
        X_test = test_data[input_features]
    
    # Save processed parts to the folder
    save_data(prep_data_dir, X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test


def select_features(data):
    """
    Guides the user to select output and input features from the dataset interactively.
    
    Args:
        data (DataFrame): The dataset to select features from.
    
    Returns:
        output_feature (str): The selected output feature.
        input_features (list): The selected input features.
    """
    # List columns and ask for output feature
    print("\nColumns in the dataset:")
    for idx, col in enumerate(data.columns, start=1):
        print(f"{idx}. {col}")
    
    output_feature = input("\nEnter the name of the output (target) feature: ").strip()
    
    # Default input features (excluding the output feature)
    default_inputs = [col for col in data.columns if col != output_feature]
    print("\nDefault input features (excluding the output):")
    print(", ".join(default_inputs))
    
    input_features = input(
        "Press Enter to use the default input features, or enter comma-separated feature names: ").strip()
    
    # If user presses Enter, use default inputs; otherwise, use the provided ones
    if not input_features:
        input_features = default_inputs
    else:
        input_features = [col.strip() for col in input_features.split(",")]
    
    return output_feature, input_features


def save_data(folder, X_train, X_test, y_train, y_test):
    """
    Saves the training and testing datasets into CSV files in the specified folder.
    
    Args:
        folder (str): Directory to save the data.
        X_train, X_test, y_train, y_test: Data to save.
    """
    print(f"\nSaving processed data to '{folder}'...")
    X_train.to_csv(os.path.join(folder, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(folder, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(folder, "y_test.csv"), index=False)
    print("Data saved successfully!")

import os
import pandas as pd

def read_prep_data(inputs, prep_folder="prep_data"):
    """
    Reads the preprocessed training and testing datasets from the specified folder
    and extracts only the specified input columns.

    Args:
        inputs (list): List of column names to be used as input features.
        prep_folder (str): Directory where the preprocessed data is stored.

    Returns:
        X_train (DataFrame): Features (specified inputs) for the training dataset.
        X_test (DataFrame): Features (specified inputs) for the testing dataset.
        y_train (Series): Labels for the training dataset.
        y_test (Series): Labels for the testing dataset.
    """
    print(f"Reading data from '{prep_folder}'...")

    # Define file paths
    X_train_path = os.path.join(prep_folder, "X_train.csv")
    X_test_path = os.path.join(prep_folder, "X_test.csv")
    y_train_path = os.path.join(prep_folder, "y_train.csv")
    y_test_path = os.path.join(prep_folder, "y_test.csv")

    # Check if all required files exist
    for file_path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Required file '{file_path}' not found in '{prep_folder}'.")

    # Load the data
    X_train_full = pd.read_csv(X_train_path)
    X_test_full = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]  # Ensure y is read as a Series
    y_test = pd.read_csv(y_test_path).iloc[:, 0]   # Ensure y is read as a Series

    # Extract specified input columns
    X_train = X_train_full[inputs]
    X_test = X_test_full[inputs]

    print("Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

import os
import pandas as pd

def read_prep_data(inputs=None, prep_folder="prep_data"):
    """
    Reads the preprocessed training and testing datasets from the specified folder.
    If `inputs` is None, all columns are returned.

    Args:
        inputs (list or None): List of column names to be used as input features.
                               If None, all columns are returned.
        prep_folder (str): Directory where the preprocessed data is stored.

    Returns:
        X_train (DataFrame): Features (specified inputs or all) for the training dataset.
        X_test (DataFrame): Features (specified inputs or all) for the testing dataset.
        y_train (Series): Labels for the training dataset.
        y_test (Series): Labels for the testing dataset.
    """
    print(f"Reading data from '{prep_folder}'...")

    # Define file paths
    X_train_path = os.path.join(prep_folder, "X_train.csv")
    X_test_path = os.path.join(prep_folder, "X_test.csv")
    y_train_path = os.path.join(prep_folder, "y_train.csv")
    y_test_path = os.path.join(prep_folder, "y_test.csv")

    # Check if all required files exist
    for file_path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Required file '{file_path}' not found in '{prep_folder}'.")

    # Load the data
    X_train_full = pd.read_csv(X_train_path)
    X_test_full = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]  # Ensure y is read as a Series
    y_test = pd.read_csv(y_test_path).iloc[:, 0]   # Ensure y is read as a Series

    # Extract specified input columns or return all columns
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

# Example usage
if __name__ == "__main__":
    # Read data and save processed parts
    X_train, X_test, y_train, y_test = read_data()
    
    print("\nData processed successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    try:
        inputs = None  # Use None to return all columns
        X_train, X_test, y_train, y_test = read_prep_data()
        print("X_train:")
        print(X_train.head())
        print("y_train:")
        print(y_train.head())
        print("X_test:")
        print(X_test.head())
    except FileNotFoundError as e:
        print(f"Error: {e}")

