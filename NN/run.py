from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score
from tabulate import tabulate
import os
from latex import *
from plot import *
from models import *
from read_data import read_prep_data, sync_prep_data_with_dataset, resolve_data_columns
import inspect
import models

list_epochs = [20, 50, 100 , 200]
best_epochs = 100 

exp_df = pd.DataFrame()

num_repeats = 5
# num_repeats shows the number of times to repeat the experiment to get its average values
# changing model_seed generates different results for each run for each model
# The same model_seed produces the same results in each run which is good for reporducability of experiments
# Dont change it if you want to reproduce the same results
data_seed = 123 # it is used for random_state of splitting data into source and train sets
# changing it creates different source and train sets.
# Since the number of data is low changing it can largely affect the results
model_seed = 123 # it is used for random_state of models
def set_model_seed(model_seed):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

learning_rate = 0.05
# The learnign rate used in ANN
#hidden_size1 = 10
hidden_size1 = 15
hidden_size2 = 10

# the number of neurons in hidden layers

# https://alexlenail.me/NN-SVG/
# use the site above to draw the following network
#
hidden_sizes = [15, 10, 3 ]
# nn.ReLU(), nn.Tanh(), nn.Identity()

list_hidden_sizes = [[10], [15, 10, 3], [8, 4], [15, 5]]
normalization_type = "z_score"


SKLEARN_MODEL_FACTORIES = {
    "RandomForestRegressor": lambda seed: RandomForestRegressor(
        n_estimators=300, random_state=seed
    ),
    "ExtraTreesRegressor": lambda seed: ExtraTreesRegressor(
        n_estimators=300, random_state=seed
    ),
    "GradientBoostingRegressor": lambda seed: GradientBoostingRegressor(random_state=seed),
    "HistGradientBoostingRegressor": lambda seed: HistGradientBoostingRegressor(
        random_state=seed
    ),
    "AdaBoostRegressor": lambda seed: AdaBoostRegressor(random_state=seed),
    "DecisionTreeRegressor": lambda seed: DecisionTreeRegressor(random_state=seed),
    "KNeighborsRegressor": lambda seed: KNeighborsRegressor(n_neighbors=5),
    "SVR_RBF": lambda seed: SVR(kernel="rbf"),
    "LinearRegression": lambda seed: LinearRegression(),
    "Ridge": lambda seed: Ridge(random_state=seed),
    "Lasso": lambda seed: Lasso(random_state=seed),
    "ElasticNet": lambda seed: ElasticNet(random_state=seed),
}


def _expand_index_token(token, options_len):
    """Expand a token like '3' or '1-4' into a list of indexes."""
    if "-" in token:
        bounds = token.split("-", 1)
        if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
            raise ValueError(f"Invalid range '{token}'. Use syntax like 1-4.")
        start = int(bounds[0])
        end = int(bounds[1])
        if start > end:
            raise ValueError(f"Invalid range '{token}'. Start must be <= end.")
        indexes = list(range(start, end + 1))
    else:
        if not token.isdigit():
            raise ValueError(f"Invalid input '{token}'. Please use indexes or ranges like 1-4.")
        indexes = [int(token)]

    for index in indexes:
        if index < 0 or index >= options_len:
            raise ValueError(
                f"Invalid selection '{index}'. Valid range is 0 to {options_len - 1}."
            )
    return indexes


def parse_multi_select(answer, options, allow_all=True):
    """Parse indexes/ranges and return selected option values.

    Supported syntax examples:
    - 0 2 4
    - 1-5
    - 1,3,6-9
    - !0 !4-6 (exclude indexes/ranges from current pool)
    """
    if not answer:
        return None if allow_all else []

    answer = answer.strip().lower()
    if allow_all and answer == "all":
        return None

    include_indexes = set()
    exclude_indexes = set()
    tokens = [token for token in answer.replace(",", " ").split() if token]
    if not tokens:
        return None if allow_all else []

    for token in tokens:
        is_exclude = token.startswith("!")
        clean = token[1:] if is_exclude else token
        if clean == "all":
            if not allow_all:
                raise ValueError("'all' is not allowed for this selection.")
            include_indexes.update(range(len(options)))
            continue

        indexes = _expand_index_token(clean, len(options))
        if is_exclude:
            exclude_indexes.update(indexes)
        else:
            include_indexes.update(indexes)

    if include_indexes:
        final_indexes = sorted(include_indexes.difference(exclude_indexes))
    else:
        # Exclude-only syntax means "all except excluded".
        if exclude_indexes:
            final_indexes = [i for i in range(len(options)) if i not in exclude_indexes]
        elif allow_all:
            return None
        else:
            final_indexes = []

    if not final_indexes:
        raise ValueError("Selection is empty after applying include/exclude filters.")

    selected_values = [options[index] for index in final_indexes]
    return selected_values


def print_selection_guide():
    print("\nSelection guide:")
    print("- Single indexes: 0 3 5")
    print("- Range syntax: 1-9")
    print("- Mixed syntax: 0 2-4 7")
    print("- Commas are also allowed: 0,2-4,7")
    print("- Exclude indexes/ranges with ! : !0 !5-8")
    print("- In auto mode, exclude-only means all features except those indexes.")


def ask_with_default(prompt, default):
    answer = input(f"{prompt} [{default}]:").strip()
    return answer if answer else str(default)


def parse_epochs_input(answer, default_epochs):
    """Parse epochs input and detect optional CV early-stop mode."""
    text = answer.strip().lower()
    if text == "0":
        return [], False

    use_cv_early_stop = False
    parsed_epochs = []
    for token in answer.split():
        lowered = token.lower()
        if lowered in ("cv", "auto"):
            use_cv_early_stop = True
            continue
        if token.isdigit() and int(token) > 0:
            parsed_epochs.append(int(token))

    if not parsed_epochs and not use_cv_early_stop:
        parsed_epochs = list(default_epochs)
    return sorted(set(parsed_epochs)), use_cv_early_stop


def infer_selected_features_from_table(table, fallback_features):
    """Infer selected features from the best-R2 row in a feature-selection table."""
    if table is None or table.empty:
        return list(fallback_features)

    if "R2" not in table.columns or "features" not in table.columns:
        return list(fallback_features)

    ranked = table.dropna(subset=["R2"])
    if ranked.empty:
        return list(fallback_features)

    best_row = ranked.loc[ranked["R2"].astype(float).idxmax()]
    features_text = str(best_row["features"]).strip()
    if not features_text:
        return list(fallback_features)

    selected = [f.strip() for f in features_text.split(",") if f.strip()]
    return selected if selected else list(fallback_features)


def compute_regression_report_metrics(y_true, y_pred):
    """
    Compute report-friendly regression metrics for the selected/best model.
    Metrics follow common WEKA-style definitions for RAE and RRSE.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    y_true_flat = y_true_arr.reshape(-1)
    y_pred_flat = y_pred_arr.reshape(-1)

    if y_true_flat.size == 0 or y_pred_flat.size == 0:
        return None

    if y_true_flat.size > 1:
        correlation = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
    else:
        correlation = np.nan

    abs_errors = np.abs(y_pred_flat - y_true_flat)
    sq_errors = (y_pred_flat - y_true_flat) ** 2
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))

    true_mean = float(np.mean(y_true_flat))
    rae_denominator = float(np.sum(np.abs(y_true_flat - true_mean)))
    rrse_denominator = float(np.sum((y_true_flat - true_mean) ** 2))
    rae = float((np.sum(abs_errors) / rae_denominator) * 100.0) if rae_denominator != 0 else np.nan
    rrse = float(np.sqrt(np.sum(sq_errors) / rrse_denominator) * 100.0) if rrse_denominator != 0 else np.nan

    return {
        "Correlation coefficient": correlation,
        "Mean absolute error": mae,
        "Root mean squared error": rmse,
        "Relative absolute error": rae,
        "Root relative squared error": rrse,
        "Total Number of Instances": int(y_true_arr.shape[0]),
    }

# Define the normalization function
def normalize(data, normalization_type):
    if normalization_type == 'z_score':
        return (data - data.mean()) / data.std()
    elif normalization_type == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    else:
        raise ValueError("Unsupported normalization type. Choose 'z_score' or 'minmax'.")

# Function to apply model on data and generate predictions
# Return predictions, MSE and R-Squared
import torch.nn.init as init

def fit_model(model, X_train, X_test, y_train, y_test, 
        num_epochs, display_steps=False, run=0):

    set_model_seed(model_seed + run)
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train_values = y_train.values if hasattr(y_train, "values") else y_train
    y_test_values = y_test.values if hasattr(y_test, "values") else y_test
    y_train = torch.tensor(y_train_values, dtype=torch.float32)
    y_test = torch.tensor(y_test_values, dtype=torch.float32)
    if y_train.ndim == 1:
        y_train = y_train.view(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.view(-1, 1)

   # Normalize inputs and targets to zero mean and unity standard deviation
    X_train_normalized = normalize(X_train, normalization_type)
    X_test_normalized = normalize(X_test, normalization_type)
    y_train_normalized = normalize(y_train, normalization_type)
    y_test_normalized = normalize(y_test, normalization_type)


    # Initialize weights
    def weights_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0)

    model.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_normalized)
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            print(f"NaN detected in outputs at epoch {epoch + 1}")
            return None, None, None, model

        loss = criterion(outputs, y_train_normalized)
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            print(f"NaN detected in loss at epoch {epoch + 1}")
            return None, None, None, model

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if (epoch + 1) % 10 == 0 and display_steps:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_normalized)
        
        # Check for NaN in predictions
        if torch.isnan(predictions).any():
            print(f"NaN detected in predictions")
            return None, None, None, model

    mse = nn.MSELoss()(predictions, y_test_normalized)
    mae = nn.L1Loss()(predictions, y_test_normalized)
    predictions_denormalized = predictions * y_test.std() + y_test.mean()

    predictions_np = predictions_denormalized.numpy()
    y_test_np = y_test.numpy()
    r2 = r2_score(y_test_np, predictions_np, multioutput='uniform_average')
    return predictions_np, mse, r2, model


def fit_sklearn_model(model, X_train, X_test, y_train, y_test):
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_values = y_train.values if hasattr(y_train, "values") else y_train
    y_test_values = y_test.values if hasattr(y_test, "values") else y_test
    if y_train_values.ndim == 2 and y_train_values.shape[1] == 1:
        y_train_fit = y_train_values.ravel()
    else:
        y_train_fit = y_train_values

    model.fit(X_train_scaled, y_train_fit)
    predictions = model.predict(X_test_scaled)

    if predictions.ndim == 1:
        predictions_2d = predictions.reshape(-1, 1)
    else:
        predictions_2d = predictions

    y_test_2d = y_test_values.reshape(-1, 1) if y_test_values.ndim == 1 else y_test_values
    mse = float(np.mean((predictions_2d - y_test_2d) ** 2))
    r2 = r2_score(y_test_2d, predictions_2d, multioutput='uniform_average')
    return predictions_2d, mse, r2, model


def is_torch_model(model_class):
    return isinstance(model_class, type) and issubclass(model_class, nn.Module)


def select_epochs_with_cv_early_stopping(model_class, hidden_sizes, features=None, folds=5, patience=20, min_delta=1e-4):
    """
    Estimate a good epoch count via K-Fold CV with early stopping on validation loss.
    Returns an integer epoch recommendation or None if it cannot be computed.
    """
    X_train, X_test, y_train, y_test = read_prep_data(features)
    full_X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    full_y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    kfold = KFold(n_splits=folds, shuffle=True, random_state=data_seed)
    best_epochs = []
    input_size = full_X.shape[1]
    output_size = full_y.shape[1] if hasattr(full_y, "shape") and len(full_y.shape) > 1 else 1
    max_epochs = max(list_epochs) if list_epochs else 200

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(full_X), start=1):
        x_train_fold = full_X.iloc[train_idx]
        x_val_fold = full_X.iloc[val_idx]
        y_train_fold = full_y.iloc[train_idx]
        y_val_fold = full_y.iloc[val_idx]

        scaler = StandardScaler()
        x_train_fold = torch.tensor(scaler.fit_transform(x_train_fold), dtype=torch.float32)
        x_val_fold = torch.tensor(scaler.transform(x_val_fold), dtype=torch.float32)
        y_train_values = y_train_fold.values if hasattr(y_train_fold, "values") else y_train_fold
        y_val_values = y_val_fold.values if hasattr(y_val_fold, "values") else y_val_fold
        y_train_fold = torch.tensor(y_train_values, dtype=torch.float32)
        y_val_fold = torch.tensor(y_val_values, dtype=torch.float32)
        if y_train_fold.ndim == 1:
            y_train_fold = y_train_fold.view(-1, 1)
        if y_val_fold.ndim == 1:
            y_val_fold = y_val_fold.view(-1, 1)

        x_train_fold = normalize(x_train_fold, normalization_type)
        x_val_fold = normalize(x_val_fold, normalization_type)
        y_train_fold = normalize(y_train_fold, normalization_type)
        y_val_fold = normalize(y_val_fold, normalization_type)

        try:
            model = model_class(input_size, hidden_sizes, output_size=output_size)
        except TypeError:
            model = model_class(input_size, hidden_sizes)

        if len(hidden_sizes) != len(model.hidden_layers):
            return None

        set_model_seed(model_seed + fold_index)
        model.apply(
            lambda m: (
                init.kaiming_uniform_(m.weight.data),
                init.constant_(m.bias.data, 0) if m.bias is not None else None
            ) if isinstance(m, nn.Linear) else None
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        best_epoch = 1
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            train_outputs = model(x_train_fold)
            train_loss = criterion(train_outputs, y_train_fold)
            if torch.isnan(train_loss):
                break
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_fold)
                val_loss = criterion(val_outputs, y_val_fold).item()

            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        best_epochs.append(best_epoch)

    if not best_epochs:
        return None
    return int(np.median(best_epochs))

############################ Feature Selection ####################
# Selects the best combination of features by removing features one by one
# Search about Backward Feature Elimination 
# Sensitivity Analysis Functions



def train_single_model(model_class, num_epochs, hidden_sizes, features=None, run=0):
    """Train one model instance and return trained model with raw train/test splits."""
    X_train, X_test, y_train, y_test = read_prep_data(features)
    input_size = X_train.shape[1]
    output_size = y_train.shape[1] if hasattr(y_train, "shape") and len(y_train.shape) > 1 else 1
    try:
        model = model_class(input_size, hidden_sizes, output_size=output_size)
    except TypeError:
        model = model_class(input_size, hidden_sizes)

    _, _, r2, model = fit_model(
        model, X_train, X_test, y_train, y_test, num_epochs, display_steps=False, run=run
    )
    return model, X_train, X_test, y_train, y_test, r2


def shap_feature_importance(model_class, inputs, num_epochs, hidden_sizes):
    """Compute SHAP-based feature importance for the selected model."""
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Install it with: pip install shap")
        return None

    model, X_train, X_test, y_train, _, r2 = train_single_model(
        model_class, num_epochs, hidden_sizes, features=inputs, run=0
    )
    if r2 is None:
        print("Could not train model for SHAP analysis.")
        return None

    scaler = StandardScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    X_train_norm = normalize(X_train_scaled, normalization_type).numpy()
    X_test_norm = normalize(X_test_scaled, normalization_type).numpy()

    background_default = min(25, len(X_train_norm))
    explain_default = min(20, len(X_test_norm))
    nsamples_default = 100

    def parse_positive_int(answer, default_value, name):
        if not answer:
            return default_value
        try:
            value = int(answer)
        except ValueError:
            print(f"Invalid {name} '{answer}'. Using default value {default_value}.")
            return default_value
        if value <= 0:
            print(f"{name} must be > 0. Using default value {default_value}.")
            return default_value
        return value

    background_size = parse_positive_int(
        input(f"Background sample size [{background_default}]:").strip(),
        background_default,
        "background sample size",
    )
    explain_size = parse_positive_int(
        input(f"Number of test samples to explain [{explain_default}]:").strip(),
        explain_default,
        "number of test samples",
    )
    nsamples = parse_positive_int(
        input(f"Kernel SHAP nsamples [{nsamples_default}]:").strip(),
        nsamples_default,
        "Kernel SHAP nsamples",
    )

    background_size = max(1, min(background_size, len(X_train_norm)))
    explain_size = max(1, min(explain_size, len(X_test_norm)))

    background = X_train_norm[:background_size]
    explain_points = X_test_norm[:explain_size]

    model.eval()

    def predict_fn(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            pred = model(x_tensor).detach().numpy()
        return pred

    print("Computing SHAP values. This may take some time ...")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_points, nsamples=nsamples)

    shap_array = np.asarray(shap_values)
    if isinstance(shap_values, list):
        shap_array = np.stack([np.asarray(sv) for sv in shap_values], axis=0)

    shap_abs = np.abs(shap_array)
    # Collapse extra output dimensions (e.g., multi-output predictions) until we get
    # a 2D array with shape (n_samples, n_features) or a 1D feature vector.
    while shap_abs.ndim > 2:
        shap_abs = shap_abs.mean(axis=0)

    mean_abs_shap = shap_abs.mean(axis=0) if shap_abs.ndim == 2 else shap_abs
    mean_abs_shap = np.asarray(mean_abs_shap).reshape(-1)
    feature_names = list(X_train.columns)

    if mean_abs_shap.shape[0] != len(feature_names):
        print(
            "Could not build SHAP table because feature and SHAP dimensions do not match. "
            f"Got {mean_abs_shap.shape[0]} SHAP values for {len(feature_names)} features. "
            f"Raw SHAP shape: {np.asarray(shap_values).shape}"
        )
        return None

    shap_table = pd.DataFrame({
        "Feature": feature_names,
        "Mean(|SHAP|)": mean_abs_shap
    }).sort_values(by="Mean(|SHAP|)", ascending=False)

    try:
        os.makedirs("plots", exist_ok=True)
        shap.summary_plot(shap_values, explain_points, feature_names=list(X_train.columns), show=False)
        plt.tight_layout()
        plt.savefig(os.path.join("plots", "shap_summary.png"), dpi=200)
        plt.close()
        print("SHAP summary plot saved at plots/shap_summary.png")
    except Exception as err:
        print(f"Could not save SHAP summary plot: {err}")

    return shap_table
def weight_analysis(model_class, data, inputs, output, num_epochs, hidden_sizes):
    X_train, X_test, y_train, y_test = read_prep_data(inputs)
    input_size = X_train.shape[1]

    output_size = y_train.shape[1] if hasattr(y_train, "shape") and len(y_train.shape) > 1 else 1
    try:
        model = model_class(input_size, hidden_sizes, output_size=output_size)
    except TypeError:
        model = model_class(input_size, hidden_sizes)
    _,_,r2, model = fit_model(model, X_train, X_test, y_train, y_test, num_epochs)
    print("R2:", r2)

    weight_importances = {}
    with torch.no_grad():
        for i, feature in enumerate(inputs):
            importance = np.abs(model.hidden_layers[0].weight[:, i].numpy()).mean()
            weight_importances[feature] = importance
    
    weight_table = pd.DataFrame(list(weight_importances.items()), columns=['Feature', 'Weight Importance'])
    return weight_table


def jackknife_sensitivity_analysis(model_class, data, inputs, output, num_epochs, hidden_sizes):
    base_model_r2, _, _, _, _, _, run = repeat_fit_model(model_class, 
            num_repeats, num_epochs, hidden_sizes)
    sensitivities = {}
    variances = []
    _num_repeats = input("Number of repeat [10]:")
    _num_repeats = int(_num_repeats) if _num_repeats else 10
    print("Please wait ...")

    for input_feature in inputs:
        reduced_inputs = [f for f in inputs if f != input_feature]
        reduced_r2_list = []
        for _ in range(_num_repeats):
            reduced_r2, _, _, _, _, _, _run = repeat_fit_model(model_class,
                    1, num_epochs, hidden_sizes, features= reduced_inputs)
            reduced_r2_list.append(reduced_r2)
        reduced_r2_mean = np.mean(reduced_r2_list)
        sensitivity = base_model_r2 - reduced_r2_mean
        variance = np.var(reduced_r2_list)
        sensitivities[input_feature] = sensitivity
        variances.append(variance)

    sensitivity_table = pd.DataFrame(list(sensitivities.items()), columns=['Feature', 'Sensitivity'])
    sensitivity_table['Variance'] = variances
    return sensitivity_table

def backward_feature_elimination(model_class, data, inputs, output, num_epochs, hidden_sizes):
    mean_r2, _, _, _, _,_, _run = repeat_fit_model(model_class,
            num_repeats, num_epochs, hidden_sizes,features=inputs)
    best_r2 = mean_r2
    print("Using all features")
    print("Features:", inputs)
    print("R2:", mean_r2)
    candidates = inputs
    rows = [] # Rows of a table to show the features and the result
    rows.append({"features": ", ".join(inputs), "R2": round(mean_r2,2)})
    while(True):
        results = {}
        for feature in candidates:
            # Selet other features except for current feature
            features = [f for f in candidates if f != feature]
            mean_r2, _, _, _,_,_, _run = repeat_fit_model(model_class, num_repeats, num_epochs, hidden_sizes, features=features)
            results[feature] = mean_r2
            print("---------------------------------------")
            # Results of removing the feature
            print("Removing:", feature)
            print("Features:", features)
            print("R2:", mean_r2)
            rows.append({"features": ", ".join(features), "R2": round(mean_r2,2)})

        if not results:
            print("No valid feature-elimination candidates were produced; stopping.")
            break

        max_results =  max(results.values())
        max_results_key = max(results, key=results.get)

        if max_results > best_r2:
            candidates.remove(max_results_key)
            best_r2 = max_results
            print('=================================')
            print('step: ' + str(len(inputs) - len(candidates)))
            print(candidates)
            print('Best R2: ' + str(max_results))
            print('================================')
        else:
            break

    print('\n\n')
    print('=============== Final Features ==================')
    print('Selected Features: ')
    print(candidates)
    print('Final R2: ' + str(best_r2))
    print('Elminated Features: ')
    print(set(data.columns).difference(candidates))

    table = pd.DataFrame(data=rows)
    return table

# Finds the combinaiton of features that best predict y
# This method add features one by one
# Search about Forward Feature Selction
def forward_feature_selection(model_class, data, inputs, output, num_epochs, hidden_sizes):
    candidates = []
    best_r2 = -1
    rows = [] # Rows of a table to show the features and the result
    while(True):
        results = {}
        # for all features except for the candidates
        for feature in data.drop(candidates, axis=1).columns:
            if len(candidates) == 0:
                features = [feature]
            else:
                features = [feature] + candidates 

            mean_r2, _, _, _,_,_, _run = repeat_fit_model(model_class, num_repeats, num_epochs, hidden_sizes, features=features)
            if mean_r2 is None:
                continue
            results[feature] = mean_r2
            print("--------------------------------")
            print("Adding feature ", feature)
            print("Features:", features)
            print("R2:", mean_r2)
            rows.append({"features": ", ".join(features), "R2": round(mean_r2,2)})

        if not results:
            print("No valid feature-selection candidates were produced; stopping.")
            break

        max_results =  max(results.values())
        max_results_key = max(results, key=results.get)

        if max_results > best_r2:
            candidates.append(max_results_key)
            best_r2 = max_results
            print('=================================')
            print('step: ' + str(len(candidates)))
            print(candidates)
            print('Best R2: ' + str(max_results))
            print('================================')
        else:
            break

    print('\n\n')
    print('=============== Final Features ==================')
    print('Selected Features: ')
    print(candidates)
    print('Final R2: ' + str(best_r2))
    print('Elminated Features: ')
    print(set(data.columns).difference(candidates))

    table = pd.DataFrame(data=rows)
    return table


# Repeats an fit_model to get average of results
def repeat_fit_model(model_class, num_repeats, 
        num_epochs, hidden_sizes, 
        display_steps=False, features=None):
    X_train, X_test, y_train, y_test = read_prep_data(features)
    r2_list = []
    mse_list = []
    max_r2 = 0
    max_run = 0
    best_preds = None
    input_size = X_train.shape[1]
    output_size = y_train.shape[1] if hasattr(y_train, "shape") and len(y_train.shape) > 1 else 1
    for i in range(num_repeats):
        if is_torch_model(model_class):
            # Recreate model on each run so repeats are independent.
            try:
                model = model_class(input_size, hidden_sizes, output_size=output_size)
            except TypeError:
                model = model_class(input_size, hidden_sizes)

            if len(hidden_sizes) != len(model.hidden_layers):
                continue

            predictions, mse, r2, model = fit_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                num_epochs,
                display_steps=display_steps,
                run=i,
            )
        else:
            model = model_class(model_seed + i)
            predictions, mse, r2, model = fit_sklearn_model(
                model, X_train, X_test, y_train, y_test
            )
        if r2 is None:
            continue

        if r2 > max_r2:
            max_r2 = r2
            max_run = i
            best_preds = predictions

        r2_list.append(r2*100)
        mse_list.append(mse)
    if display_steps:
        print(r2_list)

    mean_r2 = np.mean(r2_list) if r2_list else None
    mean_mse = np.mean(mse_list) if mse_list else None
    std_r2 = np.std(r2_list) if r2_list else None
    std_mse = np.std(mse_list) if mse_list else None

    return mean_r2, std_r2, mean_mse, best_preds, max_r2*100, r2_list, max_run

############################### Start of Program ###################
# Sync prep_data with current dataset (NN/data.csv by default).
dataset_path="convert/sugar_all_days_clean_7.csv"
print("====================================") 
print("DATASET PATH: " + dataset_path)
print("====================================") 
data = pd.read_csv(dataset_path)
output_features, input_features = resolve_data_columns(
    data,
    output_feature=None,
    input_features=None
)
print_selection_guide()

answer = input(
    "\n".join([str(i) + ")" + name for i, name in enumerate(data.columns)])
    + "\nSelect one or several output features (separated with space) [last column]:"
)
try:
    selected_output_features = (
        [data.columns[-1]]
        if not answer
        else parse_multi_select(answer, data.columns.tolist(), allow_all=False)
    )
except ValueError as err:
    print(f"Invalid output feature selection: {err}")
    exit()

available_input_features = [col for col in data.columns if col not in selected_output_features]
print("\nAvailable input features (output columns are excluded):")
print("\n".join([str(i) + ")" + name for i, name in enumerate(available_input_features)]))

answer = input(
    "\nSelect one or several input features (indexes/ranges), [all], or [auto]: "
).strip()

use_auto_feature_selection = answer.lower() == "auto"
if use_auto_feature_selection:
    auto_pool_answer = input(
        "Auto mode candidate pool [all] (use ! to exclude, e.g. !0 !3-5): "
    ).strip()
    try:
        selected_input_features = parse_multi_select(
            auto_pool_answer,
            available_input_features,
            allow_all=True,
        )
    except ValueError as err:
        print(f"Invalid auto candidate selection: {err}")
        exit()
else:
    try:
        selected_input_features = parse_multi_select(
            answer,
            available_input_features,
            allow_all=True,
        )
    except ValueError as err:
        print(f"Invalid input feature selection: {err}")
        exit()

sync_prep_data_with_dataset(
    dataset_path=dataset_path,
    prep_folder="prep_data",
    input_features=selected_input_features,
    output_feature=selected_output_features
)

# Load data from prep_data after schema sync.
X_train, X_test, y_train, y_test = read_prep_data(inputs=None, prep_folder="prep_data")

# After loading, get the column names from X_train
inputs = X_train.columns.tolist()
outputs = y_train.columns.tolist()
output = outputs

data = X_train
print("inputs:", inputs)
print("outputs:", outputs)
active_features = list(inputs)
ans = input("Are these inputs and outputs for files in prep_data folder correct?(y/n):")
if ans.strip().lower() not in ("y", "yes"):
    print("Please re-run and choose your preferred input/output features.")
    exit()

# Dynamically collect all neural-network model classes from the module
nn_models = [
    member for name, member in inspect.getmembers(models, inspect.isclass)
    if issubclass(member, models.nn.Module) and member.__module__ == models.__name__
]
sklearn_models = [
    (name, factory) for name, factory in SKLEARN_MODEL_FACTORIES.items()
]
models = nn_models + [factory for _, factory in sklearn_models]
model_names = [model.__name__ for model in nn_models] + [name for name, _ in sklearn_models]
# User input for selecting the model and number of epochs
answer = input("\n".join([str(i) + ")" + name for i,name in enumerate(model_names)]) \
        + "\nSelect one or several models (separated with space) [all]:")

if not answer:
    answer = "all"

try:
    selected_model_names = parse_multi_select(answer, model_names, allow_all=True)
except ValueError as err:
    print(f"Invalid model selection: {err}")
    exit()

if selected_model_names is None:
    selected_models = list(range(len(models)))
else:
    selected_models = [model_names.index(name) for name in selected_model_names]

print("Selected Models:", [model_names[i] for i in selected_models])
best_model_index = selected_models[0]
max_model_index = best_model_index
has_nn_model = any(is_torch_model(models[i]) for i in selected_models)

default_epochs = list(list_epochs)
answer = ask_with_default(
    "Enter epochs (e.g. '20 50 100', add 'cv' for cross-val early stop, or 0 to skip)",
    " ".join([str(e) for e in default_epochs]),
)
list_epochs, use_cv_early_stop = parse_epochs_input(answer, default_epochs)
if answer != "0":
    if not list_epochs and not use_cv_early_stop:
        print("No valid epoch values were provided.")
        exit()
    
    if list_epochs and has_nn_model:
        print("Manual epoch candidates:", list_epochs)
    if use_cv_early_stop and has_nn_model:
        print("Cross-validation early-stop epoch search is enabled.")

    if has_nn_model:
        answer = ask_with_default(
            "Enter hidden sizes (groups split by '#', e.g. '10 5 # 15 10 3')",
            " # ".join([" ".join([str(v) for v in hs]) for hs in list_hidden_sizes]),
        )
        if answer:
           list_hidden_sizes = []
           hs = answer.split("#")
           for ans in hs:
              ans = ans.strip()
              if not ans:
                  continue
              h = [int(a) for a in ans.split() if a.isdigit() and int(a) > 0]
              if h:
                  list_hidden_sizes.append(h)
        if not list_hidden_sizes:
           print("No valid hidden size values were provided.")
           exit()
    else:
        list_hidden_sizes = [[]]
        use_cv_early_stop = False

    answer = ask_with_default("Enter the number of repeating predictions", num_repeats)
    if answer:
       num_repeats = int(answer)
       if num_repeats < 1:
           print("Repeat count must be at least 1.")
           exit()

    if use_auto_feature_selection:
        print("\n================= Auto Feature Selection =================")
        print("1. Backward Feature Elimination")
        print("2. Forward Feature Selection")
        auto_method = input("Select method [1]: ").strip() or "1"

        reference_model = models[selected_models[0]]
        reference_model_name = model_names[selected_models[0]]
        reference_epochs = list_epochs[0] if list_epochs else best_epochs
        reference_hidden_sizes = list_hidden_sizes[0] if is_torch_model(reference_model) else []

        print(
            f"Running feature selection with {reference_model_name}, "
            f"epochs={reference_epochs}, hidden_sizes={reference_hidden_sizes if reference_hidden_sizes else 'N/A'}"
        )

        if auto_method == "2":
            auto_table = forward_feature_selection(
                reference_model,
                data,
                list(inputs),
                output,
                reference_epochs,
                reference_hidden_sizes
            )
        else:
            auto_table = backward_feature_elimination(
                reference_model,
                data,
                list(inputs),
                output,
                reference_epochs,
                reference_hidden_sizes
            )

        active_features = infer_selected_features_from_table(auto_table, inputs)
        print(f"Auto-selected features: {active_features}")
        print("==========================================================\n")


    best_mean_r2 = -1000
    best_mse = -1000
    best_hidden_sizes = []
    best_r2 = -1000
    best_run = 0
    max_epochs = -1
    max_hidden_sizes = []
    best_epochs = -1
    results = []
    model_best_predictions = {}
    # for all models
    for model_index in selected_models:
        model_class = models[model_index]
        model_name = model_names[model_index]
        is_nn = is_torch_model(model_class)
        hidden_size_candidates = list_hidden_sizes if is_nn else [[]]
        for hidden_sizes in hidden_size_candidates:
            epoch_candidates = sorted(set(list_epochs)) if is_nn else [1]
            if is_nn and use_cv_early_stop:
                cv_epoch = select_epochs_with_cv_early_stopping(
                    model_class,
                    hidden_sizes,
                    features=active_features,
                )
                if cv_epoch is not None and cv_epoch > 0:
                    epoch_candidates.append(cv_epoch)
                    print(
                        f"[{model_name} | {hidden_sizes}] CV early-stop suggested epochs: {cv_epoch}"
                    )
                else:
                    print(
                        f"[{model_name} | {hidden_sizes}] CV early-stop failed; using manual epochs only."
                    )

            epoch_candidates = sorted(set(epoch_candidates))
            for num_epochs in epoch_candidates:
                # Apply model on data for N repeats and get predictions, mse and r2
                mean_r2, std_r2, mean_mse, model_best_preds, max_r2, r2_list, max_run = repeat_fit_model(
                    model_class,
                    num_repeats, num_epochs, hidden_sizes, display_steps=True, features=active_features)

                # Keep best seed to generate the same predictions later
                if mean_r2 is None:
                    continue

                if max_r2 > best_r2:
                    best_r2 = max_r2
                    model_best_predictions[model_name] = model_best_preds
                    max_model_index = model_index
                    max_epochs = num_epochs
                    max_hidden_sizes = hidden_sizes
                    best_run = max_run

                if mean_r2 > best_mean_r2:
                    best_mean_r2 = mean_r2
                    best_mse = mean_mse
                    best_model_index = model_index
                    best_epochs = num_epochs
                    best_hidden_sizes = hidden_sizes
                
                total_nodes = sum(hidden_sizes) if hidden_sizes else 0

                result = {
                        "model":model_name, 
                        "R2": round(mean_r2,1), 
                        "MSE": round(mean_mse,2),
                        "R2 std": round(std_r2, 1),
                        "R2 List": [round(x, 1) for x in r2_list],
                        "hidden sizes": hidden_sizes if hidden_sizes else "N/A",
                        "total hs": total_nodes,
                        "epochs": num_epochs if is_nn else "N/A",
                        }
                results.append(result)

    # Create a table for results. Some model/config combinations can fail and return no
    # valid R2 values, so guard against missing/empty result sets.
    expected_result_cols = ["model", "R2", "MSE", "R2 std", "R2 List", "hidden sizes", "total hs", "epochs"]
    results_table = pd.DataFrame(data=results)
    if results_table.empty:
        results_table = pd.DataFrame(columns=expected_result_cols)
    else:
        missing_cols = [col for col in expected_result_cols if col not in results_table.columns]
        for col in missing_cols:
            results_table[col] = np.nan
        # Sort methods by R2 only when the column exists and contains data.
        if results_table["R2"].notna().any():
            results_table = results_table.sort_values(by="R2", ascending=False)
        else:
            print("No valid R2 values were produced for the selected model/config combinations.")

    if results_table["R2"].isna().all():
        print("Training finished, but no successful runs produced R2 values. Skipping reporting/plots.")
        results_table.to_csv("exp.csv", index=False)
        exit()

    latex_table = results_table.copy()
    # Create and save latex code for table
    latex_table["R2"] = latex_table.apply(lambda row: f"{row['R2']} ± {row['R2 std']}", axis=1)
    latex_table = latex_table.drop(columns=["R2 List","R2 std"])
    results_table_latex = generate_latex_table(latex_table, 
            caption="Results of different models", label="models")
    with open(os.path.join("tables", "results.tex"), 'w', encoding='utf-8') as f:
        print(results_table_latex, file=f)

    # Plot the performance of models across different parameters
    print("Generating plots ...")
    plot_model_performance(results_table)

    best_model = models[best_model_index]
    best_model_name = model_names[best_model_index]
    # Show results
    max_model_name = model_names[max_model_index]

    print("============ Results for models =========================")
    print(results_table)
    print("========================== Best Mean Model ===============================")
    print("Best Mean R-Squred:", best_mean_r2)
    print("Best model with better mean R-Squred:", best_model_name) 
    print("Best Hidden sizes:", best_hidden_sizes) 
    print("Best epochs:", best_epochs) 
    print("=========================== Max Model ================================")
    print("Best model with better max R-Squred:", max_model_name) 
    print("Best Hidden sizes:", max_hidden_sizes) 
    print("Best epochs:", max_epochs) 
    print("Max R-Squred:", best_r2)
 
    results_table.to_csv("exp.csv")

    X_train, X_test, y_train, y_test = read_prep_data(active_features)

    # Show and save the plot for best results
    best_predictions = model_best_predictions[max_model_name] 
    output_title = ", ".join(outputs)
    title = "Prediction of " + output_title + " with " + max_model_name + " epochs:" + str(max_epochs)
    file_name = f"R2-{best_r2:.2f}-" + max_model_name + "-" + "-".join(outputs) + ".png"

    y_test_values = y_test.values if hasattr(y_test, "values") else y_test
    best_model_metrics = compute_regression_report_metrics(y_test_values, best_predictions)
    if best_model_metrics:
        print("===================== Best Selected Model Report =====================")
        print(f"R-Squared                              {best_r2/100:.4f}")
        print(
            f"Correlation coefficient                {best_model_metrics['Correlation coefficient']:.4f}"
        )
        print(
            f"Mean absolute error                    {best_model_metrics['Mean absolute error']:.4f}"
        )
        print(
            f"Root mean squared error                {best_model_metrics['Root mean squared error']:.4f}"
        )
        print(
            f"Relative absolute error                {best_model_metrics['Relative absolute error']:.4f} %"
        )
        print(
            f"Root relative squared error            {best_model_metrics['Root relative squared error']:.4f} %"
        )
        print(
            f"Total Number of Instances              {best_model_metrics['Total Number of Instances']}"
        )
        print("=====================================================================")

    print("\n\n")
    print("Plot was saved in plots folder")
    answer = input("Do you want to see them? [y]:") 
    if len(outputs) == 1:
        target_series = y_test[outputs[0]]
        pred_series = best_predictions[:, 0] if best_predictions.ndim > 1 else best_predictions
        if answer == "y" or answer == "yes":
            plot_results(pred_series, target_series, title, file_name, show_plot=True)
        else:
            plot_results(pred_series, target_series, title, file_name, show_plot=False)
    else:
        print("Skipping scatter plot for multi-output predictions.")

    # Save results of predicitons in a file named results.csv
    results_df = pd.DataFrame()
    for i, output_name in enumerate(outputs):
        results_df[f"{output_name}_actual"] = y_test[output_name].values
        pred_col = best_predictions[:, i] if best_predictions.ndim > 1 else best_predictions
        results_df[f"{output_name}_predictions"] = np.round(pred_col, 2)
    results_df.to_csv("results.csv", index=False)
    print("Predictions of best model were saved in results.csv")
    answer = input("Do you want to see them? [y]:") 
    if answer == "y" or answer == "yes":
       print("======= Predictions of best model:", best_model_name)
       print(results_df)

best_model = models[best_model_index]
best_model_name = model_names[best_model_index]
if is_torch_model(best_model):
    while True:
        print("\n\n")
        print(f"================= Feature Selection ({best_model_name}:{best_epochs} epochs, {best_hidden_sizes}) ======")
        print("\nPlease select a feature selection or sensitivity analysis method:\n")
        print("1. Backward Feature Elimination")
        print("2. Forward Feature Selection")
        print("3. Weight Analysis")
        print("4. Jackknife Sensitivity Analysis (Node Deletion Sensitivity)")
        print("5. SHAP (SHapley Additive exPlanations)")
        print("q. Quit")

        answer = input("Enter the number of the method you want to run (or 'q' to quit): ").strip().lower()

        if answer == '1':
            print("============================= Backward Feature Elimination =============")
            backward_table = backward_feature_elimination(best_model, data, inputs, output, best_epochs, best_hidden_sizes)
            print("------------ backward feature elimination ---------------")
            print(backward_table)

            backward_table_latex = generate_latex_table(backward_table, caption="Results of Backward Feature Elimination", label="backward")
            with open(os.path.join("tables", "backward.tex"), 'w', encoding='utf-8') as f:
                print(backward_table_latex, file=f)

        elif answer == '2':
            print("============================= Forward Feature Selection ================")
            forward_table = forward_feature_selection(best_model, data, inputs, output,
                                                      best_epochs, best_hidden_sizes)
            print("\n")
            print("------------ forward feature selection ---------------")
            print(forward_table)
            forward_table_latex = generate_latex_table(forward_table, caption="Results of Forward Feature Selection for different features", label="forward")
            with open(os.path.join("tables", "forward.tex"), 'w', encoding='utf-8') as f:
                print(forward_table_latex, file=f)

        elif answer == '3':
            print("============================= Weight Analysis =============")
            weight_table = weight_analysis(best_model, data, inputs, output, best_epochs, best_hidden_sizes)
            print("------------ weight analysis ---------------")
            weight_table = weight_table.sort_values(by='Weight Importance',
                    ascending=False)
            print("Most important features:")
            print(weight_table)
            weight_table_latex = generate_latex_table(weight_table, caption="Results of Weight Analysis", label="weight_analysis")
            with open(os.path.join("tables", "weight-analysis.tex"), 'w', encoding='utf-8') as f:
                print(weight_table_latex, file=f)

        elif answer == '4':
            print("============================= Jackknife Sensitivity Analysis =============")
            jackknife_table = jackknife_sensitivity_analysis(best_model,
                    data, inputs, output, best_epochs, best_hidden_sizes)
            print("------------ jackknife sensitivity analysis ---------------")
            jackknife_table = jackknife_table.sort_values(by='Sensitivity', ascending=False)
            print(jackknife_table)
            jackknife_table_latex = generate_latex_table(jackknife_table, caption="Results of Jackknife Sensitivity Analysis", label="jackknife")
            with open(os.path.join("tables", "jackknife.tex"), 'w', encoding='utf-8') as f:
                print(jackknife_table_latex, file=f)

            important_features = jackknife_table.head(5)['Feature'].tolist()
            print("\nTop 5 important features to focus on:")
            print(important_features)

            negative_sensitivity_features = jackknife_table[jackknife_table['Sensitivity'] < 0]
            print("\nFeatures with negative sensitivity (potentially redundant or harmful):")
            print(negative_sensitivity_features)

        elif answer == '5':
            print("============================= SHAP Feature Importance =============")
            shap_table = shap_feature_importance(best_model, inputs, best_epochs, best_hidden_sizes)
            if shap_table is not None:
                print("------------ SHAP feature importance ---------------")
                print(shap_table)
                shap_table_latex = generate_latex_table(shap_table, caption="Results of SHAP Feature Importance", label="shap")
                with open(os.path.join("tables", "shap.tex"), 'w', encoding='utf-8') as f:
                    print(shap_table_latex, file=f)
                shap_table.to_csv(os.path.join("tables", "shap.csv"), index=False)

        elif answer == 'q':
            print("Exiting the feature selection and sensitivity analysis loop. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")
        print("\n----------------------------------------------------------")
        input("Press any key to return to main menu ...")
else:
    print(f"Skipping feature-selection menu because best model '{best_model_name}' is a classical sklearn model.")



print("----------------------Important! READ -------------------")
print("latex code for tables are saved in tables folder")
print("predictions are saved in results.csv file")
input("Plots have been saved in the 'plots' folder. (press any key to exit)")

# Visualize the model and save it on mlp_structure image
# dummy_input = torch.randn(1, input_size)
# from torchviz import make_dot
# dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
# dot.render("mlp_structure", format="png", cleanup=True)
