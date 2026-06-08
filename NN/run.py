from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
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
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score
from tabulate import tabulate
import os
from latex import *
from plot import *
from models import *
from read_data import (
    read_prep_data,
    sync_prep_data_with_dataset,
    prep_data_exists,
    read_prep_metadata,
    save_data,
    print_path_debug,
)
from refinery_variables import (
    CONTROL_VARIABLES,
    EARLY_VARIABLES,
    TARGET_VARIABLES,
    filter_allowed_model_inputs,
    find_leakage_columns,
    refinery_variable_group_metadata,
    validate_model_inputs,
)
from recommendation_engine import RecommendationError, recommend_operating_conditions
import inspect
import models
import json
import copy
from datetime import datetime

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
SAVED_RUNS_DIR = "saved_runs"
CHECKPOINTS_DIR = "checkpoints"


FUTURE_QUALITY_INPUT_CANDIDATES = [
    "filtercake_moisture",
    "filtercake_sugar",
    "sweetwater_brix",
    "sulphited_pH",
    "sulphited_brix",
    "sulphited_color",
    "standard_liquor_pH",
    "standard_liquor_brix",
    "standard_liquor_color",
]


def color_text(text, color):
    return f"{color}{text}{ANSI_RESET}"


def print_divider(char="=", width=68):
    print(char * width)


def print_numbered_feature_list(title, features, color):
    print_divider("=")
    print(title)
    print_divider("-")
    for idx, feature_name in enumerate(features, start=1):
        print(color_text(f"{idx:>2}. {feature_name}", color))
    print_divider("=")


def print_refinery_variable_groups():
    """Explain the industrial refinery groups used to avoid leakage."""
    print_divider("=")
    print("Industrial refinery variable logic")
    print_divider("-")
    print("EARLY_VARIABLES (available early): " + ", ".join(EARLY_VARIABLES))
    print("CONTROL_VARIABLES (operator-adjustable): " + ", ".join(CONTROL_VARIABLES))
    print("TARGET_VARIABLES (future quality outputs; never inputs): " + ", ".join(TARGET_VARIABLES))
    print("Input rule: models train only on EARLY_VARIABLES + CONTROL_VARIABLES.")
    print("Leakage rule: selected outputs and TARGET_VARIABLES are blocked from X.")
    print_divider("=")


def _format_industrial_value(value):
    """Format values for compact refinery-control demo tables."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):,.3f}"
    try:
        return f"{float(value):,.3f}"
    except (TypeError, ValueError):
        return str(value)


def _risk_badge(risk_level):
    """Return a professional text badge for LOW/MEDIUM/HIGH risk."""
    risk_level = str(risk_level).upper()
    badges = {
        "LOW": "LOW - Normal operating band",
        "MEDIUM": "MEDIUM - Watch and correct",
        "HIGH": "HIGH - Immediate operator attention",
    }
    return badges.get(risk_level, risk_level)


def print_industrial_operator_demo(recommendation, input_features, output_features):
    """Print a realistic step-by-step operator demonstration.

    The section mirrors the requested plant workflow: current conditions are
    entered, the AI predicts future sugar quality, the AI recommends controllable
    variables, then it prints expected quality and risk in a readable shift-log
    format.
    """
    current_inputs = recommendation.get("full_candidate_inputs", {}).copy()
    current_settings = recommendation.get("current_settings", {})
    for variable, value in current_settings.items():
        current_inputs[variable] = value

    recommended = recommendation.get("recommended_settings", {})
    current_prediction = recommendation.get("current_prediction", {})
    future_quality = recommendation.get("predicted_future_quality", {})
    current_risk = recommendation.get("current_risk_prediction", {})
    future_risk = recommendation.get("risk_prediction", {})
    control_ranges = recommendation.get("control_ranges", {})

    print("\n")
    print_divider("=", 76)
    print("INDUSTRIAL OPERATOR DEMO - AI REFINERY QUALITY ADVISOR")
    print_divider("=", 76)
    print("Scenario: shift operator enters live refinery conditions and requests an")
    print("AI advisory for future sugar quality, optimal controllable set-points,")
    print("expected future quality, and operating risk.")

    print("\n1) OPERATOR ENTERS CURRENT REFINERY CONDITIONS")
    print_divider("-", 76)
    condition_rows = []
    for feature in input_features:
        if feature not in current_inputs:
            continue
        group = "Controllable" if feature in CONTROL_VARIABLES else "Early/process"
        condition_rows.append([group, feature, _format_industrial_value(current_inputs[feature])])
    print(tabulate(condition_rows, headers=["Group", "Variable", "Current value"], tablefmt="github"))

    print("\n2) AI PREDICTS FUTURE SUGAR QUALITY AT CURRENT CONDITIONS")
    print_divider("-", 76)
    current_quality_rows = []
    for target in output_features:
        if target in current_prediction:
            current_quality_rows.append([target, _format_industrial_value(current_prediction[target])])
    print(tabulate(current_quality_rows, headers=["Future quality target", "Predicted value"], tablefmt="github"))
    print(f"Current-condition risk: {_risk_badge(current_risk.get('risk_level', 'UNKNOWN'))}")

    print("\n3) AI RECOMMENDS OPTIMAL CONTROLLABLE VARIABLES")
    print_divider("-", 76)
    recommendation_rows = []
    for variable, recommended_value in recommended.items():
        low, high = control_ranges.get(variable, (None, None))
        safe_range = "N/A" if low is None or high is None else f"{low:.3f} - {high:.3f}"
        current_value = current_settings.get(variable, current_inputs.get(variable))
        delta = float(recommended_value) - float(current_value)
        recommendation_rows.append([
            variable,
            _format_industrial_value(current_value),
            _format_industrial_value(recommended_value),
            f"{delta:+.3f}",
            safe_range,
        ])
    print(tabulate(
        recommendation_rows,
        headers=["Control variable", "Current", "Recommended", "Change", "Search range"],
        tablefmt="github",
    ))

    print("\n4) AI PRINTS EXPECTED FUTURE QUALITY AFTER RECOMMENDATION")
    print_divider("-", 76)
    future_quality_rows = []
    for target in output_features:
        if target not in future_quality:
            continue
        before = current_prediction.get(target)
        after = future_quality[target]
        improvement = "N/A" if before is None else f"{float(before) - float(after):+.3f}"
        future_quality_rows.append([
            target,
            _format_industrial_value(before) if before is not None else "N/A",
            _format_industrial_value(after),
            improvement,
        ])
    print(tabulate(
        future_quality_rows,
        headers=["Future quality target", "Current prediction", "Expected with AI set-points", "Improvement"],
        tablefmt="github",
    ))

    print("\n5) AI PRINTS INDUSTRIAL RISK LEVEL")
    print_divider("-", 76)
    print(f"Recommended-condition risk: {_risk_badge(future_risk.get('risk_level', 'UNKNOWN'))}")
    risk_driver_rows = []
    for item in future_risk.get("risk_drivers", []):
        risk_driver_rows.append([
            item.get("target"),
            _format_industrial_value(item.get("predicted_value")),
            item.get("risk_level"),
            item.get("reason"),
        ])
    if risk_driver_rows:
        print(tabulate(
            risk_driver_rows,
            headers=["Risk driver", "Predicted value", "Level", "Reason"],
            tablefmt="github",
        ))
    print("Operator advisory:")
    for warning in future_risk.get("operator_warnings", []):
        print(f" - {warning}")
    print(f"AI candidate simulations reviewed: {recommendation.get('searched_candidates', 'N/A')}")
    print_divider("=", 76)


def list_saved_runs(saved_dir=SAVED_RUNS_DIR):
    os.makedirs(saved_dir, exist_ok=True)
    saved = []
    for file_name in sorted(os.listdir(saved_dir)):
        if not file_name.endswith(".json"):
            continue
        path = os.path.join(saved_dir, file_name)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            payload["_path"] = path
            saved.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return saved


def prompt_saved_run_choice():
    while True:
        saved = list_saved_runs()
        if not saved:
            return None, False
        print("\nSaved runs:")
        table_rows = []
        for idx, run in enumerate(saved, start=1):
            table_rows.append(
                [
                    idx,
                    run.get("display_name", "unknown"),
                    run.get("best_r2", "N/A"),
                    run.get("saved_at", "N/A"),
                    "yes" if run.get("has_weights") else "no",
                ]
            )
        print(tabulate(table_rows, headers=["#", "Saved model", "R2", "Date", "Weights"], tablefmt="github"))
        print("Tip: type delete_<index> (example: delete_3) to remove a saved model/checkpoint.")
        choice = input("Select a saved run by index or press Enter for new run [new]: ").strip()
        if not choice:
            return None, False
        if choice.lower().startswith("delete_"):
            index_str = choice.split("_", 1)[1].strip()
            if not index_str.isdigit():
                print("Invalid delete format. Use delete_<index> such as delete_3.")
                continue
            idx = int(index_str)
            if idx < 1 or idx > len(saved):
                print("Invalid delete index.")
                continue
            selected_for_delete = saved[idx - 1]
            summary_path = selected_for_delete.get("_path")
            weights_path = selected_for_delete.get("weights_path")
            try:
                if summary_path and os.path.isfile(summary_path):
                    os.remove(summary_path)
                    print(f"Deleted summary file: {summary_path}")
                if weights_path and os.path.isfile(weights_path):
                    os.remove(weights_path)
                    print(f"Deleted checkpoint file: {weights_path}")
            except OSError as err:
                print(f"Could not delete saved model files ({err}).")
            continue
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(saved):
            print("Invalid choice. Starting a new run.")
            return None, False
        selected = saved[int(choice) - 1]
        has_weights = bool(selected.get("has_weights"))
        if has_weights:
            mode = input(
                "Selected saved run has model weights. Continue by optimizing [weights|inputs|both] [weights]: "
            ).strip().lower() or "weights"
        else:
            mode = input(
                "Selected saved run has configs only (no weights). Continue by optimizing [weights|inputs|both] [weights]: "
            ).strip().lower() or "weights"
        return selected, mode


def save_run_summary(
    model_name,
    inputs,
    outputs,
    best_r2,
    epoch_candidates=None,
    hidden_size_groups=None,
    repeat_count=None,
    optimization_scope=None,
    weights_path=None,
    saved_dir=SAVED_RUNS_DIR,
):
    os.makedirs(saved_dir, exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    display_name = f"{model_name}_in{len(inputs)}_out{len(outputs)}_R2-{best_r2:.2f}"
    payload = {
        "display_name": display_name,
        "model_name": model_name,
        "input_count": len(inputs),
        "output_count": len(outputs),
        "inputs": inputs,
        "outputs": outputs,
        "best_r2": round(float(best_r2), 2),
        "saved_at": now,
        "epoch_candidates": epoch_candidates if epoch_candidates is not None else [],
        "hidden_size_groups": hidden_size_groups if hidden_size_groups is not None else [],
        "repeat_count": int(repeat_count) if repeat_count is not None else None,
        "optimization_scope": optimization_scope,
        "weights_path": weights_path,
        "has_weights": bool(weights_path and os.path.isfile(weights_path)),
    }
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", display_name)
    summary_path = os.path.join(saved_dir, f"{safe_name}.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    print(f"Saved run summary: {summary_path}")


def normalize_hidden_size_groups(raw_groups, fallback_groups):
    """Return valid hidden-size groups as list[list[int]]."""
    parsed = []
    if isinstance(raw_groups, list):
        for group in raw_groups:
            if isinstance(group, list):
                cleaned = [int(v) for v in group if str(v).isdigit() and int(v) > 0]
                if cleaned:
                    parsed.append(cleaned)
    return parsed if parsed else [list(group) for group in fallback_groups]

list_epochs = [50, 100, 200, 300]
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

learning_rate = 0.0005
# Industrially conservative ANN weight learning rate. 0.05 was too aggressive
# for a small, standardized tabular dataset and can overshoot narrow minima.
input_learning_rate = 0.00005
# Input optimization is more sensitive than weight optimization because it moves
# the normalized training data directly, so keep it an order of magnitude lower.
ann_weight_decay = 1e-4
# L2 regularization reduces small-sample overfitting without increasing model size.
early_stopping_patience = 20
early_stopping_min_delta = 1e-4
validation_fraction = 0.2
lr_plateau_patience = 8
lr_plateau_factor = 0.5
min_learning_rate = 1e-5
max_gradient_norm = 0.5
max_input_delta = 1.5
# Conservative training guards for stable convergence on small industrial data.
#hidden_size1 = 10
hidden_size1 = 8
hidden_size2 = 4

# the number of neurons in hidden layers

# https://alexlenail.me/NN-SVG/
# use the site above to draw the following network
#
hidden_sizes = [8, 4]
# nn.ReLU(), nn.Tanh(), nn.Identity()

list_hidden_sizes = [[4], [6], [8], [8, 4]]
normalization_type = "standard_scaler"


SKLEARN_MODEL_FACTORIES = {
    "RandomForestRegressor": lambda seed: RandomForestRegressor(
        n_estimators=300, random_state=seed
    ),
    "ExtraTreesRegressor": lambda seed: ExtraTreesRegressor(
        n_estimators=300, random_state=seed
    ),
    "GradientBoostingRegressor": lambda seed: GradientBoostingRegressor(random_state=seed),
    "GBMRegressor": lambda seed: GradientBoostingRegressor(random_state=seed),
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

if XGBRegressor is not None:
    SKLEARN_MODEL_FACTORIES["XGBoostRegressor"] = lambda seed: XGBRegressor(
        n_estimators=300,
        random_state=seed,
        objective="reg:squarederror",
    )
else:
    print("Optional dependency not found: xgboost. Skipping XGBoostRegressor.")

if LGBMRegressor is not None:
    SKLEARN_MODEL_FACTORIES["LightGBMRegressor"] = lambda seed: LGBMRegressor(
        n_estimators=300,
        random_state=seed,
        verbose=-1,
    )
else:
    print("Optional dependency not found: lightgbm. Skipping LightGBMRegressor.")


def _expand_index_token(token, options_len, one_based=False):
    """Expand a token like '3' or '1-4' into zero-based indexes."""
    if "-" in token:
        bounds = token.split("-", 1)
        if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
            raise ValueError(f"Invalid range '{token}'. Use syntax like 1-4.")
        start = int(bounds[0])
        end = int(bounds[1])
        if start > end:
            raise ValueError(f"Invalid range '{token}'. Start must be <= end.")
        raw_indexes = list(range(start, end + 1))
    else:
        if not token.isdigit():
            raise ValueError(f"Invalid input '{token}'. Please use indexes or ranges like 1-4.")
        raw_indexes = [int(token)]

    indexes = [index - 1 for index in raw_indexes] if one_based else raw_indexes
    min_valid = 1 if one_based else 0
    max_valid = options_len if one_based else options_len - 1
    for raw_index, index in zip(raw_indexes, indexes):
        if index < 0 or index >= options_len:
            raise ValueError(
                f"Invalid selection '{raw_index}'. Valid range is {min_valid} to {max_valid}."
            )
    return indexes


def parse_multi_select(answer, options, allow_all=True, one_based=False):
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

        indexes = _expand_index_token(clean, len(options), one_based=one_based)
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


def print_future_quality_variables_menu():
    print("\n" + "-" * 50)
    print("Future Quality Variables")
    for idx, variable_name in enumerate(FUTURE_QUALITY_INPUT_CANDIDATES, start=1):
        print(f"{idx}) {variable_name}")
    print("-" * 50)


def append_future_quality_input_candidates(
    active_input_features,
    available_columns,
    selected_output_features=None,
    prompt_label="Select future-quality variables to add as candidate input features",
):
    """Prompt for optional future-quality inputs and append valid, unique selections.

    Future-quality variables are normally targets. This opt-in prompt lets the
    user include available non-target quality measurements as normal model
    inputs for future-quality prediction while preserving feature order and
    preventing duplicates.
    """
    active_input_features = list(active_input_features)
    available_columns = list(available_columns)
    selected_output_features = set(selected_output_features or [])

    print_future_quality_variables_menu()
    answer = input(
        f"{prompt_label} (indexes/ranges), [all], or Enter for none [none]: "
    ).strip()
    if not answer:
        print("No additional future-quality input variables selected.")
        return active_input_features, []

    try:
        selected_variables = parse_multi_select(
            answer,
            FUTURE_QUALITY_INPUT_CANDIDATES,
            allow_all=True,
            one_based=True,
        )
    except ValueError as err:
        print(f"Invalid future-quality input selection: {err}")
        exit()

    if selected_variables is None:
        selected_variables = list(FUTURE_QUALITY_INPUT_CANDIDATES)

    appended_variables = []
    skipped_missing = []
    skipped_outputs = []
    skipped_duplicates = []

    for variable_name in selected_variables:
        if variable_name in selected_output_features:
            skipped_outputs.append(variable_name)
            continue
        if variable_name not in available_columns:
            skipped_missing.append(variable_name)
            continue
        if variable_name in active_input_features:
            skipped_duplicates.append(variable_name)
            continue
        active_input_features.append(variable_name)
        appended_variables.append(variable_name)

    if appended_variables:
        print("Added future-quality candidate input feature(s): " + ", ".join(appended_variables))
    else:
        print("No future-quality variables were added to the input feature list.")

    if skipped_outputs:
        print(
            "Warning: selected future-quality variable(s) are current output target(s) "
            "and were skipped to avoid target leakage: " + ", ".join(skipped_outputs)
        )
    if skipped_missing:
        print(
            "Warning: selected future-quality variable(s) are not present in X_train.columns "
            "and were skipped: " + ", ".join(skipped_missing)
        )
    if skipped_duplicates:
        print(
            "Already selected as input feature(s); duplicates were ignored: "
            + ", ".join(skipped_duplicates)
        )

    return active_input_features, appended_variables


def print_selection_guide():
    print("\nSelection guide:")
    print("- Single indexes: 1 3 5")
    print("- Range syntax: 1-9")
    print("- Mixed syntax: 1 2-4 7")
    print("- Commas are also allowed: 1,2-4,7")
    print("- Exclude indexes/ranges with ! : !1 !5-8")
    print("- In auto mode, exclude-only means all features except those indexes.")


def prompt_temporal_options(input_features):
    print("\nTemporal/sequential feature options:")
    print("Enter indexes/ranges for sequential columns if any (empty for none).")
    print("Optional grouped dependencies: [2 4] [3 11 13] or 2>4>6,3>11>13")
    print("\n".join([str(i) + ")" + name for i, name in enumerate(input_features, start=1)]))
    seq_answer = input("Sequential feature columns/groups [none]: ").strip()
    if not seq_answer:
        return [], [], False, 1, False, False, False, False, 3, False, 3

    try:
        sequential_features, sequential_groups = parse_sequential_layout(seq_answer, input_features, one_based=True)
    except ValueError as err:
        print(f"Invalid sequential feature selection: {err}")
        exit()

    add_differences = input("Add difference features for sequential columns? [n]: ").strip().lower() in ("y", "yes")
    difference_order = 1
    if add_differences:
        difference_order = int(input("Difference order [1]: ").strip() or "1")

    add_ratio_features = input("Add ratio features (x_t / x_{t-1})? [n]: ").strip().lower() in ("y", "yes")
    add_acceleration_features = input(
        "Add acceleration features (x_t - 2*x_{t-1} + x_{t-2})? [n]: "
    ).strip().lower() in ("y", "yes")
    add_normalized_change_features = input(
        "Add normalized change features ((x_t - x_{t-1}) / x_{t-1})? [n]: "
    ).strip().lower() in ("y", "yes")

    create_rnn_windows = input("Add lag-window features for sequential columns (lag_1, lag_2, lag_3 by default)? [n]: ").strip().lower() in ("y", "yes")
    rnn_window_size = 3
    if create_rnn_windows:
        rnn_window_size = int(input("Largest lag step [3]: ").strip() or "3")

    add_rolling_dynamics = input(
        "Add rolling process dynamics (mean/std/range/slope)? [n]: "
    ).strip().lower() in ("y", "yes")
    rolling_window = 3
    if add_rolling_dynamics:
        rolling_window = int(input("Rolling dynamics window [3]: ").strip() or "3")

    return (
        sequential_features,
        sequential_groups,
        add_differences,
        difference_order,
        add_ratio_features,
        add_acceleration_features,
        add_normalized_change_features,
        create_rnn_windows,
        rnn_window_size,
        add_rolling_dynamics,
        rolling_window,
    )


def parse_sequential_layout(answer, options, one_based=False):
    """Parse independent sequential columns and optional ordered dependency groups."""
    grouped_layout = (
        "[" in answer or "]" in answer or "(" in answer or ")" in answer or ">" in answer
    )
    if not grouped_layout:
        sequential_features = parse_multi_select(answer, options, allow_all=False, one_based=one_based)
        return sequential_features, []

    groups = []
    bracket_groups = re.findall(r"[\[\(]([^\]\)]+)[\]\)]", answer)
    raw_groups = bracket_groups if bracket_groups else [item.strip() for item in answer.split(",") if item.strip()]

    for raw_group in raw_groups:
        normalized = raw_group.replace(">", " ")
        group_values = parse_multi_select(normalized, options, allow_all=False, one_based=one_based)
        if len(group_values) < 2:
            raise ValueError("Each sequential group must contain at least two columns.")
        groups.append(group_values)

    flat_features = []
    for group in groups:
        for feature in group:
            if feature not in flat_features:
                flat_features.append(feature)
    return flat_features, groups


def prepare_or_reuse_data(dataset_path="convert/sugar_all.csv", prep_folder="prep_data"):
    print("====================================")
    print("DATASET PATH: " + dataset_path)
    print("====================================")
    print(f"[path-debug] run.py file: {os.path.abspath(__file__)}")
    dataset_path = print_path_debug("raw dataset CSV", dataset_path)
    prep_folder = print_path_debug("prep_data folder", prep_folder)
    print(f"[path-debug] Effective dataset path: {dataset_path}")
    print(f"[path-debug] Effective prep_data folder: {prep_folder}")

    use_auto_feature_selection = False
    reused_prep_data = False
    print_refinery_variable_groups()

    if prep_data_exists(prep_folder):
        prep_x_train_path = os.path.join(prep_folder, "X_train.csv")
        prep_y_train_path = os.path.join(prep_folder, "y_train.csv")
        existing_all_inputs = pd.read_csv(prep_x_train_path, nrows=0).columns.tolist()
        existing_outputs = pd.read_csv(prep_y_train_path, nrows=0).columns.tolist()
        leaked_existing_inputs = find_leakage_columns(
            existing_all_inputs,
            output_features=existing_outputs,
            optional_future_quality_inputs=FUTURE_QUALITY_INPUT_CANDIDATES,
        )
        if leaked_existing_inputs:
            print(
                "Existing prep_data contains future/disallowed inputs and will be rebuilt "
                f"to prevent target leakage: {leaked_existing_inputs}"
            )
        else:
            X_train_existing, _, y_train_existing, _ = read_prep_data(inputs=None, prep_folder=prep_folder)
            existing_inputs = X_train_existing.columns.tolist()
            print_numbered_feature_list("Prepared Input Features (prep_data)", existing_inputs, ANSI_GREEN)
            existing_inputs, optional_future_quality_inputs = append_future_quality_input_candidates(
                existing_inputs,
                existing_all_inputs,
                selected_output_features=existing_outputs,
            )
            if optional_future_quality_inputs:
                X_train_existing, _, y_train_existing, _ = read_prep_data(
                    inputs=existing_inputs,
                    prep_folder=prep_folder,
                    optional_future_quality_inputs=optional_future_quality_inputs,
                )
            print_numbered_feature_list("Prepared Output Features (prep_data)", existing_outputs, ANSI_BLUE)

            metadata = read_prep_metadata(prep_folder)
            if metadata:
                print("prep_data metadata:")
                print(metadata)

            print(
                "[path-debug] Complete prep_data files were found. If you continue with prep_data, "
                "the raw dataset CSV will not be read in this run."
            )
            reuse_answer = input(
                "Continue with these prepared refinery-safe inputs/outputs? [y]: "
            ).strip().lower()
            if reuse_answer in ("", "y", "yes"):
                print("[path-debug] prep_data reuse selected; skipping raw dataset CSV read.")
                reused_prep_data = True
                return X_train_existing, existing_outputs, use_auto_feature_selection, reused_prep_data
            print("[path-debug] prep_data reuse declined; raw dataset CSV flow will be used.")

            prep_train_path = os.path.join(prep_folder, "train.csv")
            prep_test_path = os.path.join(prep_folder, "test.csv")

            if os.path.isfile(prep_train_path) and os.path.isfile(prep_test_path):
                print("\nYou chose not to continue with current prep_data selection.")
                print("1) Reselect input features from prep_data/train.csv and prep_data/test.csv")
                print("2) Rebuild prep_data from the raw dataset and select input/output/sequential options again")
                reselect_mode = input("Choose option [1]: ").strip()
                if reselect_mode in ("", "1"):
                    train_df = pd.read_csv(prep_train_path)
                    test_df = pd.read_csv(prep_test_path)
                    available_inputs = filter_allowed_model_inputs(
                        train_df.columns, output_features=existing_outputs
                    )
                    if not available_inputs:
                        print("No refinery-safe candidate input columns found. Falling back to full prepare flow.")
                    else:
                        print_numbered_feature_list(
                            "Refinery-safe Input Features from prep_data/train.csv",
                            available_inputs,
                            ANSI_GREEN,
                        )
                        reselect_answer = input(
                            "\nSelect one or several input features (indexes/ranges), or [all]: "
                        ).strip()
                        try:
                            selected_input_features = parse_multi_select(
                                reselect_answer,
                                available_inputs,
                                allow_all=True,
                                one_based=True,
                            )
                        except ValueError as err:
                            print(f"Invalid input feature selection: {err}")
                            exit()

                        resolved_inputs = (
                            available_inputs if selected_input_features is None else selected_input_features
                        )
                        resolved_inputs, optional_future_quality_inputs = append_future_quality_input_candidates(
                            resolved_inputs,
                            train_df.columns,
                            selected_output_features=existing_outputs,
                        )
                        validate_model_inputs(
                            resolved_inputs,
                            output_features=existing_outputs,
                            optional_future_quality_inputs=optional_future_quality_inputs,
                        )
                        missing_inputs_in_test = [
                            col for col in resolved_inputs if col not in test_df.columns
                        ]
                        if missing_inputs_in_test:
                            print(
                                "Selected inputs are missing from prep_data/test.csv: "
                                + str(missing_inputs_in_test)
                            )
                            print("Please rebuild prep_data from the raw dataset.")
                            exit()

                        missing_outputs_in_test = [
                            col for col in existing_outputs if col not in test_df.columns
                        ]
                        if missing_outputs_in_test:
                            print(
                                "Prepared output columns are missing from prep_data/test.csv: "
                                + str(missing_outputs_in_test)
                            )
                            print("Please rebuild prep_data from the raw dataset.")
                            exit()

                        X_train_selected = train_df[resolved_inputs]
                        X_test_selected = test_df[resolved_inputs]
                        y_train_selected = train_df[existing_outputs]
                        y_test_selected = test_df[existing_outputs]
                        metadata = read_prep_metadata(prep_folder)
                        if metadata:
                            metadata["input_features"] = list(resolved_inputs)
                            metadata["refinery_variable_groups"] = refinery_variable_group_metadata()
                        save_data(
                            prep_folder,
                            X_train_selected,
                            X_test_selected,
                            y_train_selected,
                            y_test_selected,
                            metadata=metadata,
                        )
                        print("prep_data updated with the newly selected refinery-safe input features.")
                        reused_prep_data = True
                        return (
                            X_train_selected,
                            existing_outputs,
                            use_auto_feature_selection,
                            reused_prep_data,
                        )
                elif reselect_mode == "2":
                    print("Preparing data again from the raw dataset...")
                else:
                    print("Invalid choice. Preparing data again from the raw dataset...")
            else:
                print("prep_data/train.csv or prep_data/test.csv not found. Preparing data again from the raw dataset...")

    print(f"[path-debug] prep_data did not prevent raw CSV flow; reading dataset now: {dataset_path}")
    data = pd.read_csv(dataset_path)
    print_selection_guide()

    target_candidates = [col for col in data.columns if col in TARGET_VARIABLES]
    output_options = target_candidates if target_candidates else data.columns.tolist()
    print("\nFuture quality target candidates:")
    print("\n".join([str(i) + ")" + name for i, name in enumerate(output_options, start=1)]))
    answer = input(
        "Select one or several TARGET_VARIABLES/output features (indexes/ranges) [last target candidate]:"
    )
    try:
        selected_output_features = (
            [output_options[-1]]
            if not answer
            else parse_multi_select(answer, output_options, allow_all=False, one_based=True)
        )
    except ValueError as err:
        print(f"Invalid output feature selection: {err}")
        exit()

    available_input_features = filter_allowed_model_inputs(
        data.columns, output_features=selected_output_features
    )
    if not available_input_features:
        print(
            "No EARLY_VARIABLES or CONTROL_VARIABLES were found in the dataset. "
            "Cannot train without refinery-safe inputs."
        )
        exit()
    print("\nAvailable model inputs (EARLY_VARIABLES + CONTROL_VARIABLES only):")
    print("\n".join([str(i) + ")" + name for i, name in enumerate(available_input_features, start=1)]))

    answer = input(
        "\nSelect one or several input features (indexes/ranges), [all], or [auto]: "
    ).strip()

    use_auto_feature_selection = answer.lower() == "auto"
    if use_auto_feature_selection:
        auto_pool_answer = input(
            "Auto mode candidate pool [all refinery-safe inputs] (use ! to exclude, e.g. !1 !3-5): "
        ).strip()
        try:
            selected_input_features = parse_multi_select(
                auto_pool_answer,
                available_input_features,
                allow_all=True,
                one_based=True,
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
                one_based=True,
            )
        except ValueError as err:
            print(f"Invalid input feature selection: {err}")
            exit()

    resolved_inputs = (
        available_input_features if selected_input_features is None else selected_input_features
    )
    resolved_inputs, optional_future_quality_inputs = append_future_quality_input_candidates(
        resolved_inputs,
        data.columns,
        selected_output_features=selected_output_features,
    )
    validate_model_inputs(
        resolved_inputs,
        output_features=selected_output_features,
        optional_future_quality_inputs=optional_future_quality_inputs,
    )
    (
        sequential_features,
        sequential_groups,
        add_differences,
        difference_order,
        add_ratio_features,
        add_acceleration_features,
        add_normalized_change_features,
        create_rnn_windows,
        rnn_window_size,
        add_rolling_dynamics,
        rolling_window,
    ) = prompt_temporal_options(resolved_inputs)

    sync_prep_data_with_dataset(
        dataset_path=dataset_path,
        prep_folder=prep_folder,
        input_features=resolved_inputs,
        output_feature=selected_output_features,
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

    X_train, X_test, y_train, y_test = read_prep_data(
        inputs=resolved_inputs,
        prep_folder=prep_folder,
        optional_future_quality_inputs=optional_future_quality_inputs,
    )
    _ = (X_test, y_test)
    return X_train, y_train.columns.tolist(), use_auto_feature_selection, reused_prep_data

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



def print_ann_training_robustness_notes():
    """Explain the safeguards used for neural-network training."""
    print("\nANN industrial robustness safeguards:")
    print(
        f"- learning_rate={learning_rate}: conservative AdamW step size reduces "
        "overshoot and run-to-run volatility on standardized tabular data; "
        f"ReduceLROnPlateau halves it after {lr_plateau_patience} flat checks "
        f"down to {min_learning_rate}."
    )
    print(
        f"- input_learning_rate={input_learning_rate}: input optimization moves "
        "data directly, so a smaller step prevents unrealistic optimized inputs."
    )
    print(
        f"- validation_fraction={validation_fraction}, patience={early_stopping_patience}: "
        "early stopping monitors held-out training data and restores the best weights, "
        "which limits memorization on the ~174-row dataset."
    )
    print(
        f"- ann_weight_decay={ann_weight_decay}: L2 regularization discourages oversized "
        "weights and improves generalization without adding model complexity."
    )
    print(
        f"- max_gradient_norm={max_gradient_norm}, max_input_delta={max_input_delta}: "
        "gradient and input-update bounds prevent unstable convergence and non-physical "
        "input drift."
    )
    print(
        f"- hidden size candidates={list_hidden_sizes}: small candidate networks keep "
        "capacity proportional to a small industrial dataset."
    )

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

# Neural-network normalization uses sklearn StandardScaler exclusively.
# Each scaler is fitted only on the active training partition and then reused
# for validation/test/inference data so held-out statistics never leak into the model.

# Function to apply model on data and generate predictions
# Return predictions, MSE and R-Squared



def as_2d_float_array(data, name):
    """Convert pandas/numpy-like regression data to a finite 2D float array."""
    values = data.values if hasattr(data, "values") else data
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def as_feature_matrix(data, name):
    """Convert tabular feature data to a finite 2D float matrix."""
    arr = as_2d_float_array(data, name)
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column.")
    return arr


def safe_r2_score(y_true, y_pred):
    """Compute finite multi-output R², returning None when undefined."""
    y_true_arr = as_2d_float_array(y_true, "y_true")
    y_pred_arr = as_2d_float_array(y_pred, "y_pred")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have matching shapes; got "
            f"{y_true_arr.shape} and {y_pred_arr.shape}."
        )
    if y_true_arr.shape[0] < 2:
        return None
    score = r2_score(y_true_arr, y_pred_arr, multioutput="uniform_average")
    return float(score) if np.isfinite(score) else None


def initialize_linear_weights(module):
    """Initialize Linear layers with activation-aware fan-in scaling."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_torch_model(model_class, input_size, hidden_sizes, output_size):
    """Instantiate a torch model while preserving older constructors."""
    try:
        return model_class(input_size, hidden_sizes, output_size=output_size)
    except TypeError:
        return model_class(input_size, hidden_sizes)


def validate_hidden_size_compatibility(model, hidden_sizes):
    """Check whether the requested hidden-size layout matches a model class."""
    hidden_layers = getattr(model, "hidden_layers", None)
    return hidden_layers is None or len(hidden_sizes) == len(hidden_layers)


def load_model_state_dict(model, checkpoint_path):
    """Safely restore model weights from a checkpoint path when compatible."""
    if not checkpoint_path:
        return False
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}. Training from initialized weights.")
        return False
    try:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except (OSError, RuntimeError, ValueError) as err:
        print(f"Could not load checkpoint '{checkpoint_path}' ({err}). Training from initialized weights.")
        return False
    print(f"Loaded pretrained weights from: {checkpoint_path}")
    return True


def parse_optimization_scope(answer):
    """Return training optimization scope: weights, inputs, or both."""
    normalized = (answer or "").strip().lower()
    mapping = {
        "": "weights",
        "w": "weights",
        "weight": "weights",
        "weights": "weights",
        "i": "inputs",
        "input": "inputs",
        "inputs": "inputs",
        "b": "both",
        "both": "both",
        "wi": "both",
        "iw": "both",
    }
    if normalized not in mapping:
        raise ValueError("Invalid optimization scope. Use weights, inputs, or both.")
    return mapping[normalized]


def prompt_optimized_input_features(input_features):
    print("\nInput-optimization target features:")
    print_selection_guide()
    print("\n".join([str(i) + ")" + name for i, name in enumerate(input_features, start=1)]))
    answer = input(
        "Select input features to optimize (indexes/ranges) or [all] [all]: "
    ).strip()
    if not answer or answer.lower() == "all":
        return None
    try:
        selected = parse_multi_select(answer, input_features, allow_all=True, one_based=True)
    except ValueError as err:
        print(f"Invalid optimized-input feature selection: {err}")
        exit()
    if selected is None:
        return None
    return [input_features.index(name) for name in selected]


def fit_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    num_epochs,
    display_steps=False,
    run=0,
    optimization_scope="weights",
    optimize_feature_indexes=None,
    pretrained_state_dict_path=None,
):
    """Train one ANN on tabular data with train-only normalization.

    Returns ``(predictions_original_scale, mse_original_scale, r2, model,
    optimized_inputs_report)`` for every success/failure path so callers can
    safely unpack results.  Scalers are fitted only on the training subset used
    for optimization and are then reused for validation/test data.
    """
    set_model_seed(model_seed + run)
    try:
        X_train_values = as_feature_matrix(X_train, "X_train")
        X_test_values = as_feature_matrix(X_test, "X_test")
        y_train_values = as_2d_float_array(y_train, "y_train")
        y_test_values = as_2d_float_array(y_test, "y_test")
    except ValueError as err:
        print(f"Invalid training data: {err}")
        return None, None, None, model, None

    if X_train_values.shape[0] != y_train_values.shape[0]:
        print("X_train and y_train row counts do not match.")
        return None, None, None, model, None
    if X_test_values.shape[0] != y_test_values.shape[0]:
        print("X_test and y_test row counts do not match.")
        return None, None, None, model, None

    optimize_weights = optimization_scope in ("weights", "both")
    optimize_inputs = optimization_scope in ("inputs", "both")
    if not optimize_weights and not optimize_inputs:
        print(f"Invalid optimization scope: {optimization_scope}")
        return None, None, None, model, None

    if optimize_feature_indexes is not None:
        optimize_feature_indexes = [int(idx) for idx in optimize_feature_indexes]
        invalid_indexes = [idx for idx in optimize_feature_indexes if idx < 0 or idx >= X_train_values.shape[1]]
        if invalid_indexes:
            print(f"Invalid optimized-input feature indexes: {invalid_indexes}")
            return None, None, None, model, None

    use_validation = len(X_train_values) >= 10 and validation_fraction > 0
    if use_validation:
        x_fit, x_val, y_fit, y_val = train_test_split(
            X_train_values,
            y_train_values,
            test_size=validation_fraction,
            random_state=data_seed + run,
            shuffle=True,
        )
    else:
        x_fit, y_fit = X_train_values, y_train_values
        x_val, y_val = None, None

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    try:
        X_fit_normalized = torch.as_tensor(x_scaler.fit_transform(x_fit), dtype=torch.float32)
        X_val_normalized = (
            torch.as_tensor(x_scaler.transform(x_val), dtype=torch.float32)
            if x_val is not None
            else None
        )
        X_test_normalized = torch.as_tensor(x_scaler.transform(X_test_values), dtype=torch.float32)
        y_fit_normalized = torch.as_tensor(y_scaler.fit_transform(y_fit), dtype=torch.float32)
        y_val_normalized = (
            torch.as_tensor(y_scaler.transform(y_val), dtype=torch.float32)
            if y_val is not None
            else None
        )
    except ValueError as err:
        print(f"Could not normalize data safely: {err}")
        return None, None, None, model, None

    model.input_scaler_ = x_scaler
    model.target_scaler_ = y_scaler

    model.apply(initialize_linear_weights)
    load_model_state_dict(model, pretrained_state_dict_path)

    criterion = nn.MSELoss()
    optimizer = (
        optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=ann_weight_decay, eps=1e-8)
        if optimize_weights
        else None
    )
    original_train_inputs = X_fit_normalized.clone().detach()
    train_inputs = X_fit_normalized.clone().detach().requires_grad_(optimize_inputs)
    input_optimizer = optim.Adam([train_inputs], lr=input_learning_rate, eps=1e-8) if optimize_inputs else None
    weight_scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            min_lr=min_learning_rate,
        )
        if optimizer is not None
        else None
    )

    best_monitor_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    best_train_inputs = train_inputs.detach().clone()
    epochs_without_improvement = 0

    for epoch in range(int(num_epochs)):
        model.train()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if input_optimizer is not None:
            input_optimizer.zero_grad(set_to_none=True)

        outputs = model(train_inputs)
        if outputs.shape != y_fit_normalized.shape:
            print(
                f"Model output shape {tuple(outputs.shape)} does not match "
                f"target shape {tuple(y_fit_normalized.shape)}."
            )
            return None, None, None, model, None
        if not torch.isfinite(outputs).all():
            print(f"Non-finite outputs detected at epoch {epoch + 1}")
            return None, None, None, model, None

        loss = criterion(outputs, y_fit_normalized)
        if not torch.isfinite(loss):
            print(f"Non-finite loss detected at epoch {epoch + 1}")
            return None, None, None, model, None

        loss.backward()

        if optimize_inputs and optimize_feature_indexes is not None and train_inputs.grad is not None:
            mask = torch.zeros_like(train_inputs.grad)
            mask[:, optimize_feature_indexes] = 1.0
            train_inputs.grad.mul_(mask)

        if optimize_weights:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
            optimizer.step()

        if optimize_inputs:
            torch.nn.utils.clip_grad_norm_([train_inputs], max_norm=max_gradient_norm)
            input_optimizer.step()
            with torch.no_grad():
                lower_bound = original_train_inputs - max_input_delta
                upper_bound = original_train_inputs + max_input_delta
                train_inputs.clamp_(min=lower_bound, max=upper_bound)

        model.eval()
        with torch.no_grad():
            if X_val_normalized is not None:
                monitor_outputs = model(X_val_normalized)
                monitor_loss = criterion(monitor_outputs, y_val_normalized).item()
            else:
                monitor_loss = loss.item()

        if not np.isfinite(monitor_loss):
            print(f"Non-finite validation loss detected at epoch {epoch + 1}")
            return None, None, None, model, None

        if weight_scheduler is not None:
            weight_scheduler.step(monitor_loss)

        if monitor_loss < best_monitor_loss - early_stopping_min_delta:
            best_monitor_loss = monitor_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            best_train_inputs = train_inputs.detach().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 and display_steps:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {loss.item():.6f}, Monitor Loss: {monitor_loss:.6f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            if display_steps:
                print(
                    f"Early stopping at epoch {epoch + 1}; "
                    f"best epoch {best_epoch} with monitor loss {best_monitor_loss:.6f}"
                )
            break

    if best_epoch == 0:
        print("Training did not complete a finite epoch.")
        return None, None, None, model, None

    model.load_state_dict(best_state_dict)
    train_inputs = best_train_inputs
    model.best_epoch_ = best_epoch
    model.best_monitor_loss_ = best_monitor_loss

    model.eval()
    with torch.no_grad():
        predictions_norm = model(X_test_normalized)
        if not torch.isfinite(predictions_norm).all():
            print("Non-finite predictions detected")
            return None, None, None, model, None

    try:
        predictions_np = y_scaler.inverse_transform(predictions_norm.detach().cpu().numpy())
        predictions_np = as_2d_float_array(predictions_np, "predictions")
        mse = float(np.mean((predictions_np - y_test_values) ** 2))
        r2 = safe_r2_score(y_test_values, predictions_np)
    except ValueError as err:
        print(f"Could not compute prediction metrics: {err}")
        return None, None, None, model, None

    optimized_inputs_report = None
    if optimize_inputs:
        with torch.no_grad():
            delta = train_inputs.detach() - original_train_inputs
            optimized_inputs_report = {
                "original_inputs": original_train_inputs.detach().cpu().numpy(),
                "optimized_inputs": train_inputs.detach().cpu().numpy(),
                "delta_inputs": delta.cpu().numpy(),
                "mean_abs_delta_per_feature": np.mean(np.abs(delta.cpu().numpy()), axis=0),
            }
    return predictions_np, mse, r2, model, optimized_inputs_report


def fit_sklearn_model(model, X_train, X_test, y_train, y_test):
    """Fit a scikit-learn regressor with train-only feature scaling."""
    X_train_values = as_feature_matrix(X_train, "X_train")
    X_test_values = as_feature_matrix(X_test, "X_test")
    y_train_values = as_2d_float_array(y_train, "y_train")
    y_test_values = as_2d_float_array(y_test, "y_test")

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_values)
    X_test_scaled = scaler_x.transform(X_test_values)

    is_multi_output = y_train_values.shape[1] > 1
    fit_estimator = model
    y_train_fit = y_train_values if is_multi_output else y_train_values.ravel()

    if is_multi_output:
        model_name = getattr(model, "__class__", type(model)).__name__
        supports_multioutput = False
        if hasattr(model, "_get_tags"):
            supports_multioutput = bool(model._get_tags().get("multioutput", False))
        if not supports_multioutput:
            print(
                f"{model_name} does not natively support multi-output regression. "
                "Wrapping it with MultiOutputRegressor."
            )
            fit_estimator = MultiOutputRegressor(model)

    fit_estimator.fit(X_train_scaled, y_train_fit)
    predictions = fit_estimator.predict(X_test_scaled)
    predictions_2d = as_2d_float_array(predictions, "predictions")

    if predictions_2d.shape != y_test_values.shape:
        raise ValueError(
            f"Prediction shape {predictions_2d.shape} does not match target shape {y_test_values.shape}."
        )

    fit_estimator.feature_scaler_ = scaler_x
    mse = float(np.mean((predictions_2d - y_test_values) ** 2))
    r2 = safe_r2_score(y_test_values, predictions_2d)
    return predictions_2d, mse, r2, fit_estimator


def is_torch_model(model_class):
    return isinstance(model_class, type) and issubclass(model_class, nn.Module)


def select_epochs_with_cv_early_stopping(
    model_class,
    hidden_sizes,
    features=None,
    folds=5,
    patience=early_stopping_patience,
    min_delta=early_stopping_min_delta,
):
    """Recommend an epoch count using leakage-safe K-Fold early stopping."""
    X_train, _, y_train, _ = read_prep_data(features)
    full_X = X_train.reset_index(drop=True)
    full_y = y_train.reset_index(drop=True)
    if len(full_X) < 2:
        return None

    n_splits = min(int(folds), len(full_X))
    if n_splits < 2:
        return None

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=data_seed)
    best_epochs = []
    input_size = full_X.shape[1]
    output_size = as_2d_float_array(full_y, "full_y").shape[1]
    max_epochs = max(list_epochs) if list_epochs else 200

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(full_X), start=1):
        x_train_fold = full_X.iloc[train_idx]
        x_val_fold = full_X.iloc[val_idx]
        y_train_fold = full_y.iloc[train_idx]
        y_val_fold = full_y.iloc[val_idx]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        try:
            x_train_tensor = torch.as_tensor(x_scaler.fit_transform(x_train_fold), dtype=torch.float32)
            x_val_tensor = torch.as_tensor(x_scaler.transform(x_val_fold), dtype=torch.float32)
            y_train_values = as_2d_float_array(y_train_fold, "y_train_fold")
            y_val_values = as_2d_float_array(y_val_fold, "y_val_fold")
            y_train_tensor = torch.as_tensor(y_scaler.fit_transform(y_train_values), dtype=torch.float32)
            y_val_tensor = torch.as_tensor(y_scaler.transform(y_val_values), dtype=torch.float32)
            model = make_torch_model(model_class, input_size, hidden_sizes, output_size)
        except (TypeError, ValueError) as err:
            print(f"CV fold {fold_index} skipped: {err}")
            continue

        if not validate_hidden_size_compatibility(model, hidden_sizes):
            return None

        set_model_seed(model_seed + fold_index)
        model.apply(initialize_linear_weights)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=ann_weight_decay, eps=1e-8)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            min_lr=min_learning_rate,
        )

        best_val_loss = float("inf")
        best_epoch = None
        epochs_without_improvement = 0

        for epoch in range(1, int(max_epochs) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_outputs = model(x_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            if not torch.isfinite(train_loss):
                break
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if not np.isfinite(val_loss):
                break
            scheduler.step(val_loss)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        if best_epoch is not None:
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
    model = make_torch_model(model_class, input_size, hidden_sizes, output_size)

    _, _, r2, model, _ = fit_model(
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

    scaler = getattr(model, "input_scaler_", None)
    if scaler is None:
        scaler = StandardScaler().fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

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
    model = make_torch_model(model_class, input_size, hidden_sizes, output_size)
    _, _, r2, model, _ = fit_model(model, X_train, X_test, y_train, y_test, num_epochs)
    print("R2:", r2)

    weight_importances = {}
    with torch.no_grad():
        for i, feature in enumerate(inputs):
            importance = np.abs(model.hidden_layers[0].weight[:, i].numpy()).mean()
            weight_importances[feature] = importance

    weight_table = pd.DataFrame(list(weight_importances.items()), columns=['Feature', 'Weight Importance'])
    return weight_table


def jackknife_sensitivity_analysis(model_class, data, inputs, output, num_epochs, hidden_sizes):
    base_model_r2, _, _, _, _, _, run, _ = repeat_fit_model(model_class,
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
            reduced_r2, _, _, _, _, _, _run, _ = repeat_fit_model(
                model_class, 1, num_epochs, hidden_sizes, features=reduced_inputs
            )
            if reduced_r2 is not None:
                reduced_r2_list.append(reduced_r2)
        if not reduced_r2_list or base_model_r2 is None:
            continue
        reduced_r2_mean = np.mean(reduced_r2_list)
        sensitivity = base_model_r2 - reduced_r2_mean
        variance = np.var(reduced_r2_list)
        sensitivities[input_feature] = sensitivity
        variances.append(variance)

    sensitivity_table = pd.DataFrame(list(sensitivities.items()), columns=['Feature', 'Sensitivity'])
    sensitivity_table['Variance'] = variances
    return sensitivity_table

def backward_feature_elimination(model_class, data, inputs, output, num_epochs, hidden_sizes):
    mean_r2, _, _, _, _, _, _run, _ = repeat_fit_model(model_class,
            num_repeats, num_epochs, hidden_sizes,features=inputs)
    if mean_r2 is None:
        print("Backward elimination skipped because the baseline model did not train successfully.")
        return pd.DataFrame(columns=["features", "R2"])
    best_r2 = mean_r2
    print("Using all features")
    print("Features:", inputs)
    print("R2:", mean_r2)
    candidates = list(inputs)
    rows = [] # Rows of a table to show the features and the result
    rows.append({"features": ", ".join(inputs), "R2": round(mean_r2,2)})
    while(True):
        results = {}
        for feature in candidates:
            # Selet other features except for current feature
            features = [f for f in candidates if f != feature]
            mean_r2, _, _, _, _, _, _run, _ = repeat_fit_model(model_class, num_repeats, num_epochs, hidden_sizes, features=features)
            if mean_r2 is None:
                continue
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
        for feature in [feature for feature in inputs if feature not in candidates]:
            if len(candidates) == 0:
                features = [feature]
            else:
                features = [feature] + candidates

            mean_r2, _, _, _, _, _, _run, _ = repeat_fit_model(model_class, num_repeats, num_epochs, hidden_sizes, features=features)
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
def repeat_fit_model(
    model_class,
    num_repeats,
    num_epochs,
    hidden_sizes,
    display_steps=False,
    features=None,
    optimization_scope="weights",
    optimize_feature_indexes=None,
    pretrained_state_dict_path=None,
):
    """Run repeated training/evaluation and summarize successful runs.

    Returns eight values in all cases: mean R² (%), std R² (%), mean MSE,
    best predictions, best R² (%), per-run R² list (%), best run index, and
    the optimized-input report from the best run.
    """
    X_train, X_test, y_train, y_test = read_prep_data(features)
    r2_list = []
    mse_list = []
    max_r2 = -float("inf")
    max_run = 0
    best_preds = None
    best_optimized_inputs_report = None
    input_size = X_train.shape[1]
    output_size = as_2d_float_array(y_train, "y_train").shape[1]

    for i in range(int(num_repeats)):
        try:
            if is_torch_model(model_class):
                model = make_torch_model(model_class, input_size, hidden_sizes, output_size)
                if not validate_hidden_size_compatibility(model, hidden_sizes):
                    print(f"Skipping incompatible hidden sizes {hidden_sizes} for {model_class.__name__}.")
                    continue
                predictions, mse, r2, model, optimized_inputs_report = fit_model(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    num_epochs,
                    display_steps=display_steps,
                    run=i,
                    optimization_scope=optimization_scope,
                    optimize_feature_indexes=optimize_feature_indexes,
                    pretrained_state_dict_path=pretrained_state_dict_path,
                )
            else:
                model = model_class(model_seed + i)
                predictions, mse, r2, model = fit_sklearn_model(model, X_train, X_test, y_train, y_test)
                optimized_inputs_report = None
        except (TypeError, ValueError, RuntimeError) as err:
            print(f"Run {i + 1} failed: {err}")
            continue

        if r2 is None or mse is None or predictions is None:
            continue

        if r2 > max_r2:
            max_r2 = r2
            max_run = i
            best_preds = predictions
            best_optimized_inputs_report = optimized_inputs_report

        r2_list.append(float(r2) * 100.0)
        mse_list.append(float(mse))

    if display_steps:
        print(r2_list)

    if not r2_list:
        return None, None, None, None, None, [], max_run, None

    mean_r2 = float(np.mean(r2_list))
    mean_mse = float(np.mean(mse_list))
    std_r2 = float(np.std(r2_list))
    return (
        mean_r2,
        std_r2,
        mean_mse,
        best_preds,
        float(max_r2) * 100.0,
        r2_list,
        max_run,
        best_optimized_inputs_report,
    )


############################### Start of Program ###################
dataset_path = "convert/sugar_all.csv"
selected_saved_run, loaded_optimization_scope = prompt_saved_run_choice()
prepared_X_train, prepared_outputs, use_auto_feature_selection, reused_prep_data = prepare_or_reuse_data(
    dataset_path=dataset_path,
    prep_folder="prep_data",
)

saved_inputs = selected_saved_run.get("inputs") if selected_saved_run else None
saved_outputs = selected_saved_run.get("outputs") if selected_saved_run else None
run_inputs = saved_inputs if saved_inputs else prepared_X_train.columns.tolist()
run_optional_future_quality_inputs = [
    feature for feature in run_inputs
    if feature in FUTURE_QUALITY_INPUT_CANDIDATES
]
X_train, X_test, y_train, y_test = read_prep_data(
    inputs=run_inputs,
    prep_folder="prep_data",
    optional_future_quality_inputs=run_optional_future_quality_inputs,
)

# After loading, get the column names from X_train
inputs = X_train.columns.tolist()
if saved_outputs:
    missing_saved_outputs = [col for col in saved_outputs if col not in y_train.columns.tolist()]
    if missing_saved_outputs:
        print(
            "Saved run output columns are missing from prep_data: "
            f"{missing_saved_outputs}. Please rebuild prep_data or choose a different saved run."
        )
        exit()
    y_train = y_train[saved_outputs]
    y_test = y_test[saved_outputs]
outputs = y_train.columns.tolist()
output = outputs

data = X_train
print_numbered_feature_list("Selected Input Features", inputs, ANSI_GREEN)
print_numbered_feature_list("Selected Output Features", outputs, ANSI_BLUE)
active_features = list(inputs)
if selected_saved_run:
    print("Loaded saved run metadata; reusing prepared inputs/outputs.")
    loaded_hidden_size_groups = normalize_hidden_size_groups(
        selected_saved_run.get("hidden_size_groups"),
        list_hidden_sizes,
    )
    loaded_epoch_candidates = [
        int(e)
        for e in selected_saved_run.get("epoch_candidates", [])
        if str(e).isdigit() and int(e) > 0
    ]
    loaded_repeat_count = selected_saved_run.get("repeat_count")
    if isinstance(loaded_repeat_count, str) and loaded_repeat_count.isdigit():
        loaded_repeat_count = int(loaded_repeat_count)
    if not isinstance(loaded_repeat_count, int) or loaded_repeat_count < 1:
        loaded_repeat_count = None
else:
    loaded_hidden_size_groups = [list(group) for group in list_hidden_sizes]
    loaded_epoch_candidates = []
    loaded_repeat_count = None
    ans = input("Confirm these numbered inputs/outputs from prep_data folder [y]: ").strip().lower()
    if ans not in ("", "y", "yes"):
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
if selected_saved_run and selected_saved_run.get("model_name") in model_names:
    selected_models = [model_names.index(selected_saved_run["model_name"])]
    print(f"Using saved model: {selected_saved_run['model_name']}")
else:
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
if has_nn_model:
    print_ann_training_robustness_notes()

default_epochs = list(list_epochs)
if loaded_epoch_candidates:
    default_epochs = loaded_epoch_candidates
answer = " ".join([str(e) for e in default_epochs])
optimization_scope_answer = ask_with_default(
    "Optimization scope for NN training [weights|inputs|both]",
    loaded_optimization_scope if loaded_optimization_scope else "weights",
)
try:
    optimization_scope = parse_optimization_scope(optimization_scope_answer)
except ValueError as err:
    print(err)
    exit()
if optimization_scope in ("inputs", "both") and not has_nn_model:
    print(
        "Input optimization is only available for neural-network models. "
        "Selected non-NN models will train/evaluate weights only."
    )
optimize_feature_indexes = None
if optimization_scope in ("inputs", "both"):
    optimize_feature_indexes = prompt_optimized_input_features(active_features)
    if optimize_feature_indexes is None:
        print("Input optimization target: all selected input features.")
    else:
        selected_names = [active_features[idx] for idx in optimize_feature_indexes]
        print("Input optimization target features:", selected_names)
change_training_options = False
if selected_saved_run:
    change_training_options = input(
        "Change training options (epochs/hidden sizes/repeats)? [n]: "
    ).strip().lower() in ("y", "yes")

if not selected_saved_run or change_training_options:
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

    if has_nn_model and (not selected_saved_run or change_training_options):
        answer = ask_with_default(
            "Enter hidden sizes (groups split by '#', e.g. '10 5 # 15 10 3')",
            " # ".join([" ".join([str(v) for v in hs]) for hs in loaded_hidden_size_groups]),
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
    elif has_nn_model:
        list_hidden_sizes = [list(group) for group in loaded_hidden_size_groups]
        use_cv_early_stop = False
    else:
        list_hidden_sizes = [[]]
        use_cv_early_stop = False

    if not selected_saved_run or change_training_options:
        repeat_default = loaded_repeat_count if loaded_repeat_count is not None else num_repeats
        answer = ask_with_default("Enter the number of repeating predictions", repeat_default)
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
    best_optimized_inputs_report = None
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
                pretrained_weights_path = (
                    selected_saved_run.get("weights_path")
                    if selected_saved_run and is_nn and selected_saved_run.get("has_weights")
                    else None
                )
                # Apply model on data for N repeats and get predictions, mse and r2
                mean_r2, std_r2, mean_mse, model_best_preds, max_r2, r2_list, max_run, optimized_inputs_report = repeat_fit_model(
                    model_class,
                    num_repeats, num_epochs, hidden_sizes, display_steps=True, features=active_features, optimization_scope=optimization_scope, optimize_feature_indexes=optimize_feature_indexes, pretrained_state_dict_path=pretrained_weights_path)

                # Keep best seed to generate the same predictions later
                if mean_r2 is None:
                    continue

                if max_r2 > best_r2:
                    best_r2 = max_r2
                    model_best_predictions[model_name] = model_best_preds
                    best_optimized_inputs_report = optimized_inputs_report
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
    best_predictions = model_best_predictions.get(max_model_name)
    if best_predictions is None:
        print("Best model did not produce predictions. Skipping reporting/plots.")
        exit()
    output_title = ", ".join(outputs)
    title = "Prediction of " + output_title + " with " + max_model_name + " epochs:" + str(max_epochs)
    file_name = f"R2-{best_r2:.2f}-" + max_model_name + "-" + "-".join(outputs) + ".png"

    y_test_values = y_test.values if hasattr(y_test, "values") else y_test
    best_model_metrics = compute_regression_report_metrics(y_test_values, best_predictions)
    if best_model_metrics:
        print("===================== Best Selected Model Report =====================")
        print(f"Selected model name                    {max_model_name}")
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

    if best_optimized_inputs_report is not None:
        print("\n================= Optimized Input Report =================")
        selected_names = (
            active_features
            if optimize_feature_indexes is None
            else [active_features[idx] for idx in optimize_feature_indexes]
        )
        mean_abs_delta = best_optimized_inputs_report["mean_abs_delta_per_feature"]
        report_df = pd.DataFrame(
            {
                "feature": active_features,
                "mean_abs_delta_normalized": mean_abs_delta,
            }
        ).sort_values(by="mean_abs_delta_normalized", ascending=False)
        report_df["optimized"] = report_df["feature"].isin(selected_names)
        report_path = os.path.join("tables", "optimized-inputs-summary.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Saved summary: {report_path}")
        print(report_df)

        original_df = pd.DataFrame(
            best_optimized_inputs_report["original_inputs"], columns=active_features
        )
        optimized_df = pd.DataFrame(
            best_optimized_inputs_report["optimized_inputs"], columns=active_features
        )
        original_df.to_csv(os.path.join("tables", "optimized-inputs-before.csv"), index=False)
        optimized_df.to_csv(os.path.join("tables", "optimized-inputs-after.csv"), index=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(report_df["feature"], report_df["mean_abs_delta_normalized"])
        ax.set_title("Mean absolute input change after input optimization")
        ax.set_xlabel("Input feature")
        ax.set_ylabel("Mean abs change (normalized)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join("plots", "optimized-inputs-change-summary.png"), format="png")
        plt.close(fig)
        print("Saved plot: plots/optimized-inputs-change-summary.png")
        print("==========================================================")

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
    weights_path = None
    if is_torch_model(models[max_model_index]):
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        checkpoint_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{max_model_name}_R2-{best_r2:.2f}_run-{best_run}.pt")
        weights_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
        input_size = X_train.shape[1]
        output_size = as_2d_float_array(y_train, "y_train").shape[1]
        best_model_instance = make_torch_model(models[max_model_index], input_size, max_hidden_sizes, output_size)
        checkpoint_predictions, checkpoint_mse, checkpoint_r2, best_model_instance, _ = fit_model(
            best_model_instance,
            X_train,
            X_test,
            y_train,
            y_test,
            max_epochs,
            display_steps=False,
            run=best_run,
            optimization_scope="weights",
            optimize_feature_indexes=None,
            pretrained_state_dict_path=None,
        )
        checkpoint_ready = (
            checkpoint_r2 is not None
            and checkpoint_predictions is not None
            and getattr(best_model_instance, "input_scaler_", None) is not None
            and getattr(best_model_instance, "target_scaler_", None) is not None
        )
        if checkpoint_ready:
            torch.save(best_model_instance.state_dict(), weights_path)
            print(f"Saved model checkpoint: {weights_path}")

            try:
                recommendation = recommend_operating_conditions(
                    current_conditions=X_test.iloc[0],
                    trained_model=best_model_instance,
                    input_features=active_features,
                    output_features=outputs,
                    historical_inputs=X_train,
                    historical_targets=y_train,
                )
                recommendation_path = os.path.join("tables", "recommended-operating-conditions.json")
                with open(recommendation_path, "w", encoding="utf-8") as fp:
                    json.dump(recommendation, fp, indent=2, ensure_ascii=False, default=float)
                print_industrial_operator_demo(
                    recommendation=recommendation,
                    input_features=active_features,
                    output_features=outputs,
                )
                print(f"Saved recommendation: {recommendation_path}")
            except (RecommendationError, ValueError, KeyError, TypeError) as err:
                print(f"Recommendation engine skipped: {err}")
        else:
            print("Checkpoint save skipped because the best model could not be retrained safely.")
            weights_path = None

    save_run_summary(
        model_name=max_model_name,
        inputs=active_features,
        outputs=outputs,
        best_r2=best_r2,
        epoch_candidates=list_epochs,
        hidden_size_groups=list_hidden_sizes if is_torch_model(models[max_model_index]) else [],
        repeat_count=num_repeats,
        optimization_scope=optimization_scope,
        weights_path=weights_path,
    )
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
