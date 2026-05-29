"""Prescriptive industrial AI recommendation engine for refinery operation.

This module turns a trained predictive quality model into a prescriptive search
engine.  The main entry point, :func:`recommend_operating_conditions`, receives
the current refinery state, varies only approved controllable variables, uses the
trained model to predict future quality, and returns the feasible setting with
the lowest color/risk objective.
"""

from __future__ import annotations

from itertools import product
from math import isfinite
from statistics import mean, pstdev

try:
    from refinery_variables import CONTROL_VARIABLES, validate_model_inputs
except ImportError:  # Allows package-style imports when NN is imported as a package.
    from .refinery_variables import CONTROL_VARIABLES, validate_model_inputs


# Industrially conservative fallback envelopes.  When historical data is passed,
# these are tightened to the observed operating window before searching.
DEFAULT_CONTROL_RANGES = {
    "co2_percent": (0.5, 8.0),
    "carbonated_pH": (7.0, 11.5),
    "lime_alkalinity": (0.01, 0.25),
    "carbonated_alkalinity": (0.01, 0.35),
}

# Only the requested knobs are optimized by default.  carbonated_alkalinity stays
# available for callers that explicitly include it in control_variables.
DEFAULT_RECOMMENDED_CONTROLS = (
    "co2_percent",
    "carbonated_pH",
    "lime_alkalinity",
)

# Prefer outputs that are direct color measurements, then the aggregate quality
# score used by this project.  Callers can override this with target_weights.
DEFAULT_COLOR_TARGET_PRIORITY = (
    "standard_liquor_color",
    "sulphited_color",
    "white_total_points",
)


class RecommendationError(ValueError):
    """Raised when a recommendation cannot be produced safely."""


def _as_mapping(row):
    """Convert dict/Series/one-row DataFrame-like inputs to a plain mapping."""
    if hasattr(row, "to_dict"):
        converted = row.to_dict()
        # pandas one-row DataFrame returns {col: {index: value}}; flatten it.
        if converted and all(isinstance(value, dict) for value in converted.values()):
            return {key: next(iter(value.values())) for key, value in converted.items() if value}
        return converted
    return dict(row)


def _is_number(value):
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _numeric_values(values):
    return [float(value) for value in values if _is_number(value)]


def _column_values(table, column):
    """Read a numeric column from pandas/numpy/list-of-dict/list-of-list data."""
    if table is None:
        return []
    if hasattr(table, "columns") and column in getattr(table, "columns"):
        try:
            return _numeric_values(table[column].tolist())
        except AttributeError:
            return _numeric_values(table[column])
    if isinstance(table, list):
        values = []
        for row in table:
            if isinstance(row, dict) and column in row:
                values.append(row[column])
        return _numeric_values(values)
    return []


def _percentile(sorted_values, percentile):
    """Small dependency-free percentile helper for industrial range estimates."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _historical_range(values, fallback_range):
    """Return a robust 5th-to-95th percentile range clipped to fallback bounds."""
    clean_values = sorted(_numeric_values(values))
    if len(clean_values) < 5:
        return fallback_range

    low = _percentile(clean_values, 0.05)
    high = _percentile(clean_values, 0.95)
    if low is None or high is None or low >= high:
        return fallback_range

    fallback_low, fallback_high = fallback_range
    return max(float(low), fallback_low), min(float(high), fallback_high)


def _build_control_ranges(control_variables, historical_data=None, control_ranges=None):
    """Build safe search ranges from user overrides, history, and fallbacks."""
    resolved = {}
    for variable in control_variables:
        fallback = DEFAULT_CONTROL_RANGES.get(variable)
        if fallback is None:
            raise RecommendationError(
                f"No default industrial range is known for control variable '{variable}'. "
                "Pass control_ranges explicitly for this variable."
            )

        if control_ranges and variable in control_ranges:
            low, high = control_ranges[variable]
            low, high = float(low), float(high)
            if low >= high:
                raise RecommendationError(f"Invalid range for {variable}: lower bound must be < upper bound.")
            fallback_low, fallback_high = fallback
            resolved[variable] = (max(low, fallback_low), min(high, fallback_high))
        else:
            resolved[variable] = _historical_range(_column_values(historical_data, variable), fallback)
    return resolved


def _grid_values(low, high, steps):
    if steps < 2:
        return [round((low + high) / 2.0, 6)]
    step = (high - low) / float(steps - 1)
    return [round(low + index * step, 6) for index in range(steps)]


def _model_predict_one(trained_model, feature_row, input_features, output_features):
    """Predict one future quality row with sklearn-like or torch-like models."""
    values = [[float(feature_row[feature]) for feature in input_features]]

    if hasattr(trained_model, "predict"):
        model_input = values
        scaler = getattr(trained_model, "feature_scaler_", None) or getattr(trained_model, "input_scaler_", None)
        if scaler is not None:
            model_input = scaler.transform(values)
        prediction = trained_model.predict(model_input)
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        first = prediction[0] if prediction and isinstance(prediction[0], (list, tuple)) else prediction
        return {name: float(first[index]) for index, name in enumerate(output_features)}

    try:
        import torch
    except ImportError as exc:
        raise RecommendationError(
            "Torch is required to use a trained PyTorch model for recommendations."
        ) from exc

    scaler = getattr(trained_model, "input_scaler_", None)
    if scaler is not None:
        values = scaler.transform(values)
    tensor = torch.tensor(values, dtype=torch.float32)

    trained_model.eval()
    with torch.no_grad():
        prediction = trained_model(tensor)
    prediction_np = prediction.detach().numpy()
    target_scaler = getattr(trained_model, "target_scaler_", None)
    if target_scaler is not None:
        prediction_np = target_scaler.inverse_transform(prediction_np)
    prediction_values = prediction_np.reshape(-1).tolist()
    return {name: float(prediction_values[index]) for index, name in enumerate(output_features)}


def _target_weights(output_features, target_weights=None):
    if target_weights:
        return {name: float(weight) for name, weight in target_weights.items() if name in output_features}

    weights = {}
    for preferred in DEFAULT_COLOR_TARGET_PRIORITY:
        if preferred in output_features:
            weights[preferred] = 1.0
            break
    if not weights:
        # Fallback: optimize the last selected future-quality output because this
        # codebase often stores the final quality/color metric last.
        weights[output_features[-1]] = 1.0
    return weights


def _quality_statistics(output_features, historical_targets=None):
    stats = {}
    for output in output_features:
        values = _column_values(historical_targets, output)
        if values:
            stats[output] = {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "p80": _percentile(sorted(values), 0.80),
            }
    return stats


def _score_prediction(prediction, candidate, current, control_ranges, output_features, weights, target_stats):
    """Lower score is better: future color + high-risk + movement penalties."""
    weighted_quality = 0.0
    weight_total = 0.0
    risk_penalty = 0.0

    for output, weight in weights.items():
        predicted_value = float(prediction[output])
        stats = target_stats.get(output, {})
        scale = stats.get("std") or max(abs(stats.get("mean", predicted_value)), 1.0)
        normalized_value = predicted_value / scale
        weighted_quality += weight * normalized_value
        weight_total += abs(weight)

        risk_threshold = stats.get("p80")
        if risk_threshold is not None and predicted_value > risk_threshold:
            risk_penalty += (predicted_value - risk_threshold) / max(scale, 1.0)

    if weight_total == 0:
        raise RecommendationError("At least one non-zero target weight is required.")

    movement_penalty = 0.0
    for variable, (low, high) in control_ranges.items():
        span = max(high - low, 1e-9)
        current_value = float(current[variable])
        candidate_value = float(candidate[variable])
        movement_penalty += abs(candidate_value - current_value) / span

        # Keep recommendations away from hard edges unless model benefit is high.
        center = (low + high) / 2.0
        half_span = span / 2.0
        movement_penalty += 0.10 * abs(candidate_value - center) / max(half_span, 1e-9)

    return (weighted_quality / weight_total) + 0.50 * risk_penalty + 0.05 * movement_penalty


def recommend_operating_conditions(
    current_conditions,
    trained_model,
    input_features,
    output_features,
    historical_inputs=None,
    historical_targets=None,
    control_variables=DEFAULT_RECOMMENDED_CONTROLS,
    control_ranges=None,
    grid_steps=7,
    target_weights=None,
):
    """Recommend controllable refinery settings using the trained prediction model.

    Parameters
    ----------
    current_conditions:
        Mapping, Series, or one-row DataFrame containing the current refinery
        conditions for all `input_features`.
    trained_model:
        A fitted sklearn-like model with `predict()` or a fitted PyTorch model.
        Existing `feature_scaler_`, `input_scaler_`, and `target_scaler_`
        attributes are honored so the search uses the same preprocessing as
        training.
    input_features / output_features:
        The exact trained model input and future-quality output column names.
    historical_inputs / historical_targets:
        Optional training data used to tighten ranges and define risk thresholds.
    control_variables:
        Controllable variables to search.  Defaults to CO2 percentage, pH, and
        lime alkalinity, matching the requested industrial levers.
    control_ranges:
        Optional explicit ranges, e.g. {"carbonated_pH": (8.8, 10.5)}.
    grid_steps:
        Number of candidate values per control.  Total simulations are
        grid_steps ** number_of_controls.
    target_weights:
        Optional output weights.  If omitted, the engine optimizes the most
        relevant selected future color/quality output.

    Returns
    -------
    dict
        Best recommended settings, predicted future quality, search ranges,
        objective score, and a plain-English explanation of the logic.
    """
    input_features = list(input_features)
    output_features = list(output_features)
    control_variables = tuple(control_variables)

    if not output_features:
        raise RecommendationError("output_features must include at least one future quality target.")
    if grid_steps < 2:
        raise RecommendationError("grid_steps must be at least 2 so each range is actually searched.")

    validate_model_inputs(input_features, output_features=output_features)
    disallowed_controls = [variable for variable in control_variables if variable not in CONTROL_VARIABLES]
    if disallowed_controls:
        raise RecommendationError(f"Only refinery control variables can be optimized: {disallowed_controls}")
    missing_controls = [variable for variable in control_variables if variable not in input_features]
    if missing_controls:
        raise RecommendationError(f"Control variables are not present in the trained input features: {missing_controls}")

    current = _as_mapping(current_conditions)
    missing_inputs = [feature for feature in input_features if feature not in current]
    if missing_inputs:
        raise RecommendationError(f"Current conditions are missing trained input features: {missing_inputs}")
    non_numeric = [feature for feature in input_features if not _is_number(current[feature])]
    if non_numeric:
        raise RecommendationError(f"Recommendation search requires numeric model inputs: {non_numeric}")

    ranges = _build_control_ranges(control_variables, historical_inputs, control_ranges)
    weights = _target_weights(output_features, target_weights)
    target_stats = _quality_statistics(output_features, historical_targets)

    current_prediction = _model_predict_one(trained_model, current, input_features, output_features)
    best = None
    searched_candidates = 0
    variable_grids = [
        _grid_values(ranges[variable][0], ranges[variable][1], grid_steps)
        for variable in control_variables
    ]

    for candidate_values in product(*variable_grids):
        candidate = dict(current)
        for variable, value in zip(control_variables, candidate_values):
            candidate[variable] = value

        prediction = _model_predict_one(trained_model, candidate, input_features, output_features)
        score = _score_prediction(
            prediction,
            candidate,
            current,
            ranges,
            output_features,
            weights,
            target_stats,
        )
        searched_candidates += 1

        if best is None or score < best["objective_score"]:
            best = {
                "recommended_settings": {variable: candidate[variable] for variable in control_variables},
                "predicted_future_quality": prediction,
                "objective_score": float(score),
                "full_candidate_inputs": {feature: candidate[feature] for feature in input_features},
            }

    if best is None:
        raise RecommendationError("No feasible operating candidate was generated.")

    best["current_settings"] = {variable: float(current[variable]) for variable in control_variables}
    best["current_prediction"] = current_prediction
    best["searched_candidates"] = searched_candidates
    best["control_ranges"] = ranges
    best["target_weights"] = weights
    best["logic"] = [
        "Only approved controllable refinery variables are changed; early/raw conditions remain fixed.",
        "Each controllable variable is searched over an industrially realistic range, tightened by historical 5th-95th percentiles when available.",
        "Every candidate is sent through the trained future-quality model using the same scalers saved during training.",
        "The objective minimizes predicted future sugar color/quality output, adds penalty for high-risk predictions above historical P80, and adds a small movement/edge penalty to avoid unrealistic set-point jumps.",
        "The returned recommendation is the feasible candidate with the lowest total objective score.",
    ]
    return best
