"""Industrial refinery variable definitions and leakage checks.

The model is designed for operational prediction: it may only see variables that
are known early in the refinery process plus variables that operators can change.
Future/downstream quality measurements are reserved for targets and are blocked
from the input matrix to avoid target leakage.
"""

import re


# 1) Variables available early in the process before downstream quality is known.
EARLY_VARIABLES = (
    "sheet_name",
    "shift_name",
    "raw_sugar_color",
    "raw_syrup_brix",
    "raw_syrup_color",
)


# 2) Operator-adjustable variables.  Keep this group limited to actionable
# settings/controlled measurements that can be changed before final quality is
# observed (for example CO2, pH, lime/alkalinity controls).
CONTROL_VARIABLES = (
    "lime_alkalinity",
    "co2_percent",
    "carbonated_alkalinity",
    "carbonated_pH",
)


# 3) Future/downstream quality outputs.  These columns must not be used as
# model inputs, even when the user selects "all" or uses automatic feature
# selection, because they are measured after the early/control decision point.
TARGET_VARIABLES = (
    "filtercake_moisture",
    "filtercake_sugar",
    "sweetwater_brix",
    "sulphited_pH",
    "sulphited_brix",
    "sulphited_color",
    "standard_liquor_pH",
    "standard_liquor_brix",
    "standard_liquor_color",
    "white_total_points",
)


_ALLOWED_INPUTS = set(EARLY_VARIABLES) | set(CONTROL_VARIABLES)
_TARGETS = set(TARGET_VARIABLES)
_DERIVED_SEPARATOR = "__"


def _canonical_name(name):
    """Normalize names so pH/PH and punctuation changes do not hide leakage."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


_ALLOWED_CANONICAL = {_canonical_name(name) for name in _ALLOWED_INPUTS}
_TARGET_CANONICAL = {_canonical_name(name) for name in _TARGETS}


def base_variable_name(column_name):
    """Return the source variable for engineered columns such as x__diff_1."""
    return str(column_name).split(_DERIVED_SEPARATOR, 1)[0]


def is_target_variable(column_name, output_features=None):
    """Return True when a column is a selected or known future quality output."""
    output_features = set(output_features or [])
    base_name = base_variable_name(column_name)
    canonical_base = _canonical_name(base_name)
    canonical_column = _canonical_name(column_name)
    return (
        column_name in output_features
        or base_name in output_features
        or canonical_base in _TARGET_CANONICAL
        or canonical_column in _TARGET_CANONICAL
    )


def _optional_future_quality_input_set(optional_future_quality_inputs=None):
    """Normalize opt-in future-quality variables that may be used as inputs."""
    return {_canonical_name(name) for name in (optional_future_quality_inputs or [])}


def is_allowed_model_input(column_name, output_features=None, optional_future_quality_inputs=None):
    """Return True for standard inputs plus explicitly opted-in future-quality inputs."""
    output_features = set(output_features or [])
    base_name = base_variable_name(column_name)
    if column_name in output_features or base_name in output_features:
        return False

    optional_quality_inputs = _optional_future_quality_input_set(optional_future_quality_inputs)
    if _canonical_name(base_name) in optional_quality_inputs:
        return True

    if is_target_variable(column_name, output_features=output_features):
        return False
    return _canonical_name(base_name) in _ALLOWED_CANONICAL


def filter_allowed_model_inputs(columns, output_features=None, optional_future_quality_inputs=None):
    """Keep allowed input columns, preserving the incoming order."""
    return [
        column for column in columns
        if is_allowed_model_input(
            column,
            output_features=output_features,
            optional_future_quality_inputs=optional_future_quality_inputs,
        )
    ]


def find_leakage_columns(input_features, output_features=None, optional_future_quality_inputs=None):
    """Identify selected inputs that would leak future quality information."""
    return [
        column for column in input_features
        if not is_allowed_model_input(
            column,
            output_features=output_features,
            optional_future_quality_inputs=optional_future_quality_inputs,
        )
    ]


def validate_model_inputs(input_features, output_features=None, optional_future_quality_inputs=None):
    """Raise a clear error if an input list contains leaked/disallowed variables."""
    leakage_columns = find_leakage_columns(
        input_features,
        output_features=output_features,
        optional_future_quality_inputs=optional_future_quality_inputs,
    )
    if leakage_columns:
        allowed = list(EARLY_VARIABLES) + list(CONTROL_VARIABLES)
        targets = list(TARGET_VARIABLES)
        raise ValueError(
            "Target-leakage prevention blocked disallowed model inputs: "
            f"{leakage_columns}. Allowed inputs are EARLY_VARIABLES + "
            f"CONTROL_VARIABLES only: {allowed}. Future quality variables are "
            "TARGET_VARIABLES and must be predicted, not used as inputs unless "
            "explicitly selected in run.py's Future Quality Variables prompt "
            f"and not also selected as output targets: {targets}."
        )
    return list(input_features)


def refinery_variable_group_metadata():
    """Return serializable metadata explaining the refinery variable groups."""
    return {
        "EARLY_VARIABLES": list(EARLY_VARIABLES),
        "CONTROL_VARIABLES": list(CONTROL_VARIABLES),
        "TARGET_VARIABLES": list(TARGET_VARIABLES),
        "input_rule": "Model inputs = EARLY_VARIABLES + CONTROL_VARIABLES only.",
        "leakage_rule": "TARGET_VARIABLES and selected outputs are never allowed as inputs.",
    }
