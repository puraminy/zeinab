"""Industrial refinery variable definitions and leakage checks.

The model is designed for operational prediction: it may only see variables that
are known early in the refinery process plus variables that operators can change.
Future/downstream quality measurements are reserved for targets and are blocked
from the input matrix to avoid target leakage.
"""

import re


# 1) Variables available early in the process before downstream quality is known.
EARLY_VARIABLES = (
    "shift_name",
    "year",
    "month",
    "day",
    "day_of_week",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "raw_sugar_color",
    "raw_syrup_brix",
    "raw_syrup_color",
)


# 2) Operator-adjustable variables.  Keep this group limited to actionable
# settings/controlled measurements that can be changed before final quality is
# observed (for example CO2, pH, lime/alkalinity controls).
CONTROL_VARIABLES = (
    "lime_milk_baume",
    "lime_alkalinity",
    "co2_percent",
    "carbonated_alkalinity",
    "carbonated_pH",
    "sulphited_pH",
    "sulphited_brix",
    "standard_liquor_pH",
    "standard_liquor_brix",
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
LEAKAGE_NAME_PATTERNS = (
    "average_to_date",
    "average_two_shifts",
    "moving_average",
    "future",
    "target",
)

# 4) Leakage-prone feature set for the main model.
#
# These final/QC white-sugar fields are derived from quality scoring or final
# quality-control calculations. They are hard-blocked from the main predictive
# model input matrix, even if someone tries to opt them in as future-quality
# inputs, because they would leak post-process quality information.
MAIN_MODEL_EXCLUDED_FEATURE_PATTERNS = (
    "white_total_points",
    "white_quality_",
    "white_average_",
)

# 5) Optional diagnostic-only feature set.
#
# These columns may be used only in a separate diagnostic/comparison model that
# is explicitly labeled as leakage-prone. Keeping this alias separate from the
# main-model exclusion rule makes the policy visible to callers without allowing
# these columns through normal training validation.
DIAGNOSTIC_ONLY_FEATURE_PATTERNS = MAIN_MODEL_EXCLUDED_FEATURE_PATTERNS



# 6) Target-specific feature-importance chronology.
#
# Feature-importance reports must explain a target only with measurements that
# would already exist before that target is measured.  For white_total_points,
# this intentionally stops at evaporation/boiling and excludes any final white
# sugar QC/scoring fields or other contemporaneous/downstream targets.
PRE_TARGET_FEATURE_IMPORTANCE_INPUTS = {
    "white_total_points": (
        # Raw material
        "raw_sugar_color",
        "raw_sugar_purity",
        "raw_sugar_moisture",
        # Stage 2: Clarification
        "lime_alkalinity",
        "carbonated_pH",
        "co2_percent",
        # Stage 3: Sulphitation
        "sulphited_color",
        # Stage 4: Evaporation/Boiling
        "boiler_pH",
    ),
}

# Backward-compatible name used by the existing leakage checks.
LEAKAGE_RISK_FEATURE_PATTERNS = MAIN_MODEL_EXCLUDED_FEATURE_PATTERNS


def _matches_exact_or_prefix(column_name, patterns):
    """Return True when a base column equals an exact rule or starts with a prefix rule."""
    lower_name = str(base_variable_name(column_name)).lower()
    return any(lower_name == pattern or lower_name.startswith(pattern) for pattern in patterns)


def is_main_model_excluded_feature(column_name):
    """Return True for features that must be removed from the main model."""
    return _matches_exact_or_prefix(column_name, MAIN_MODEL_EXCLUDED_FEATURE_PATTERNS)


def is_diagnostic_only_feature(column_name):
    """Return True for optional leakage-prone features reserved for diagnostics."""
    return _matches_exact_or_prefix(column_name, DIAGNOSTIC_ONLY_FEATURE_PATTERNS)


def leakage_pattern_matches(column_name):
    """Return leakage marker substrings/prefixes found in a column name."""
    lower_name = str(base_variable_name(column_name)).lower()
    matches = [pattern for pattern in LEAKAGE_NAME_PATTERNS if pattern in lower_name]
    matches.extend(
        pattern
        for pattern in LEAKAGE_RISK_FEATURE_PATTERNS
        if lower_name == pattern or lower_name.startswith(pattern)
    )
    return matches


def is_name_based_leakage_column(column_name):
    """Return True when a column name contains an automatic leakage marker."""
    return bool(leakage_pattern_matches(column_name))


def find_name_based_leakage_columns(columns, output_features=None, include_outputs=False):
    """Find columns whose names contain configured leakage marker substrings.

    Selected output targets can be kept out of removal lists by leaving
    ``include_outputs`` as False, which preserves final targets while still
    detecting leakage candidates in X.
    """
    output_features = set(output_features or [])
    leakage_columns = []
    for column in columns:
        if not include_outputs and column in output_features:
            continue
        if is_name_based_leakage_column(column):
            leakage_columns.append(column)
    return leakage_columns


def remove_name_based_leakage_inputs(input_features, output_features=None):
    """Remove automatic name-based leakage columns from an X feature list."""
    _ = set(output_features or [])  # Final targets are preserved by callers in y, not in X.
    cleaned_inputs = []
    removed_inputs = []
    for column in input_features:
        if is_name_based_leakage_column(column):
            removed_inputs.append(column)
        else:
            cleaned_inputs.append(column)
    return cleaned_inputs, removed_inputs


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

    if is_name_based_leakage_column(column_name):
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


def feature_importance_inputs_for_target(columns, target):
    """Return target-chronology-safe feature-importance inputs.

    For targets with an explicit chronology policy, only columns listed for that
    target are eligible.  Matching is done against base variable names so future
    engineered variants inherit the same before-target restriction.  Other
    targets keep the existing generic leakage-name filtering behavior.
    """
    allowed_for_target = PRE_TARGET_FEATURE_IMPORTANCE_INPUTS.get(target)
    if not allowed_for_target:
        return [column for column in columns if column != target and not leakage_pattern_matches(column)]

    allowed_canonical = {_canonical_name(name) for name in allowed_for_target}
    return [
        column for column in columns
        if column != target
        and _canonical_name(base_variable_name(column)) in allowed_canonical
        and not leakage_pattern_matches(column)
    ]


def find_leakage_columns(input_features, output_features=None, optional_future_quality_inputs=None):
    """Identify selected inputs that would leak future quality information."""
    leakage_columns = []
    for column in input_features:
        if column in leakage_columns:
            continue
        if is_name_based_leakage_column(column):
            leakage_columns.append(column)
            continue
        if not is_allowed_model_input(
            column,
            output_features=output_features,
            optional_future_quality_inputs=optional_future_quality_inputs,
        ):
            leakage_columns.append(column)
    return leakage_columns


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
            f"and not also selected as output targets: {targets}. Columns containing "
            f"automatic leakage markers are always blocked from X: "
            f"{list(LEAKAGE_NAME_PATTERNS) + list(LEAKAGE_RISK_FEATURE_PATTERNS)}."
        )
    return list(input_features)


def refinery_variable_group_metadata():
    """Return serializable metadata explaining the refinery variable groups."""
    return {
        "EARLY_VARIABLES": list(EARLY_VARIABLES),
        "CONTROL_VARIABLES": list(CONTROL_VARIABLES),
        "TARGET_VARIABLES": list(TARGET_VARIABLES),
        "MAIN_MODEL_EXCLUDED_FEATURE_PATTERNS": list(MAIN_MODEL_EXCLUDED_FEATURE_PATTERNS),
        "DIAGNOSTIC_ONLY_FEATURE_PATTERNS": list(DIAGNOSTIC_ONLY_FEATURE_PATTERNS),
        "PRE_TARGET_FEATURE_IMPORTANCE_INPUTS": {
            target: list(inputs)
            for target, inputs in PRE_TARGET_FEATURE_IMPORTANCE_INPUTS.items()
        },
        "input_rule": "Model inputs = EARLY_VARIABLES + CONTROL_VARIABLES only.",
        "main_model_exclusion_rule": (
            "Remove white_total_points plus every column whose base name starts "
            "with white_quality_ or white_average_ from the main model input set."
        ),
        "diagnostic_only_rule": (
            "The excluded white-sugar scoring/average fields may be used only in "
            "a separate diagnostic comparison model that is clearly labeled as "
            "leakage-prone; they are never allowed through main training validation."
        ),
        "leakage_rule": (
            "TARGET_VARIABLES, selected outputs, and columns containing "
            "average_to_date/average_two_shifts/moving_average/future/target, "
            "or final white-sugar QC fields (white_total_points, white_quality_*, "
            "white_average_*) are never allowed as inputs."
        ),
    }
