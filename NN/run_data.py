"""Data preparation and interactive feature-selection helpers for NN/run.py."""

from run_core import (
    append_future_quality_input_candidates,
    apply_missing_value_pipeline,
    apply_target_value_pipeline,
    clean_train_test_for_modeling,
    coerce_refinery_numeric_frame,
    coerce_refinery_numeric_series,
    optional_input_overrides_for_selection,
    parse_multi_select,
    parse_sequential_layout,
    prepare_or_reuse_data,
    prompt_temporal_options,
    remove_leakage_inputs_for_training,
    save_leakage_report,
    selectable_input_features_from_csv,
    selectable_output_features_from_csv,
    validation_report,
    validation_report_for_training,
)
