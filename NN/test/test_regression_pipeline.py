import unittest
import pandas as pd
import numpy as np
import tempfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_core
from read_data import build_data_quality_report, safe_numeric_conversion, save_data, read_prep_data
from refinery_variables import classify_refinery_variable, is_allowed_model_input


class PipelineRegressionTests(unittest.TestCase):
    def test_training_config_and_epochs_are_initialized(self):
        config = run_core.validate_training_config(run_core.DEFAULT_TRAINING_CONFIG)
        self.assertEqual(config.epoch_candidates, [50, 100, 200, 300])
        state = run_core.initialize_training_run_state(config)
        self.assertEqual(state.epoch_candidates, [50, 100, 200, 300])
        self.assertEqual(state.hidden_size_groups[0], [16])
        epochs, use_cv = run_core.parse_epochs_input("", config.epoch_candidates)
        self.assertEqual(epochs, [50, 100, 200, 300])
        self.assertFalse(use_cv)

    def test_model_registry_and_selection_validation(self):
        models, names = run_core.validate_model_registry([object()], ["DummyModel"])
        self.assertEqual(names, ["DummyModel"])
        self.assertEqual(run_core.validate_selected_model_indexes([0], models), [0])
        with self.assertRaisesRegex(ValueError, "no models were selected"):
            run_core.validate_selected_model_indexes([], models)
        with self.assertRaisesRegex(ValueError, "invalid model index"):
            run_core.validate_selected_model_indexes([3], models)

    def test_output_selection_uses_csv_order_for_indexes(self):
        df = pd.DataFrame(columns=["col1", "filtercake_moisture", "work_period", "target"])
        self.assertEqual(run_core.selectable_output_features_from_csv(df), list(df.columns))
        selected = run_core.parse_multi_select("3", run_core.selectable_output_features_from_csv(df), allow_all=False, one_based=True)
        self.assertEqual(selected, ["work_period"])
        self.assertEqual(
            run_core.validate_output_selection_preserved(["work_period"], ["work_period"]),
            ["work_period"],
        )
        with self.assertRaisesRegex(ValueError, "changed selected output columns"):
            run_core.validate_output_selection_preserved(["white_total_points"], ["work_period"])

    def test_data_quality_report_detects_repeated_headers_and_text_numeric_contamination(self):
        df = pd.DataFrame({"lime_milk_baume": ["12.5", "بومه شیرآهک", "lime_milk_baume"], "co2_percent": ["1", "2", "co2_percent"]})
        report = build_data_quality_report(df, "X_train")
        issues = {(row["column"], row["issue"]) for row in report}
        self.assertIn(("<row>", "repeated_header_row"), issues)
        self.assertIn(("lime_milk_baume", "non_numeric_value"), issues)
        numeric = safe_numeric_conversion(df)
        self.assertTrue(np.isnan(numeric.loc[1, "lime_milk_baume"]))

    def test_read_prep_data_drops_repeated_header_rows_without_breaking_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            X_train = pd.DataFrame({
                "lime_milk_baume": ["10", "lime_milk_baume", "12"],
                "co2_percent": ["1", "co2_percent", "3"],
            })
            X_test = pd.DataFrame({"lime_milk_baume": ["11"], "co2_percent": ["2"]})
            y_train = pd.DataFrame({"white_total_points": ["90", "white_total_points", "95"]})
            y_test = pd.DataFrame({"white_total_points": ["92"]})
            save_data(tmpdir, X_train, X_test, y_train, y_test)

            X_train_loaded, _, y_train_loaded, _ = read_prep_data(
                inputs=["lime_milk_baume", "co2_percent"],
                prep_folder=tmpdir,
                optional_future_quality_inputs=["lime_milk_baume", "co2_percent"],
            )

        self.assertEqual(len(X_train_loaded), 2)
        self.assertEqual(len(y_train_loaded), 2)
        self.assertTrue(np.isfinite(y_train_loaded.to_numpy(dtype=float)).all())

    def test_refinery_chronology_classification(self):
        blocked = ["filtercake_moisture", "filtercake_sugar", "sweetwater_brix", "sulphited_color", "standard_liquor_color"]
        allowed = ["sulphited_pH", "sulphited_brix", "standard_liquor_pH", "standard_liquor_brix", "boiler_pH", "boiling_r1_pol"]
        for name in blocked:
            self.assertEqual(classify_refinery_variable(name), "B_future_information_leakage")
            self.assertFalse(is_allowed_model_input(name, output_features=[]))
        for name in allowed:
            self.assertEqual(classify_refinery_variable(name), "A_safe_input")
            self.assertTrue(is_allowed_model_input(name, output_features=[]))


if __name__ == "__main__":
    unittest.main()
