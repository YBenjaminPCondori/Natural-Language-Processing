import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modules.cuad_external import (
    apply_cuad_label_mapping,
    convert_cuad_json_to_raw_dataframe,
    evaluate_sklearn_model_on_cuad,
    validate_cuad_against_ledgar_label_space,
)
from modules.data_setup import ProjectPaths


class FixedModel:
    def predict(self, texts):
        return np.array([0 for _ in texts])

    def predict_proba(self, texts):
        return np.array([[0.8, 0.2] for _ in texts])


def temp_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=root,
        raw_data_dir=root / "data" / "raw",
        processed_data_dir=root / "data" / "processed",
        ledgar_raw_dir=root / "data" / "raw" / "lexglue_ledgar",
        legacy_ledgar_raw_dir=root / "data" / "raw" / "ledgar",
        cuad_raw_dir=root / "data" / "raw" / "cuad",
        results_dir=root / "results",
        figures_dir=root / "results" / "figures",
    )


class CuadExternalTests(unittest.TestCase):
    def sample_cuad_json(self):
        return {
            "data": [
                {
                    "title": "contract_a.pdf",
                    "paragraphs": [
                        {
                            "qas": [
                                {
                                    "id": "a__Governing_Law",
                                    "question": "Highlight the parts related to Governing Law.",
                                    "answers": [
                                        {"text": "This agreement shall be governed by the laws of New York."},
                                        {"text": "This agreement shall be governed by the laws of New York."},
                                    ],
                                },
                                {
                                    "id": "a__Parties",
                                    "question": "Highlight the parties.",
                                    "answers": [{"text": "Acme"}],
                                },
                                {
                                    "id": "a__Insurance",
                                    "question": "Highlight the parts related to Insurance.",
                                    "answers": [{"text": "The supplier shall maintain commercial general liability insurance during the term."}],
                                },
                            ]
                        }
                    ],
                }
            ]
        }

    def test_conversion_mapping_and_compatibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = temp_paths(root)
            raw_df, stats = convert_cuad_json_to_raw_dataframe(self.sample_cuad_json(), min_span_chars=20, min_span_words=4)
            self.assertEqual(len(raw_df), 2)
            self.assertEqual(stats["duplicate_rows_removed"], 1)
            self.assertEqual(stats["short_spans_removed"], 1)

            mapped_df, report, _ = apply_cuad_label_mapping(raw_df, paths)
            self.assertEqual(len(mapped_df), 2)
            self.assertEqual(report["number_of_samples_retained_after_mapping"], 2)

            label2id = {"Governing Laws": 0, "Insurances": 1}
            eval_df, compatibility, _ = validate_cuad_against_ledgar_label_space(mapped_df, paths, label2id)
            self.assertEqual(len(eval_df), 2)
            self.assertEqual(compatibility["status"], "completed")
            self.assertEqual(eval_df["dataset_name"].unique().tolist(), ["CUAD_external"])

    def test_external_model_evaluation_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = temp_paths(root)
            cuad_df = pd.DataFrame(
                [
                    {
                        "text": "This agreement shall be governed by the laws of New York.",
                        "label_original": "Governing Law",
                        "label_mapped": "Governing Laws",
                        "label": "Governing Laws",
                        "label_id": 0,
                        "source_dataset": "CUAD",
                        "dataset_name": "CUAD_external",
                        "split": "external",
                        "contract_id": "contract_a.pdf",
                        "span_id": "cuad_0_0_0_0",
                    }
                ]
            )
            metrics = evaluate_sklearn_model_on_cuad(
                FixedModel(),
                model_name="fixed_model",
                cuad_df=cuad_df,
                id2label={0: "Governing Laws", 1: "Insurances"},
                paths=paths,
                notes="unit test",
            )
            self.assertEqual(metrics["status"], "completed")
            self.assertTrue((root / "outputs" / "predictions" / "cuad_external_fixed_model_predictions.csv").exists())
            self.assertTrue((root / "outputs" / "metrics" / "cuad_external_fixed_model_metrics.json").exists())
            payload = json.loads((root / "outputs" / "metrics" / "cuad_external_fixed_model_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_name"], "CUAD_external")


if __name__ == "__main__":
    unittest.main()
