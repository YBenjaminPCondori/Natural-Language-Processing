import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modules.data_setup import ProjectPaths
from modules.report_exports import export_hpt_report_aliases, plot_hpt_validation_macro_f1, plot_pipeline_overview


class ReportExportTests(unittest.TestCase):
    def test_hpt_aliases_and_report_figures_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = ProjectPaths(
                project_root=root,
                raw_data_dir=root / "data" / "raw",
                processed_data_dir=root / "data" / "processed",
                ledgar_raw_dir=root / "data" / "raw" / "ledgar",
                legacy_ledgar_raw_dir=root / "data" / "raw" / "LEDGAR",
                cuad_raw_dir=root / "data" / "raw" / "cuad",
                results_dir=root / "results",
                figures_dir=root / "results" / "figures",
            )
            (root / "outputs").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "stage": "stage5a_random_trial",
                        "trial_number": 1,
                        "model_name": "distilbert-base-uncased",
                        "status": "completed",
                        "validation_macro_f1": 0.91,
                    }
                ]
            ).to_csv(root / "outputs" / "hyperparameter_search_results.csv", index=False)
            (root / "outputs" / "best_transformer_configs.json").write_text(
                json.dumps({"distilbert-base-uncased": {"learning_rate": 2e-5}}),
                encoding="utf-8",
            )

            aliases = export_hpt_report_aliases(paths)
            pipeline_figure = plot_pipeline_overview(root / "figures" / "pipeline_overview.png")
            hpt_figure = plot_hpt_validation_macro_f1(paths, root / "figures" / "hpt_validation_macro_f1.png")

            self.assertTrue(aliases["sweep_results"].exists())
            self.assertTrue(aliases["best_hyperparameters"].exists())
            self.assertTrue(aliases["hpt_summary"].exists())
            self.assertGreater(pipeline_figure.stat().st_size, 0)
            self.assertGreater(hpt_figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
