from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modules.classical_models import build_classical_pipeline
from modules.data_setup import ProjectPaths
from modules.preprocessing import (
    REQUIRED_SCHEMA,
    bpe_encode_word,
    clean_html_entities,
    corpus_word_frequencies,
    legal_safe_tokenise,
    negation_aware_tokenise,
    preprocess_ledgar,
    regex_tokenise,
    train_bpe_tokeniser,
)


class PreprocessingHelperTests(unittest.TestCase):
    def test_clean_html_entities(self) -> None:
        text = "The&nbsp;Borrower <b>shall</b> pay &amp; perform."
        self.assertEqual(clean_html_entities(text), "The Borrower shall pay & perform.")

    def test_regex_and_legal_safe_tokenisation(self) -> None:
        text = "Section 5.1: Borrower SHALL pay."
        self.assertEqual(regex_tokenise(text), ["Section", "5", "1", "Borrower", "SHALL", "pay"])
        self.assertEqual(legal_safe_tokenise(text), ["section", "5", "1", "borrower", "shall", "pay"])

    def test_negation_aware_tokenisation(self) -> None:
        text = "Borrower shall not be liable and never waive notice."
        tokens = negation_aware_tokenise(text)
        self.assertIn("not", tokens)
        self.assertIn("NOT_be", tokens)
        self.assertIn("never", tokens)
        self.assertIn("NOT_waive", tokens)

    def test_bpe_oov_fallback(self) -> None:
        word_counts = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
        merges, _ = train_bpe_tokeniser(word_counts, num_merges=10)
        self.assertTrue(merges)
        self.assertEqual(bpe_encode_word("xyz", merges), ["x", "y", "z"])

    def test_classical_pipeline_uses_lab_tokenizer(self) -> None:
        pipeline = build_classical_pipeline(
            "logistic_regression",
            max_features=100,
            ngram_range=(1, 2),
            min_df=1,
        )
        vectorizer = pipeline.named_steps["vectorizer"]
        features = vectorizer.fit_transform(["Borrower shall not be liable", "Borrower may assign rights"])
        names = vectorizer.get_feature_names_out()
        self.assertEqual(features.shape[0], 2)
        self.assertIn("NOT_be", set(names))
        self.assertIn("not NOT_be", set(names))

    def test_preprocess_ledgar_schema_is_preserved(self) -> None:
        raw_splits = {
            "train": pd.DataFrame(
                [
                    {"text": "Borrower shall not be liable.", "label": "Liability"},
                    {"text": "Party may terminate this Agreement.", "label": "Termination"},
                ]
            ),
            "validation": pd.DataFrame([{"text": "Borrower shall pay fees.", "label": "Liability"}]),
            "test": pd.DataFrame([{"text": "Party may terminate.", "label": "Termination"}]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ProjectPaths(
                project_root=root,
                raw_data_dir=root / "data" / "raw",
                processed_data_dir=root / "data" / "processed",
                ledgar_raw_dir=root / "data" / "raw" / "lexglue_ledgar",
                legacy_ledgar_raw_dir=root / "data" / "raw" / "original_ledgar",
                cuad_raw_dir=root / "data" / "raw" / "cuad",
                results_dir=root / "results",
                figures_dir=root / "figures",
            )
            processed, label2id, id2label = preprocess_ledgar(raw_splits, paths, top_k_labels=2)

        self.assertEqual(list(processed["train"].columns), REQUIRED_SCHEMA)
        self.assertEqual(set(label2id), {"Liability", "Termination"})
        self.assertEqual(set(id2label.values()), {"Liability", "Termination"})


class BpeCorpusTests(unittest.TestCase):
    def test_corpus_word_frequencies_uses_legal_safe_tokenise(self) -> None:
        counts = corpus_word_frequencies(["Borrower SHALL pay.", "Borrower may pay."])
        self.assertEqual(counts["borrower"], 2)
        self.assertEqual(counts["shall"], 1)


if __name__ == "__main__":
    unittest.main()
