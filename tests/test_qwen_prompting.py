from __future__ import annotations

import unittest

from modules.qwen_prompting import parse_llm_label_with_reason


class QwenPromptingTests(unittest.TestCase):
    def test_parser_accepts_only_allowed_labels(self) -> None:
        labels = ["Assignments", "Governing Laws"]
        self.assertEqual(parse_llm_label_with_reason("Governing Laws", labels), ("Governing Laws", ""))
        self.assertEqual(
            parse_llm_label_with_reason("The answer is Governing Laws.", labels),
            ("INVALID_PREDICTION", "allowed_label_with_extra_text"),
        )
        self.assertEqual(
            parse_llm_label_with_reason("Confidentiality", labels),
            ("INVALID_PREDICTION", "not_in_allowed_label_set"),
        )


if __name__ == "__main__":
    unittest.main()
