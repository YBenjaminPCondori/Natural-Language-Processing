# Error Analysis Summary

- Error tables are derived from saved predictions and classification reports.
- Existing example-level reasons are kept as manual-review placeholders; no automatic legal explanation is invented.
- Linear SVM and DistilBERT are close in macro-F1, which should be preserved as a finding if supported by the final table.
- Likely discussion factors: boilerplate overlap, lexical ambiguity, long clauses/truncation, rare labels, semantically similar labels, and invalid LLM outputs.
